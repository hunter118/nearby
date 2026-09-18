"""Explicit final-label evidence for NEW causal-history experiments.

This module never reads a trade tape, its final print, or a replay end date.
Gamma fields are only retrospective metadata proxies. Strict callers must
supply independently checked ConditionResolution receipts/block timestamps and
token-slot mapping; accepting that evidence does not verify an RPC response or
certify the rest of a historical market universe.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import math
import re
from typing import Any, Literal

Tier = Literal['unusable', 'metadata_proxy', 'chain_event_reconstructed', 'chain_event_observed']
RESOLUTION_DATE_FIELDS = ('resolvedAt', 'umaEndDate', 'closedTime', 'resolveDate')
# Polymarket/contract-security README, independently checked 2026-09-18.
POLYGON_CTF = '0x4d97dcd97ec945f40cf65f87097ace5ea0476045'


def _epoch(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer UTC epoch second')
    return value


def _timestamp(value: Any) -> int | None:
    if value in (None, ''):
        return None
    if not isinstance(value, str):
        raise ValueError('Metadata dates must be explicit ISO date strings')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    # Gamma also uses unzoned strings; preserve the documented UTC assumption.
    parsed = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    if parsed.microsecond:
        # Do not round a subsecond evidence timestamp backwards.
        return math.ceil(parsed.timestamp())
    return int(parsed.timestamp())


def _json_list(value: Any) -> list:
    parsed = json.loads(value) if isinstance(value, str) else value
    return parsed if isinstance(parsed, list) else []


def _hex(value: str, length: int, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch('0x[0-9a-fA-F]{'+str(length)+'}', value):
        raise ValueError(f'Invalid {name}')
    return value.lower()


@dataclass(frozen=True)
class ResolutionEvidence:
    market_id: str
    payout_yes: float | None
    event_at: int | None
    label_available_at: int | None
    tier: Tier
    provenance: tuple[str, ...]
    issues: tuple[str, ...] = ()
    metadata_observed_at: int | None = None
    raw_date_candidates: tuple[tuple[str, int], ...] = ()
    uma_status: str | None = None

    def __post_init__(self):
        if not self.market_id or self.tier not in ('unusable','metadata_proxy','chain_event_reconstructed','chain_event_observed'):
            raise ValueError('Invalid evidence identity or tier')
        for key in ('event_at','label_available_at','metadata_observed_at'):
            value = getattr(self,key)
            if value is not None:
                _epoch(value,key)
        if not self.provenance:
            raise ValueError('Evidence provenance must be explicit')
        if self.tier == 'unusable':
            if self.payout_yes is not None or self.label_available_at is not None:
                raise ValueError('Unusable evidence cannot release a payout label')
        else:
            if self.payout_yes is None or not math.isfinite(self.payout_yes) or not 0 <= self.payout_yes <= 1:
                raise ValueError('A usable payout must lie in [0,1]')
            if self.event_at is None or self.label_available_at is None or self.label_available_at <= self.event_at:
                raise ValueError('Labels must be released strictly after the resolution event second')
        if self.tier == 'chain_event_observed':
            if self.metadata_observed_at is None or self.label_available_at <= self.metadata_observed_at:
                raise ValueError('Observed chain evidence must wait until after actual receipt')

    def as_dict(self) -> dict:
        return asdict(self)


def metadata_resolution_evidence(raw: dict, *, metadata_observed_at: int,
                                 expected_market_id: str | None = None) -> ResolutionEvidence:
    """Inspect current Gamma, never promote it into strict historical evidence.

    The explicit proxy convention is max(four claimed resolution dates)+1 sec,
    irrespective of today's download time. That is a declared reconstruction,
    NOT an assertion that the label was observed then. Proposed/disputed/missing
    UMA state, missing dates or ambiguous current prices release no label.
    """
    observed = _epoch(metadata_observed_at,'metadata_observed_at')
    mid = str(raw.get('conditionId') or '')
    if not mid or (expected_market_id is not None and mid.lower()!=expected_market_id.lower()):
        raise ValueError('Gamma condition identity does not match the requested market')
    candidates, issues = [], []
    for key in RESOLUTION_DATE_FIELDS:
        try:
            value = _timestamp(raw.get(key))
        except (TypeError, ValueError, OverflowError):
            issues.append(f'invalid_resolution_date:{key}')
            continue
        if value is not None:
            candidates.append((key,value))
    event = max((v for _,v in candidates),default=None)
    uma = raw.get('umaResolutionStatus') or None
    if raw.get('closed') is not True:
        issues.append('not_closed')
    if uma != 'resolved':
        issues.append('uma_not_resolved')
    if event is None:
        issues.append('no_explicit_resolution_date')
    if event is not None and event > observed:
        issues.append('claimed_event_after_metadata_observation')
    payout = None
    try:
        labels = _json_list(raw.get('outcomes'))
        prices = [float(v) for v in _json_list(raw.get('outcomePrices'))]
        if len(labels)!=2 or len(set(labels))!=2 or len(prices)!=2 or not all(math.isfinite(p) and 0<=p<=1 for p in prices):
            issues.append('ambiguous_binary_structure_or_prices')
        elif abs(prices[0]-.5)<1e-12 and abs(prices[1]-.5)<1e-12:
            payout = .5
        elif prices[0]>=.99 and prices[1]<=.01:
            payout = 1.
        elif prices[1]>=.99 and prices[0]<=.01:
            payout = 0.
        else:
            issues.append('no_unambiguous_terminal_price_proxy')
    except (TypeError, ValueError, OverflowError):
        issues.append('invalid_prices_or_outcomes')
    if issues or payout is None:
        return ResolutionEvidence(mid,None,event,None,'unusable',
            ('current Gamma metadata; not a contemporaneous resolution observation',),tuple(issues),observed,tuple(candidates),uma)
    return ResolutionEvidence(mid,payout,event,event+1,'metadata_proxy',
        ('current Gamma closed+UMA-resolved+outcome-price proxy',
         'claimed resolution timestamp reconstructed; not independently proved historical availability'),
        ('current_metadata_not_historical_observation','terminal_price_not_onchain_payout'),observed,tuple(candidates),uma)


def condition_resolution_evidence(*, market_id: str, condition_id: str,
        payout_numerators: list[int] | tuple[int, int], first_outcome_slot: int,
        event_at: int, finality_at: int, chain_id: int, contract_address: str,
        transaction_hash: str, block_hash: str, log_index: int,
        receipt_verified: bool, token_mapping_verified: bool,
        receipt_sha256: str, block_sha256: str, mapping_sha256: str,
        observed_at: int | None = None) -> ResolutionEvidence:
    """Convert independently verified CTF evidence, without contacting a chain.

    Caller must check the canonical successful transaction receipt, decoded
    ConditionResolution log/contract, block timestamp/finality and condition-
    token slot mapping. Hashes bind those saved proofs; booleans are explicit
    attestations, not cryptographic verification performed by this helper.
    `finality_at` is a confirmation/block-evidence time, never a last-trade date.
    Without actual historical receipt logs the tier remains reconstructed.
    """
    mid = _hex(market_id,64,'market_id')
    condition = _hex(condition_id,64,'condition_id')
    if mid!=condition or chain_id!=137 or _hex(contract_address,40,'contract_address')!=POLYGON_CTF:
        raise ValueError('Unmatched condition, chain or supported CTF deployment')
    if receipt_verified is not True or token_mapping_verified is not True:
        raise ValueError('Unverified receipt or token-slot mapping cannot establish finality')
    event = _epoch(event_at,'event_at'); finality = _epoch(finality_at,'finality_at')
    if finality < event:
        raise ValueError('Finality cannot precede its event')
    _epoch(log_index,'log_index')
    tx = _hex(transaction_hash,64,'transaction_hash'); block = _hex(block_hash,64,'block_hash')
    for value in (receipt_sha256,block_sha256,mapping_sha256):
        if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
            raise ValueError('Saved receipt/block/mapping SHA256 bindings are required')
    if (len(payout_numerators)!=2 or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in payout_numerators)
            or sum(payout_numerators)<=0 or isinstance(first_outcome_slot,bool) or first_outcome_slot not in (0,1)):
        raise ValueError('Expected a positive binary payout vector and verified first-token slot')
    payout = payout_numerators[first_outcome_slot]/sum(payout_numerators)
    available = finality+1
    issues = ('historical_chain_publication_reconstruction_not_actual_observer_log',)
    tier: Tier = 'chain_event_reconstructed'
    if observed_at is not None:
        observed_at = _epoch(observed_at,'observed_at')
        if observed_at < event:
            raise ValueError('Actual receipt cannot precede the emitted event')
        available = max(finality,observed_at)+1
        tier,issues = 'chain_event_observed',()
    provenance = (f'Polygon CTF ConditionResolution:{condition}',f'transaction:{tx}:log:{log_index}',
        f'block:{block}',f'receipt_sha256:{receipt_sha256}',f'block_sha256:{block_sha256}',
        f'token_mapping_sha256:{mapping_sha256}',f'first_outcome_slot:{first_outcome_slot}',
        f'finality_at:{finality}')
    return ResolutionEvidence(mid,payout,event,available,tier,provenance,issues,observed_at)


def history_boundary(evidence: ResolutionEvidence, *, as_of: int,
                     allow_metadata_proxy: bool = False,
                     require_observed: bool = False) -> dict | None:
    """Fail closed unless the declared evidence tier is allowed and now known.

    `history_end_exclusive` excludes the event's entire timestamp-second. It
    is never expanded when label release is delayed by confirmations/receipt.
    """
    as_of = _epoch(as_of,'as_of')
    if evidence.tier=='unusable' or evidence.label_available_at is None:
        return None
    if evidence.tier=='metadata_proxy' and not allow_metadata_proxy:
        return None
    if require_observed and evidence.tier!='chain_event_observed':
        return None
    if as_of < evidence.label_available_at:
        return None
    return dict(market_id=evidence.market_id,payout_yes=evidence.payout_yes,
        event_at=evidence.event_at,history_end_exclusive=evidence.event_at,
        label_available_at=evidence.label_available_at,tier=evidence.tier,
        historical_availability_certified=evidence.tier=='chain_event_observed',
        issues=list(evidence.issues))
