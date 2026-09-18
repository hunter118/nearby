"""Expert labels from already-ended markets, independent of current-flow gates.

Current markets may never use a future final-N cutoff. Once a previous market's
label is known, a past pre-resolution tail is observable. This module builds
only that historical evidence, after an observed $1m volume admission. No model
is fitted, and no frozen source or dataset is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from build_expanded_replay import iter_selected_rows
from expand_market_universe import ROOT, START, END, read, dump, sha
from market_activation import settlement_arrays
from prepare_activation_study import trade_outcome

HISTORY_VERSION = 1
VARIANTS = ('all_after_admission', 'tail500', 'tail1000', 'tail2000', 'days7', 'days28')


def _second(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be an integer UTC second.')
    return int(value)


def volume_admission(times, sizes, token_prices, min_volume=1_000_000.):
    """Aggregate complete equal-second groups, admit at the following second.

    Uses observed actual-token dollars, not archived final volume or any future
    stopping time. A left-truncated sum is a lower bound, not lifetime volume.
    """
    if not math.isfinite(min_volume) or min_volume < 1_000_000:
        raise ValueError('User minimum cumulative admission is $1m.')
    times, sizes, prices = np.asarray(times), np.asarray(sizes), np.asarray(token_prices)
    if times.ndim != 1 or sizes.shape != times.shape or prices.shape != times.shape:
        raise ValueError('Unaligned volume arrays.')
    if not np.issubdtype(times.dtype, np.integer) or np.any(np.diff(times) < 0):
        raise ValueError('Chronological integer trade seconds are required.')
    if not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError('Trade quantities must be finite and positive.')
    if not np.isfinite(prices).all() or np.any((prices < 0) | (prices > 1)):
        raise ValueError('Actual-token prices must be finite in [0,1].')
    if not len(times):
        return None
    cumulative = np.cumsum(sizes*prices, dtype=np.float64)
    ends = np.r_[np.flatnonzero(times[:-1] != times[1:]), len(times)-1]
    crossed = ends[cumulative[ends] >= min_volume]
    if not len(crossed):
        return None
    index = int(crossed[0])
    return dict(time=int(times[index])+1, crossing_timestamp=int(times[index]),
        observed_cumulative_notional=float(cumulative[index]), threshold=float(min_volume),
        crossing_group_last_index=index, timing='Complete crossing second, then +1 second; trigger-second prints excluded.')


def build_histories(*, times, wallet, side, size, yes_price, token_price,
                    payout_yes, event_at, label_available_at, wallet_count,
                    observed_until, min_volume=1_000_000., admission=None):
    """Pure past-only arrays plus audit metadata; never consults future last print.

    `admission=None` selects the volume-only rule. An explicit mapping with
    `time=None` disables history; a later `time` supports joint-gate controls.
    Every tail is drawn from admission <= timestamp < event_at. Count tails use
    stable original row order; same-second rows are all already past at label
    availability. A tail is incremental trading contribution, not account P&L.
    """
    arrays = {k: np.asarray(v) for k, v in dict(time=times, wallet=wallet, side=side,
        size=size, yes_price=yes_price, token_price=token_price).items()}
    times = arrays['time']; n = len(times)
    if any(a.ndim != 1 or len(a) != n for a in arrays.values()):
        raise ValueError('Unaligned historical arrays.')
    if not isinstance(wallet_count, int) or wallet_count < 0:
        raise ValueError('Invalid wallet count.')
    observed_until = _second(observed_until, 'observed_until')
    if n and int(times[-1]) > observed_until:
        raise ValueError('Trade exceeds source observation bound.')
    if not np.issubdtype(arrays['wallet'].dtype, np.integer) or np.any((arrays['wallet'] < 0) | (arrays['wallet'] >= wallet_count)):
        raise ValueError('Invalid wallet indices.')
    if not np.issubdtype(arrays['side'].dtype, np.integer) or np.any((arrays['side'] < 0) | (arrays['side'] > 3)):
        raise ValueError('Invalid trade sides.')
    if not np.isfinite(arrays['yes_price']).all() or np.any((arrays['yes_price'] < 0) | (arrays['yes_price'] > 1)):
        raise ValueError('Invalid YES-equivalent prices.')
    expected = np.where((arrays['side'] == 0) | (arrays['side'] == 2), arrays['yes_price'], 1-arrays['yes_price'])
    if not np.allclose(expected, arrays['token_price'], rtol=0, atol=1e-12):
        raise ValueError('Actual-token and YES-equivalent price mapping disagree.')
    # Validate all volume rows but compute admission only from the known-ended
    # prefix, so appending post-resolution rows cannot alter historical metadata.
    volume_admission(times, arrays['size'], arrays['token_price'], min_volume)
    if event_at is not None:
        event_at = _second(event_at, 'event_at')
    if label_available_at is not None:
        label_available_at = _second(label_available_at, 'label_available_at')
        if event_at is None or label_available_at <= event_at:
            raise ValueError('Labels must become available at least one second after the resolution boundary.')
    prefix = n if event_at is None else int(np.searchsorted(times, event_at, side='left'))
    automatic = volume_admission(times[:prefix], arrays['size'][:prefix], arrays['token_price'][:prefix], min_volume)
    admitted = automatic if admission is None else dict(admission)
    if admitted is not None and admitted.get('time') is not None:
        admitted['time'] = _second(admitted['time'], 'admission time')
        if automatic is None or admitted['time'] < automatic['time']:
            raise ValueError('Explicit admission precedes verified $1m observed-volume admission.')
    if payout_yes is not None and (not math.isfinite(payout_yes) or not 0 <= payout_yes <= 1):
        raise ValueError('Invalid payout.')
    usable = (payout_yes is not None and event_at is not None
        and label_available_at is not None and label_available_at <= observed_until)
    active_at = None if admitted is None else admitted.get('time')
    base = (np.flatnonzero((times >= active_at) & (times < event_at))
        if usable and active_at is not None else np.array([], dtype=np.int64))
    output, variants = {}, {}
    for variant in VARIANTS:
        if variant.startswith('tail'):
            indices = base[-int(variant[4:]):]
        elif variant.startswith('days'):
            indices = base[times[base] >= event_at-int(variant[4:])*86400] if usable else base
        else:
            indices = base
        history = settlement_arrays(arrays['wallet'][indices], arrays['side'][indices],
            arrays['size'][indices], arrays['yes_price'][indices], payout_yes if usable else None, wallet_count)
        for key, values in history.items():
            output[f'{variant}_history_{key}'] = values
        variants[variant] = dict(selected_prints=len(indices), wallet_rows=len(history['wallet']),
            first_timestamp=int(times[indices[0]]) if len(indices) else None,
            last_timestamp=int(times[indices[-1]]) if len(indices) else None,
            source_indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest())
    audit = dict(admission=admitted, volume_only_admission=automatic, admission_time=active_at,
        min_cumulative_notional=min_volume, raw_resolution_boundary=event_at,
        label_available_at=label_available_at, payout_yes=payout_yes if usable else None,
        label_usable=usable, source_observed_until=observed_until,
        prints_before_boundary=prefix if event_at is not None else None,
        prints_at_or_after_boundary=n-prefix if event_at is not None else None,
        variants=variants, history_definition='Signed incremental trading P&L divided by absolute traded notional, clipped [-1,1]; no inferred opening inventory or total wallet profit.',
        availability='Use only at as_of >= label_available_at; every selected print is strictly before raw resolution boundary. Never infer the boundary from final print, END, scheduled close or updatedAt.')
    return output, audit


def read_corrected_tape(source, mid):
    """Read verified raw public rows; retain corrected token mapping and zeros."""
    source = Path(source)
    complete_path = source/'trades'/mid/'complete.json'
    complete = read(complete_path)
    if (complete.get('market_id') != mid or complete['status'] != 'api_window_exhausted'
            or (complete['start'], complete['end']) != (START, END)):
        raise ValueError('Raw window is incomplete or uses a different observation bound.')
    inherited_path = source/'normalized'/mid/'manifest.json'
    inherited = read(inherited_path); market = inherited['market']
    if (inherited.get('market_id') != mid or market.get('market_id') != mid
            or inherited.get('source_manifest_sha256') != sha(complete_path)):
        raise ValueError('Inherited token metadata does not bind the same raw market checkpoint.')
    for page in complete['pages']:
        page_path = (source/'trades'/mid/page['file']).resolve()
        if not page_path.is_relative_to((source/'trades'/mid).resolve()):
            raise ValueError('Raw page path escapes its market directory.')
    tokens = {str(token): i for i, token in enumerate(market['clob_token_ids'])}
    wallets, index, records, counts, quarantined = [], {}, [], Counter(), []
    for row in iter_selected_rows(source/'trades'/mid, complete):
        counts['source_rows'] += 1
        if not START <= int(row['timestamp']) <= END:
            raise ValueError('Raw print lies outside the bound observation window.')
        price, quantity = float(row['price']), float(row['size'])
        if not np.isfinite(price) or not 0 <= price <= 1:
            counts['invalid_price_quarantined'] += 1
            quarantined.append(dict(source_row_index=counts['source_rows']-1, row=row,
                reason='Nonfinite or out-of-range actual-token price.'))
            continue
        address = str(row.get('proxyWallet') or '')
        if not re.fullmatch(r'0x[0-9a-fA-F]{40}', address) or not np.isfinite(quantity) or quantity <= 0:
            raise ValueError('Invalid wallet or size.')
        outcome, corrected = trade_outcome(row, tokens, market['outcome_labels'])
        if outcome not in (0, 1) or row.get('side') not in ('BUY', 'SELL'):
            raise ValueError('Unmapped trade side.')
        counts['outcome_index_corrected'] += int(corrected)
        if address not in index:
            index[address] = len(wallets); wallets.append(address)
        records.append((int(row['timestamp']), index[address], outcome+(2 if row['side'] == 'SELL' else 0),
            quantity, price if outcome == 0 else 1-price, price))
    keys = ('times', 'wallet', 'side', 'size', 'yes_price', 'token_price')
    dtypes = (np.int64, np.uint32, np.uint8, np.float64, np.float64, np.float64)
    arrays = {k: np.array([row[i] for row in records], dtype=dtypes[i]) for i, k in enumerate(keys)}
    if counts['source_rows'] != complete['rows']:
        raise ValueError('Raw row count differs from bound checkpoint.')
    if np.any(np.diff(arrays['times']) < 0):
        raise ValueError('Verified raw iterator is not chronologically ordered.')
    return arrays, np.array(wallets, dtype='U42'), dict(counters=dict(counts),
        raw_complete_sha256=sha(complete_path), inherited_manifest_sha256=sha(inherited_path),
        source_market=market, observation_start=complete['start'], observation_end=complete['end'],
        full_valid_prints=len(records), stable_chronological_order=True,
        quarantine_sha256=hashlib.sha256(json.dumps(quarantined, sort_keys=True).encode()).hexdigest(),
        quarantined_price_rows=quarantined)


def prepare_market(source, output, mid, raw_metadata, *, allow_metadata_proxy=False,
                   evidence=None, metadata_observed_at=None, admission=None, min_volume=1_000_000.):
    """Write one independent sidecar; missing/unapproved evidence stays explicit."""
    from resolution_evidence import metadata_resolution_evidence, history_boundary
    source, output = Path(source).resolve(), Path(output).resolve()
    if (source == output or source in output.parents or output in source.parents
            or not output.is_relative_to(ROOT/'artifacts') or output == ROOT/'artifacts'):
        raise ValueError('Use a dedicated artifact output separate from raw source.')
    folder = output/'normalized'/mid
    if (not re.fullmatch(r'0x[0-9a-fA-F]{64}', mid)
            or (output/'normalized').is_symlink() or folder.is_symlink()
            or not folder.resolve().is_relative_to(output)):
        raise ValueError('Invalid market output path or symlink.')
    if any((folder/name).is_symlink() for name in ('data.npz','data.tmp.npz','manifest.json')):
        raise ValueError('Refusing a symlinked sidecar output file.')
    if evidence is None:
        if metadata_observed_at is None:
            raise ValueError('Explicit present metadata observation time is required; never infer historical observation from file mtime.')
        evidence = metadata_resolution_evidence(raw_metadata,
            metadata_observed_at=metadata_observed_at, expected_market_id=mid)
    if evidence.market_id.lower() != mid.lower():
        raise ValueError('Resolution evidence belongs to another market.')
    bound = history_boundary(evidence, as_of=END, allow_metadata_proxy=allow_metadata_proxy)
    tape, wallets, provenance = read_corrected_tape(source, mid)
    arrays, audit = build_histories(**tape, wallet_count=len(wallets),
        payout_yes=bound['payout_yes'] if bound else None,
        event_at=bound['history_end_exclusive'] if bound else evidence.event_at,
        label_available_at=bound['label_available_at'] if bound else None,
        observed_until=END, min_volume=min_volume, admission=admission)
    arrays['wallets'] = wallets
    # Full corrected tape is deliberately NOT screened on archived deadlines.
    # This permits a later deadline-free strategy without parsing raw files twice.
    for key, source_key in (('time','times'), ('wallet','wallet'), ('side','side'),
                            ('price','yes_price'), ('size','size'), ('token_price','token_price')):
        arrays['trade_'+key] = tape[source_key]
    evidence_dict = json.loads(json.dumps(asdict(evidence)))
    binding = dict(history_version=HISTORY_VERSION, market_id=mid, source=str(source),
        source_files_sha256={name:sha(ROOT/name) for name in (
            'src/prepare_causal_expert_history.py','src/resolution_evidence.py',
            'src/market_activation.py','src/build_expanded_replay.py',
            'src/prepare_activation_study.py','src/expand_market_universe.py')},
        raw_metadata_sha256=hashlib.sha256(json.dumps(raw_metadata, sort_keys=True).encode()).hexdigest(),
        evidence=evidence_dict, evidence_tier=evidence.tier, allow_metadata_proxy=allow_metadata_proxy,
        history_boundary_policy='Strict event timestamp cut; labels >= event+1 second. Metadata proxy opt-in does not meet independent finality verification.',
        admission_override=admission, min_cumulative_notional=min_volume, provenance=provenance)
    if (folder/'manifest.json').exists():
        previous = read(folder/'manifest.json')
        if previous['binding'] != binding or sha(folder/'data.npz') != previous['array_sha256']:
            raise ValueError('Existing causal-history sidecar has different input/source binding.')
        return previous
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder/'data.tmp.npz'
    np.savez_compressed(temporary, **arrays); temporary.replace(folder/'data.npz')
    manifest = dict(binding=binding, market_id=mid, wallets=len(wallets), **audit,
        full_valid_prints=len(tape['times']), trade_tape='All valid source-window prints, without scheduled-deadline or activation filtering.',
        evidence=evidence_dict, evidence_tier=evidence.tier, array_sha256=sha(folder/'data.npz'))
    dump(folder/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--market-ids', nargs='+', required=True, help='Explicit small-sample IDs; no portfolio replay.')
    parser.add_argument('--allow-metadata-proxy', action='store_true')
    parser.add_argument('--metadata-observed-at', type=int, required=True,
        help='Actual present observation of archived metadata; never substituted for historical availability.')
    parser.add_argument('--min-volume', type=float, default=1_000_000.)
    args = parser.parse_args()
    metadata = read(args.source/'metadata.json.gz')
    cohort = {r['market_id'] for r in read(args.source/'cohort.json')}
    if len(set(args.market_ids)) != len(args.market_ids) or not set(args.market_ids) <= cohort:
        raise ValueError('Select unique IDs in the supplied inventory.')
    records = []
    for mid in args.market_ids:
        manifest = prepare_market(args.source, args.output, mid, metadata[mid],
            allow_metadata_proxy=args.allow_metadata_proxy, metadata_observed_at=args.metadata_observed_at,
            min_volume=args.min_volume)
        records.append(dict(market_id=mid, array_sha256=manifest['array_sha256'],
            manifest_sha256=sha(args.output/'normalized'/mid/'manifest.json')))
        print(mid, manifest['evidence_tier'], {k:v['wallet_rows'] for k,v in manifest['variants'].items()}, flush=True)
    dump(args.output/'sample_manifest.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        source=str(args.source.resolve()), records=records, allow_metadata_proxy=args.allow_metadata_proxy,
        metadata_observed_at=args.metadata_observed_at,
        min_cumulative_notional=args.min_volume, source_sha256=sha(Path(__file__))))


if __name__ == '__main__':
    main()
