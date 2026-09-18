from audit_causal_portfolios import cash_path, epoch, validate_execution_limits, validate_parent_limits
import pytest


def test_resolution_cash_precedes_same_second_fill():
    stamp='2025-01-01 01:00:00'
    fills=[dict(market_id='a',filled_at=stamp,notional=15.,fee=0.)]
    closed=[dict(market_id='z',resolved_at=stamp,payout=10.)]
    assert cash_path(10.,fills,closed)==(5.,5.)


def test_borrowing_not_hidden_by_later_resolution():
    fills=[dict(market_id='a',filled_at='2025-01-01 00:00:00',notional=15.,fee=0.)]
    closed=[dict(market_id='z',resolved_at='2025-01-01 01:00:00',payout=10.)]
    with pytest.raises(AssertionError,match='Negative'):
        cash_path(10.,fills,closed)


def test_naive_and_explicit_utc_timestamps_agree():
    assert epoch('2025-01-01 00:00:00')==epoch('2025-01-01T00:00:00+00:00')


def child():
    return dict(quantity=25.,source_trade_size=100.,source_trade_side='BUY_NO',direction='NO',
        signal_time='2025-01-01 00:00:00',filled_at='2025-01-02 00:05:00',
        signal_token_price=.8,fill_price=.808,notional=20.2,
        order_requested_notional=100.,child_fill_index=1)


def execution():
    return dict(max_fill_participation=.25,execution_trade_filter='target_token_buy',
        delay_seconds=300,pending_order_expiry_seconds=86400,max_price_deterioration_bps=100,
        execution_one_order_per_market=True,execution_partial_fill=True,
        execution_max_child_fills=1000,execution_slices=1)


def test_exact_execution_boundaries_are_allowed():
    validate_execution_limits(child(),execution())
    validate_parent_limits([child()],execution())


@pytest.mark.parametrize('change,message',[
    ({'quantity':25.1},'participation'),
    ({'source_trade_side':'BUY_YES'},'execution token'),
    ({'source_trade_side':'SELL_NO'},'taker side'),
    ({'filled_at':'2025-01-02 00:05:01'},'expiry'),
    ({'fill_price':.809},'relative price'),
])
def test_invalid_execution_is_detected(change,message):
    with pytest.raises(AssertionError,match=message):
        validate_execution_limits(child()|change,execution())


@pytest.mark.parametrize('change,message',[
    ({'signal_time':'2025-01-01 00:01:00'},'parent signal'),
    ({'order_requested_notional':101.},'budget changed'),
    ({'notional':90.},'parent budget'),
    ({'child_fill_index':3},'contiguous'),
])
def test_invalid_parent_aggregation_is_detected(change,message):
    second=child()|{'child_fill_index':2}|change
    with pytest.raises(AssertionError,match=message):
        validate_parent_limits([child(),second],execution())


def test_parent_child_count_cap_is_checked():
    with pytest.raises(AssertionError,match='Too many'):
        validate_parent_limits([child(),child()|{'child_fill_index':2}],
            execution()|{'execution_max_child_fills':1})
