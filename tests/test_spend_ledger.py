"""TDD for the paid-API spend ledger (Phase 15 step 0, design record 15g).

The user's hard cap is $10 for the whole project. A silent bug here either
overspends real money or blocks every call, so the money arithmetic and the
guard are pinned: cost from billed usage, worst-case reservation before a
call, refusal before the provider is contacted, concurrency, persistence.
"""

from __future__ import annotations

import threading

import pytest

from aegis.cost.ledger import BudgetExceededError, SpendLedger

PRICES = {"m": {"input": 0.40, "cached_input": 0.10, "output": 1.60}}  # USD per 1M tokens


def _ledger(tmp_path, budget=10.0):
    return SpendLedger(tmp_path / "spend.sqlite", budget_usd=budget, prices=PRICES)


def test_cost_uses_billed_usage_and_discounts_cached_input(tmp_path):
    led = _ledger(tmp_path)
    # 1M prompt tokens of which 250k cached, 100k output
    cost = led.cost("m", input_tokens=1_000_000, cached_input_tokens=250_000, output_tokens=100_000)
    assert cost == pytest.approx(0.75 * 0.40 + 0.25 * 0.10 + 0.1 * 1.60)


def test_unknown_model_is_refused_not_priced_at_zero(tmp_path):
    with pytest.raises(KeyError):
        _ledger(tmp_path).cost("gpt-unknown", input_tokens=1, cached_input_tokens=0, output_tokens=1)


def test_reserve_settle_records_actual_spend(tmp_path):
    led = _ledger(tmp_path)
    r = led.reserve("m", input_tokens=1000, max_output_tokens=500, purpose="test")
    led.settle(r, input_tokens=900, cached_input_tokens=0, output_tokens=200)
    assert led.spent_usd() == pytest.approx(led.cost("m", 900, 0, 200))
    assert led.reserved_usd() == 0.0


def test_guard_refuses_a_call_whose_worst_case_would_cross_the_budget(tmp_path):
    led = _ledger(tmp_path, budget=0.001)
    with pytest.raises(BudgetExceededError):
        led.reserve("m", input_tokens=1000, max_output_tokens=1000, purpose="test")  # worst case 0.0020 > 0.001
    assert led.spent_usd() == 0.0


def test_reservations_count_against_the_budget_until_settled(tmp_path):
    led = _ledger(tmp_path, budget=0.003)
    led.reserve("m", input_tokens=1000, max_output_tokens=1000, purpose="a")  # 0.0020 reserved
    with pytest.raises(BudgetExceededError):
        led.reserve("m", input_tokens=1000, max_output_tokens=1000, purpose="b")


def test_release_frees_a_failed_calls_reservation(tmp_path):
    led = _ledger(tmp_path, budget=0.003)
    r = led.reserve("m", input_tokens=1000, max_output_tokens=1000, purpose="a")
    led.release(r)
    led.reserve("m", input_tokens=1000, max_output_tokens=1000, purpose="b")  # fits again


def test_concurrent_reservations_never_exceed_the_budget(tmp_path):
    led = _ledger(tmp_path, budget=0.010)  # room for exactly 5 worst-case calls of 0.0020
    ok: list[int] = []
    lock = threading.Lock()

    def worker():
        try:
            led.reserve("m", input_tokens=1000, max_output_tokens=1000, purpose="c")
            with lock:
                ok.append(1)
        except BudgetExceededError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ok) == 5


def test_spend_persists_across_processes(tmp_path):
    led = _ledger(tmp_path)
    r = led.reserve("m", input_tokens=1000, max_output_tokens=10, purpose="p")
    led.settle(r, input_tokens=1000, cached_input_tokens=0, output_tokens=10)
    assert _ledger(tmp_path).spent_usd() == pytest.approx(led.spent_usd())


def test_summary_breaks_spend_down_by_purpose_and_model(tmp_path):
    led = _ledger(tmp_path)
    for purpose in ("router", "router", "generation"):
        r = led.reserve("m", input_tokens=100, max_output_tokens=10, purpose=purpose)
        led.settle(r, input_tokens=100, cached_input_tokens=0, output_tokens=10)
    s = led.summary()
    assert s["calls"] == 3 and set(s["by_purpose"]) == {"router", "generation"}
    assert s["remaining_usd"] == pytest.approx(10.0 - led.spent_usd())
