"""TDD for the /chat rate limiter (Phase 14 build item 6, design record 14f).
A token bucket per client key; the clock is injected so refill is tested
exactly, without sleeping."""

import threading

from aegis.api.ratelimit import TokenBucketLimiter


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _limiter(capacity=3, per_minute=6.0, max_clients=100) -> tuple[TokenBucketLimiter, _Clock]:
    clock = _Clock()
    return TokenBucketLimiter(capacity, per_minute, max_clients, clock=clock), clock


def test_allows_a_burst_up_to_capacity_then_refuses():
    limiter, _ = _limiter(capacity=3)

    results = [limiter.acquire("a")[0] for _ in range(4)]

    assert results == [True, True, True, False]


def test_refusal_reports_seconds_until_the_next_token():
    limiter, _ = _limiter(capacity=1, per_minute=6.0)  # one token per 10 s
    limiter.acquire("a")

    allowed, retry_after = limiter.acquire("a")

    assert allowed is False and retry_after == 10.0


def test_tokens_refill_with_elapsed_time():
    limiter, clock = _limiter(capacity=1, per_minute=6.0)
    limiter.acquire("a")

    clock.now = 9.9
    assert limiter.acquire("a")[0] is False
    clock.now = 10.0  # the refused call at 9.9 s must not have consumed the partial token
    assert limiter.acquire("a")[0] is True


def test_refill_never_exceeds_capacity():
    limiter, clock = _limiter(capacity=2, per_minute=6.0)
    clock.now = 3600.0  # an hour idle

    results = [limiter.acquire("a")[0] for _ in range(3)]

    assert results == [True, True, False]


def test_clients_have_independent_buckets():
    limiter, _ = _limiter(capacity=1)
    limiter.acquire("a")

    assert limiter.acquire("a")[0] is False
    assert limiter.acquire("b")[0] is True


def test_tracked_clients_are_bounded_least_recently_seen_evicted_first():
    limiter, _ = _limiter(capacity=1, max_clients=2)
    limiter.acquire("a")
    limiter.acquire("b")
    limiter.acquire("c")  # evicts "a", the least recently seen

    assert len(limiter) == 2
    assert limiter.acquire("a")[0] is True  # evicted -> starts again with a full bucket
    assert limiter.acquire("c")[0] is False  # still tracked, still empty


def test_concurrent_acquires_never_grant_more_than_capacity():
    limiter, _ = _limiter(capacity=50)
    granted: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(20):
            ok = limiter.acquire("a")[0]
            with lock:
                granted.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(granted) == 50
