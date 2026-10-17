"""Per-client token bucket for `/chat` (plan.md Phase 14 step 6).

Hand-rolled rather than `slowapi`: one uvicorn process, no Redis, and the whole
mechanism is the refill formula below. A library would add a dependency and a
decorator signature on every limited route without changing the behavior
(design record 14f has the comparison).

    tokens(t) = min(capacity, tokens(t0) + (t - t0) * rate)

`rate` is tokens per second. A request is allowed when tokens >= 1, and then
costs 1. A refused request costs nothing, and is told how long until the next
whole token: (1 - tokens) / rate seconds.

State is in-process: it resets on restart and is not shared across workers.
That is the deployment this project has; a multi-worker deployment would need
a shared store.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable


class TokenBucketLimiter:
    def __init__(
        self,
        capacity: int,
        refill_per_minute: float,
        max_clients: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1 or refill_per_minute <= 0 or max_clients < 1:
            raise ValueError("capacity, refill_per_minute and max_clients must be positive")
        self.capacity = float(capacity)
        self.rate = refill_per_minute / 60.0
        self.max_clients = max_clients
        self._clock = clock
        # key -> (tokens, last_update). Ordered by last access, so the bound
        # evicts the client seen longest ago. An evicted client comes back with
        # a full bucket -- at most one extra burst, never a lockout.
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def acquire(self, key: str) -> tuple[bool, float]:
        """`(allowed, retry_after_s)`; `retry_after_s` is 0.0 when allowed."""
        with self._lock:
            now = self._clock()
            tokens, last = self._buckets.pop(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                allowed, retry_after = True, 0.0
                tokens -= 1.0
            else:
                allowed, retry_after = False, round((1.0 - tokens) / self.rate, 3)
            self._buckets[key] = (tokens, now)
            while len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)
            return allowed, retry_after

    def __len__(self) -> int:
        return len(self._buckets)
