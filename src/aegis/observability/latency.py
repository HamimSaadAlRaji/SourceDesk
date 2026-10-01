"""Per-stage latency aggregation (plan.md Phase 13 build item 3): turns a
batch of `PipelineResult.stage_latencies_ms` dicts into p50/p95/p99 per stage.

Percentiles use numpy's default linear interpolation. With small batches
p99 collapses toward the max -- callers report `n` next to every percentile
so that isn't mistaken for a tail measurement.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StageStats:
    n: int
    mean: float
    p50: float
    p95: float
    p99: float
    max: float


def stage_latency_summary(rows: list[dict[str, float]]) -> dict[str, StageStats]:
    by_stage: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for stage, ms in row.items():
            by_stage[stage].append(ms)

    summary: dict[str, StageStats] = {}
    for stage, values in by_stage.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[stage] = StageStats(
            n=len(values),
            mean=float(arr.mean()),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            max=float(arr.max()),
        )
    return summary
