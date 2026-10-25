"""Host provenance for gate runs (2026-10-24). Parsing is the part that can silently
go wrong: a missing or odd /proc/cpuinfo must give "unknown", never crash a gate run."""

from __future__ import annotations

from aegis.eval.retrieval_gate import parse_cpuinfo

EPYC = """processor\t: 0
vendor_id\t: AuthenticAMD
model name\t: AMD EPYC 7763 64-Core Processor
flags\t\t: fpu sse sse2 avx avx2 fma bmi2
processor\t: 1
model name\t: AMD EPYC 7763 64-Core Processor
"""

XEON = """model name\t: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
flags\t\t: fpu avx avx2 avx512f avx512dq avx512bw
"""


def test_model_and_simd_levels_are_extracted():
    model, simd = parse_cpuinfo(EPYC)
    assert model == "AMD EPYC 7763 64-Core Processor"
    assert simd == "avx,avx2"


def test_avx512_is_reported_when_present():
    _, simd = parse_cpuinfo(XEON)
    assert simd == "avx,avx2,avx512f"


def test_empty_or_unexpected_text_is_unknown_not_an_error():
    assert parse_cpuinfo("") == ("unknown", "")
    assert parse_cpuinfo("gibberish with no colon lines") == ("unknown", "")
