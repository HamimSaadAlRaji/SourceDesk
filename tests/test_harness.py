"""TDD for the eval harness (plan.md Phase 6, step 2): config in, metrics
table out, cached and reproducible from a config hash, reported per-stratum.

`run_eval` takes a `search_fn` callback rather than a real retriever, so
the composition/caching/per-stratum logic here is fully unit-tested with a
fake -- the same treatment `tests/test_hybrid.py` gives `HybridRetriever`.
The real retrieval path (`HybridRetriever` + live Qdrant + `bm25s`) is
already tested/verified in Phase 5; this harness is what calls it from
`experiments/06_retrieval_ablation.py`.
"""

from aegis.eval.harness import RetrievalConfig, load_cached, run_eval
from aegis.types import GoldenQuery, Stratum


def _query(id: str, stratum: Stratum, ground_truth: list[str]) -> GoldenQuery:
    return GoldenQuery(id=id, text=f"question {id}", is_negative=False, stratum=stratum, ground_truth_chunk_ids=ground_truth)


def _negative(id: str) -> GoldenQuery:
    return GoldenQuery(id=id, text=f"negative {id}", is_negative=True, negative_category=None)


def _config(**overrides) -> RetrievalConfig:
    defaults = dict(
        name="test-config",
        chunking_strategy="heading_aware",
        contextual_enrichment=False,
        retrieval_strategy="hybrid_rrf",
        encoder="bge_large",
    )
    defaults.update(overrides)
    return RetrievalConfig(**defaults)


def test_run_eval_scores_a_perfect_retriever_as_recall_1():
    queries = [_query("q1", Stratum.HOW_TO, ["c1"]), _query("q2", Stratum.HOW_TO, ["c2"])]
    search_fn = lambda q: [(q.ground_truth_chunk_ids[0], 1.0)]  # noqa: E731

    result = run_eval(_config(), queries, search_fn, metrics=["recall@10"], cache_dir=None)

    assert result.overall["recall@10"] == 1.0
    assert result.n_queries == 2


def test_run_eval_scores_a_useless_retriever_as_recall_0():
    queries = [_query("q1", Stratum.HOW_TO, ["c1"])]
    search_fn = lambda q: [("totally_wrong_chunk", 1.0)]  # noqa: E731

    result = run_eval(_config(), queries, search_fn, metrics=["recall@10"], cache_dir=None)

    assert result.overall["recall@10"] == 0.0


def test_run_eval_breaks_down_by_stratum():
    queries = [
        _query("q1", Stratum.HOW_TO, ["c1"]),
        _query("q2", Stratum.ERROR_DEBUGGING, ["c2"]),
    ]

    def search_fn(q):
        # Perfect for how_to, wrong for error_debugging.
        return [(q.ground_truth_chunk_ids[0], 1.0)] if q.stratum == Stratum.HOW_TO else [("wrong", 1.0)]

    result = run_eval(_config(), queries, search_fn, metrics=["recall@10"], cache_dir=None)

    assert result.per_stratum["how_to"]["recall@10"] == 1.0
    assert result.per_stratum["error_debugging"]["recall@10"] == 0.0


def test_run_eval_excludes_negatives_from_retrieval_metrics():
    queries = [_query("q1", Stratum.HOW_TO, ["c1"]), _negative("n1")]
    calls = []

    def search_fn(q):
        calls.append(q.id)
        return [(q.ground_truth_chunk_ids[0], 1.0)] if not q.is_negative else []

    result = run_eval(_config(), queries, search_fn, metrics=["recall@10"], cache_dir=None)

    assert result.n_queries == 1
    assert "n1" not in calls


def test_run_eval_caches_by_config_hash(tmp_path):
    queries = [_query("q1", Stratum.HOW_TO, ["c1"])]
    calls = []

    def search_fn(q):
        calls.append(q.id)
        return [(q.ground_truth_chunk_ids[0], 1.0)]

    config = _config()
    run_eval(config, queries, search_fn, metrics=["recall@10"], cache_dir=tmp_path)
    run_eval(config, queries, search_fn, metrics=["recall@10"], cache_dir=tmp_path)

    assert calls == ["q1"]  # second call served from cache, search_fn not re-invoked


def test_run_eval_cache_is_scoped_to_the_evaluated_query_set(tmp_path):
    # A cache keyed only on config_hash would collide across different
    # query sets evaluated under the same config -- the second call below
    # would silently return the first call's (wrong) cached result.
    config = _config()
    search_fn = lambda q: [("c1", 1.0)]  # always retrieves c1, regardless of query  # noqa: E731

    queries_hit = [_query("q1", Stratum.HOW_TO, ["c1"])]  # ground truth is c1 -> recall 1.0
    queries_miss = [_query("q2", Stratum.HOW_TO, ["c2"])]  # ground truth is c2 -> recall 0.0

    result_hit = run_eval(config, queries_hit, search_fn, metrics=["recall@10"], cache_dir=tmp_path)
    result_miss = run_eval(config, queries_miss, search_fn, metrics=["recall@10"], cache_dir=tmp_path)

    assert result_hit.overall["recall@10"] == 1.0
    assert result_miss.overall["recall@10"] == 0.0


def test_different_configs_do_not_share_a_cache_entry(tmp_path):
    queries = [_query("q1", Stratum.HOW_TO, ["c1"])]
    search_fn = lambda q: [(q.ground_truth_chunk_ids[0], 1.0)]  # noqa: E731

    result_a = run_eval(_config(encoder="bge_large"), queries, search_fn, metrics=["recall@10"], cache_dir=tmp_path)
    result_b = run_eval(_config(encoder="bge_m3"), queries, search_fn, metrics=["recall@10"], cache_dir=tmp_path)

    assert result_a.config_hash != result_b.config_hash


def test_config_hash_ignores_the_human_readable_name():
    a = _config(name="run one")
    b = _config(name="a completely different label")

    assert a.config_hash() == b.config_hash()


def test_config_hash_changes_with_any_tunable():
    a = _config(retrieval_strategy="hybrid_rrf")
    b = _config(retrieval_strategy="dense")

    assert a.config_hash() != b.config_hash()


def test_harness_reruns_any_config_from_its_hash(tmp_path):
    queries = [_query("q1", Stratum.HOW_TO, ["c1"])]
    search_fn = lambda q: [(q.ground_truth_chunk_ids[0], 1.0)]  # noqa: E731
    config = _config()

    original = run_eval(config, queries, search_fn, metrics=["recall@10"], cache_dir=tmp_path)
    reloaded = load_cached(tmp_path, original.config_hash)

    assert reloaded is not None
    assert reloaded.config.chunking_strategy == config.chunking_strategy
    assert reloaded.overall == original.overall
