"""Phase 17 (design record 17c): do the direct and graph pipelines produce the same span tree
under a real Langfuse/OTel pipeline? An in-memory exporter stands in for the server, so no
network or spend. Distinct public keys because Langfuse keeps one resource manager per key
and would otherwise hand the second client the first one's exporter. The "Transient error"
lines are Langfuse's own exporter retrying a closed port; they are not this script's."""

import sys

sys.path.insert(0, "tests")

import test_pipeline as t  # noqa: E402
from langfuse import Langfuse  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from aegis.graph.pipeline import LangGraphPipeline  # noqa: E402
from aegis.pipeline import AnswerPipeline  # noqa: E402
from aegis.types import Query  # noqa: E402


def span_tree(cls: type[AnswerPipeline]) -> list[tuple[str, str | None, int]]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    langfuse = Langfuse(
        public_key=f"pk-{cls.__name__}", secret_key="sk", base_url="http://127.0.0.1:1", tracer_provider=provider
    )
    pipeline, _ = t._pipeline(cls=cls, confidence=1.0, tau=0.1, llm=t._FakeLLM("answer [1]"))
    pipeline.tracer = langfuse
    pipeline.answer(Query(id="x", text="how do I use dependency injection"))
    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    return [
        (s.name, by_id[s.parent.span_id].name if s.parent and s.parent.span_id in by_id else None, s.context.trace_id)
        for s in spans
    ]


if __name__ == "__main__":
    trees = {cls: span_tree(cls) for cls in (AnswerPipeline, LangGraphPipeline)}
    for cls, tree in trees.items():
        print(cls.__name__, "traces:", len({trace for _, _, trace in tree}))
        for name, parent, _ in tree:
            print("  ", name, "<-", parent)
    same = [(n, p) for n, p, _ in trees[AnswerPipeline]] == [(n, p) for n, p, _ in trees[LangGraphPipeline]]
    print("identical span trees:", same)
