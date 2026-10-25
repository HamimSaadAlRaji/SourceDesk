"""Phase 17 (design record 17f): what does a developer see when a stage blows up?

The same fault (the retriever raises) injected into both pipelines. Reported: whether the original
exception type and message survive, how deep the traceback is, and how many of its frames are the
project's own code versus the framework's. Fewer foreign frames between the fault and the project's
code is the whole of what 'debuggable' can be measured as without a debugger session.
"""

import sys
import traceback

sys.path.insert(0, "tests")

import test_graph_pipeline as g  # noqa: E402
import test_pipeline as t  # noqa: E402

from aegis.graph.pipeline import LangGraphPipeline  # noqa: E402
from aegis.pipeline import AnswerPipeline  # noqa: E402
from aegis.types import Query  # noqa: E402


class Boom(RuntimeError):
    pass


class FailingRetriever:
    def search(self, query, top_k, **kwargs):
        raise Boom("qdrant timed out")


def trace_of(cls) -> dict:
    kwargs = g._SCENARIOS["generated_answer"]()
    kwargs["retriever"] = FailingRetriever()
    pipeline, _ = t._pipeline(cls=cls, **kwargs)
    try:
        pipeline.answer(Query(id="x", text="how do I use dependency injection"))
    except Exception as exc:  # noqa: BLE001
        frames = traceback.extract_tb(exc.__traceback__)
        own = [f for f in frames if "/aegis/" in f.filename or "/experiments/" in f.filename or "/tests/" in f.filename]
        return {
            "type": type(exc).__name__, "message": str(exc), "frames": len(frames), "project_frames": len(own),
            "framework_frames": len(frames) - len(own),
            "framework_modules": sorted({f.filename.split("site-packages/")[-1].split("/")[0] for f in frames if f not in own}),
            "chained": exc.__cause__ is not None or (exc.__context__ is not None and not exc.__suppress_context__),
        }
    return {"error": "no exception raised"}


if __name__ == "__main__":
    for cls in (AnswerPipeline, LangGraphPipeline):
        print(cls.__name__, trace_of(cls))
