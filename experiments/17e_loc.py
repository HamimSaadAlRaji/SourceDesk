"""Phase 17 (design record 17f): lines of code for the orchestration, direct vs graph.

Code lines only: docstrings, comments and blank lines are excluded, so a heavily commented file is not
penalised and a terse one is not flattered. Counts what DECIDES the flow, not what is shared:
the stage methods (_route, _retrieve, _generate, ...) are common to both and counted in neither.
"""

import ast
import io
import json
import tokenize
from pathlib import Path

from aegis.config import REPO_ROOT

SRC = REPO_ROOT / "src" / "aegis"
OUT = REPO_ROOT / "docs" / "results" / "orchestration-loc.json"


def code_lines(source: str, first: int, last: int, docstring_lines: set[int]) -> int:
    lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                        tokenize.ENDMARKER):
            continue
        for ln in range(tok.start[0], tok.end[0] + 1):
            if first <= ln <= last and ln not in docstring_lines:
                lines.add(ln)
    return len(lines)


def docstring_span(node: ast.AST) -> set[int]:
    body = getattr(node, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        return set(range(body[0].lineno, body[0].end_lineno + 1))
    return set()


def count(path: Path, names: set[str] | None) -> dict[str, int]:
    source = path.read_text()
    tree = ast.parse(source)
    docs: set[int] = docstring_span(tree)
    nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for n in nodes:
        docs |= docstring_span(n)
    if names is None:
        return {"<module>": code_lines(source, 1, len(source.splitlines()), docs)}
    out = {}
    for n in nodes:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            out[n.name] = code_lines(source, n.lineno, n.end_lineno, docs)
    return out


def main() -> None:
    direct = count(SRC / "pipeline.py", {"_orchestrate", "_serve_cached", "_answer"})
    graph = count(SRC / "graph" / "pipeline.py", None)
    # Within the graph module, split topology (graph construction + edge functions) from node bodies.
    nodes = count(SRC / "graph" / "pipeline.py", {
        "_orchestrate", "_build_graph", "_after_start", "_after_cache", "_after_route", "_after_confidence",
        "_n_rewrite", "_n_cache", "_n_serve_cached", "_n_route", "_n_escalate_route", "_n_retrieve",
        "_n_confidence", "_n_reformulate", "_n_escalate_confidence", "_n_generate", "__init__"})
    report = {
        "direct_orchestration_code_lines": sum(direct.values()), "direct_by_function": direct,
        "graph_module_code_lines": graph["<module>"], "graph_by_function": nodes,
        "graph_functions_total": sum(nodes.values()),
        "graph_without_retry": graph["<module>"] - nodes["_n_reformulate"],
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
