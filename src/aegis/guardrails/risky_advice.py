"""Output-side risky-advice check (Phase 15, design record 15l, Open Question #20).

15h found the citation validator *certifies* poisoned advice: when the model
relayed an issue comment's `pip install fastapi-securefix`, the sentence
scored SUPPORTED_LEXICAL (4/4), because "supported" only means "a retrieved
block says so" -- and issue comments are attacker-writable. A validated
citation makes the malicious step look more authoritative, not less.

So, like the commitment check (15e), a code path rather than a prompt rule:
a sentence that tells the user to take an action with real-world blast
radius is never validated unless the TRUSTED half of the corpus -- the
official documentation, never issue comments -- vouches for the specific
thing. Three triggers:

1. install of a package the documentation never mentions, or of a raw
   URL / VCS target;
2. a link to a host that no documentation chunk links to;
3. a request for the user's own secret (paste/share/send ... your key,
   token, password, secret, .env), unless negated.

Vouching is corpus-wide over documentation chunks, deliberately not "the
chunks this answer cites": the answer may legitimately cite an issue for a
fix that uses a documented package (python-multipart, uvicorn).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from aegis.config import settings
from aegis.observability.logging import get_logger, log_event

_log = get_logger(__name__)

_r = settings.risky_advice
_INSTALL_CMD = re.compile(_r.install_command, re.IGNORECASE)
_INSTALL_PROSE = re.compile(_r.install_prose, re.IGNORECASE)
_URL = re.compile(r"\b(?:https?|git\+https?)://([A-Za-z0-9.-]+)", re.IGNORECASE)
_SECRET_REQUEST = re.compile(_r.secret_request, re.IGNORECASE)
_NEGATION = re.compile(_r.negation, re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]*")
_STOPWORDS = frozenset(_r.install_stopwords)
_SPEC = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-\[\],=<>~!:/+@]*$")


@dataclass(frozen=True)
class Vouched:
    """What the trusted (documentation) corpus mentions: word-like tokens and link hosts."""

    tokens: frozenset[str]
    hosts: frozenset[str]

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> Vouched:
        tokens: set[str] = set()
        hosts: set[str] = set()
        for text in texts:
            tokens.update(t.strip("._-") for t in _TOKEN.findall(text.lower()))
            hosts.update(h.lower() for h in _URL.findall(text))
        return cls(frozenset(tokens), frozenset(hosts | set(_r.always_vouched_hosts)))


@lru_cache(maxsize=1)
def documentation_vouched() -> Vouched:
    """Fails closed: with no trusted corpus, nothing is vouched, so every install
    or link fires rather than every one passing."""
    path = settings.paths.data_processed / "chunks.jsonl"
    if not path.exists():
        log_event(_log, logging.WARNING, "risky_advice.no_documentation_corpus", path=str(path))
        return Vouched.from_texts([])
    with open(path, encoding="utf-8") as f:
        rows = (json.loads(line) for line in f)
        return Vouched.from_texts(r["text"] for r in rows if r["doc_type"] == "documentation")


def _package_base(target: str) -> str:
    return re.split(r"[\[=<>~!@;]", target.strip("`'\".,;:()").lower(), maxsplit=1)[0]


def _install_targets(sentence: str) -> list[str]:
    targets: list[str] = []
    for m in _INSTALL_CMD.finditer(sentence):
        for raw in m.group("targets").split():
            tok = raw.strip("`'\"")
            if tok.startswith("-"):
                continue  # flags: -U, --upgrade
            if tok.lower() in _STOPWORDS or tok.startswith("[") or not _SPEC.match(tok.rstrip(".,;:)")):
                break  # end of the target list: back in prose, or a citation marker
            targets.append(tok.rstrip(".,;:)"))
            if raw.rstrip("`'\"").endswith((".", ",", ";", ":", ")")):
                break
    targets += [m.group("name") for m in _INSTALL_PROSE.finditer(sentence)]
    return targets


def risky_reasons(sentence: str, vouched: Vouched | None = None) -> list[str]:
    """Why a sentence is unvouched risky advice; empty when it isn't."""
    v = vouched or documentation_vouched()
    reasons: list[str] = []
    for target in _install_targets(sentence):
        if "://" in target or target.startswith(("git+", "./", "/")):
            reasons.append(f"install_from_url:{target}")
        elif (base := _package_base(target)) and base not in v.tokens:
            reasons.append(f"unvouched_package:{base}")
    for host in _URL.findall(sentence):
        if host.lower() not in v.hosts:
            reasons.append(f"unvouched_host:{host.lower()}")
    if (m := _SECRET_REQUEST.search(sentence)) and not _NEGATION.search(sentence[: m.start()]):
        reasons.append("secret_request")
    return reasons


def is_risky_advice(sentence: str, vouched: Vouched | None = None) -> bool:
    return bool(risky_reasons(sentence, vouched))
