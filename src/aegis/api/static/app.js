// Aegis chat UI. All model/corpus text goes in via textContent only (no HTML
// parsing of it anywhere), because answers and source chunks (GitHub issue text) are untrusted.
const thread = document.getElementById("thread");
const form = document.getElementById("form");
const queryEl = document.getElementById("query");
const sendBtn = document.getElementById("send");
const MAX_HISTORY = 20; // server rejects longer histories (ApiSettings.max_history_messages)
let history = [];

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

// Inline markdown the model actually emits (**bold**, `code`), built from DOM
// nodes so nothing is ever parsed as HTML.
function appendInline(container, text) {
  for (const seg of text.split(/(\*\*[^*]+\*\*|`[^`]+`)/)) {
    if (seg.length > 4 && seg.startsWith("**") && seg.endsWith("**")) container.append(el("strong", "", seg.slice(2, -2)));
    else if (seg.length > 2 && seg.startsWith("`") && seg.endsWith("`")) container.append(el("code", "", seg.slice(1, -1)));
    else if (seg) container.append(seg);
  }
}

// Render answer text with [n] markers. A marker the validator confirmed becomes
// a clickable chip that expands the real source chunk; one it did not confirm
// is shown struck through, so an unverified citation never looks verified.
function renderAnswer(container, text, sources) {
  container.replaceChildren();
  const parts = text.split(/(\[\d+\])/);
  for (const part of parts) {
    const m = part.match(/^\[(\d+)\]$/);
    if (!m) { appendInline(container, part); continue; }
    const id = Number(m[1]);
    if (sources.has(id)) {
      const b = el("button", "cite", part);
      b.type = "button";
      b.setAttribute("aria-expanded", "false");
      b.title = "Show source";
      b.onclick = () => toggleSource(container.parentElement, id, sources.get(id), b);
      container.append(b);
    } else {
      const s = el("span", "nocite", part);
      s.title = "This citation could not be verified against the retrieved sources";
      container.append(s);
    }
  }
}

function toggleSource(card, id, src, btn) {
  const existing = card.querySelector(`.source[data-id="${id}"]`);
  if (existing) { existing.remove(); btn.setAttribute("aria-expanded", "false"); return; }
  const box = el("div", "source");
  box.dataset.id = id;
  box.append(el("b", "", `[${id}] ${src.heading_path.length ? src.heading_path.join(" › ") : src.doc_id}  (${src.chunk_id})`));
  box.append(el("pre", "", src.text));
  card.append(box);
  btn.setAttribute("aria-expanded", "true");
}

function addFeedback(card, traceId) {
  const row = el("div", "actions");
  const up = el("button", "", "👍"), down = el("button", "", "👎");
  for (const b of [up, down]) b.setAttribute("aria-pressed", "false");
  const status = el("span", "note", "");
  const comment = el("input"); comment.placeholder = "What was wrong? (optional)"; comment.maxLength = 1000; comment.hidden = true;
  const send = async (rating) => {
    up.setAttribute("aria-pressed", String(rating === "up"));
    down.setAttribute("aria-pressed", String(rating === "down"));
    comment.hidden = rating !== "down";
    const r = await fetch("/feedback", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ trace_id: traceId, rating, comment: comment.value || null }),
    });
    status.textContent = r.ok ? "Thanks — recorded." : `Could not save feedback (${r.status}).`;
  };
  up.onclick = () => send("up");
  down.onclick = () => send("down");
  comment.onchange = () => send("down");
  const trace = el("a", "", "audit trail"); trace.href = `/trace/${traceId}`; trace.target = "_blank";
  trace.title = "JSON; can take ~20 s for a fresh trace";
  row.append(up, down, comment, status, trace);
  card.append(row);
}

async function ask(text) {
  const turn = el("div", "turn");
  turn.append(el("div", "q", text));
  const card = el("div", "card");
  const answerEl = el("div", "answer");
  card.append(answerEl);
  const meta = el("div", "note", "Searching…");
  card.append(meta);
  turn.append(card);
  thread.append(turn);
  turn.scrollIntoView({ block: "end" });

  const sources = new Map();
  let answer = "", escalated = false, failed = false, assistantText = "";
  const t0 = performance.now();
  let firstToken = null;

  const handle = (ev) => {
    const d = ev.data;
    if (ev.type === "token") {
      if (firstToken === null) firstToken = performance.now() - t0;
      answer += d.text;
      answerEl.textContent = answer;
      meta.textContent = "Answering…";
    } else if (ev.type === "citation") {
      sources.set(d.citation_id, d);
    } else if (ev.type === "escalation") {
      escalated = true;
      const box = el("div", "escalation");
      card.replaceWith(box);
      box.setAttribute("role", "status");
      box.append(el("h2", "", "⚠ Escalated to a human — no answer was generated"));
      box.append(el("p", "", d.message));
      box.append(el("p", "note", d.after_retrieval
        ? "Why: the system searched the docs but wasn't confident enough that the sources answer this."
        : "Why: this kind of request is always handed to a person."));
      if (d.reason) box.append(el("p", "note", d.reason));
      assistantText = d.message;
      turn._box = box;
    } else if (ev.type === "done") {
      const target = escalated ? turn._box : card;
      if (!escalated) {
        renderAnswer(answerEl, answer, sources);
        assistantText = answer;
        const ms = Math.round(performance.now() - t0);
        const cache = d.cache_status !== "miss" ? ` · cache: ${d.cache_status}` : "";
        meta.textContent = `${firstToken !== null ? `first text ${Math.round(firstToken)} ms · ` : ""}done ${ms} ms${cache}${d.provider ? ` · ${d.provider}` : ""}`;
      } else {
      }
      if (d.rewritten_query) target.append(el("div", "note", `Searched for: “${d.rewritten_query}”`));
      addFeedback(target, d.trace_id);
    } else if (ev.type === "error") {
      failed = true;
      meta.remove();
      const msg = answer ? "The answer above is incomplete — the request failed partway." : "The request could not be completed. Please try again.";
      (escalated ? turn : card).append(el("div", "error", msg));
    }
  };

  try {
    const resp = await fetch("/chat", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: text, history }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    const parser = createSSEParser();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const ev of parser.feed(dec.decode(value, { stream: true }))) handle(ev);
    }
  } catch (e) {
    failed = true;
    meta.remove();
    card.append(el("div", "error", `Could not reach the server (${e.message}).`));
  }
  if (!failed && assistantText) {
    history.push({ role: "user", text }, { role: "assistant", text: assistantText });
    history = history.slice(-MAX_HISTORY);
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = queryEl.value.trim();
  if (!text) return;
  queryEl.value = "";
  sendBtn.disabled = true;
  try { await ask(text); } finally { sendBtn.disabled = false; queryEl.focus(); }
});
queryEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
document.getElementById("reset").onclick = () => { history = []; thread.replaceChildren(); queryEl.focus(); };
