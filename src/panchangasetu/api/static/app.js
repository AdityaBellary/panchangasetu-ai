// PanchangaSetu web chat. Plain DOM, no framework, no external requests: every element is built
// with textContent, so nothing the server returns is ever interpreted as HTML.

const KEY_STORE = "psetu.apiKey";
const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";

const STARTERS = [
  "ఈరోజు తిథి ఏమిటి?",
  "When is the next Ekadashi?",
  "ఈరోజు రాహుకాలం ఎప్పుడు?",
  "Which months are good for griha pravesham?",
  "మకర రాశి వారికి ఉద్యోగ ఫలితాలు ఎలా ఉన్నాయి?",
  "What is Kuja dosha?",
];

const state = { key: null, sessionId: null, busy: false, today: null };

// ------------------------------------------------------------------ api
class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

async function api(path, { method = "GET", body } = {}) {
  const res = await fetch(path, {
    method,
    headers: { Authorization: `Bearer ${state.key}`, ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (res.status === 204) return null;
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail; } catch { /* not JSON */ }
    throw new ApiError(res.status, typeof detail === "string" ? detail : "");
  }
  return res.headers.get("content-type")?.includes("json") ? res.json() : res.blob();
}

function explain(err) {
  if (err.status === 401) return "This API key was not accepted. Enter a valid key to continue.";
  if (err.status === 429) return "That is more than a minute's worth of questions. Wait a moment and ask again.";
  if (err.status === 404) return "That conversation no longer exists. Start a new one.";
  if (err.status === 422) return err.message || "The question could not be read.";
  return "The server could not answer just now. Check that it is running and ask again.";
}

// ------------------------------------------------------------------ small DOM helpers
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c);
  return node;
}

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const c of children) node.append(c);
  return node;
}

const isTelugu = (s) => /[ఀ-౿]/.test(s || "");
const langOf = (s) => (isTelugu(s) ? "te" : "en");

// ------------------------------------------------------------------ moon
// Tithi 1-15 waxing (15 = full), 16-30 waning (30 = new). The lit part is a semicircle plus a
// terminator ellipse whose width follows the illuminated fraction.
function moon(number, size = 120) {
  const r = 46, c = 50;
  const root = svg("svg", { viewBox: "0 0 100 100", width: size, height: size, role: "img" });
  root.append(svg("circle", { cx: c, cy: c, r, fill: "#0d1333" }));
  if (!number) return root;
  const waxing = number <= 15;
  const lit = waxing ? number / 15 : (30 - number) / 15; // 0 new .. 1 full
  if (lit > 0) {
    const rx = Math.abs(1 - 2 * lit) * r;
    const sweep = lit > 0.5 ? 1 : 0;
    const d = `M ${c} ${c - r} A ${r} ${r} 0 0 1 ${c} ${c + r} A ${rx} ${r} 0 0 ${sweep} ${c} ${c - r} Z`;
    const path = svg("path", { d, fill: "#f3d27a" });
    if (!waxing) path.setAttribute("transform", `translate(${2 * c} 0) scale(-1 1)`);
    root.append(path);
  }
  root.append(svg("circle", { cx: c, cy: c, r, fill: "none", stroke: "rgba(243,210,122,.35)", "stroke-width": 1 }));
  return root;
}

// ------------------------------------------------------------------ time formatting
function clock(hhmm) {
  const [h, m] = hhmm.split(":").map(Number);
  return `${h % 12 || 12}:${String(m).padStart(2, "0")} ${h < 12 ? "AM" : "PM"}`;
}

function untilText(iso, todayIso) {
  if (!iso) return "";
  const t = clock(iso.slice(11, 16)); // printed local time; never shifted by the browser's timezone
  const day = iso.slice(0, 10);
  if (day === todayIso) return `until ${t}`;
  const next = new Date(`${todayIso}T12:00`);
  next.setDate(next.getDate() + 1);
  return day === next.toISOString().slice(0, 10) ? `until ${t} tomorrow` : `until ${t} on ${day}`;
}

function when(iso) {
  const d = new Date(iso);
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  const y = new Date(now); y.setDate(now.getDate() - 1);
  const time = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  if (sameDay) return `Today, ${time}`;
  if (d.toDateString() === y.toDateString()) return `Yesterday, ${time}`;
  return d.toLocaleDateString([], { day: "numeric", month: "short", year: d.getFullYear() === now.getFullYear() ? undefined : "numeric" });
}

// ------------------------------------------------------------------ today
function renderToday(t) {
  const box = $("today");
  box.replaceChildren();
  const date = new Date(`${t.date}T12:00`).toLocaleDateString("en-IN", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
  if (!t.available || !t.tithi) {
    box.append(moon(null, 120), el("div", {}, el("p", { class: "today-date", text: date }),
      el("p", { class: "today-missing", text: "The panchangam has no entry for today. You can still ask about other dates and topics." })));
    return;
  }
  const tithi = t.tithi;
  const art = moon(tithi.number, 120);
  art.classList.add("today-moon");
  art.setAttribute("aria-label", `Moon: ${tithi.paksha_en} paksha ${tithi.name_en}`);
  const facts = el("dl", { class: "today-facts" });
  const fact = (label, value) => value && facts.append(el("div", {}, el("dt", { text: label }), el("dd", { text: value })));
  fact("Tithi", `${tithi.name_en} ${untilText(tithi.ends_at, t.date)}`);
  if (t.nakshatra) fact("Nakshatram", `${t.nakshatra.name_en} ${untilText(t.nakshatra.ends_at, t.date)}`);
  if (t.rahukalam) fact("Rahukalam", `${clock(t.rahukalam.start)} to ${clock(t.rahukalam.end)}`);
  if (t.sunrise) fact("Sunrise", `${t.sunrise.replace("-", ":")} AM`);
  const masa = [tithi.masa, tithi.paksha].filter(Boolean).join(" ");
  const masaEn = [tithi.masa_en, tithi.paksha_en && `${tithi.paksha_en} paksha`].filter(Boolean).join(", ");
  box.append(
    art,
    el("div", {},
      el("p", { class: "today-date", text: date }),
      el("h1", { class: "today-tithi", lang: "te", text: tithi.name }),
      el("p", { class: "today-sub" }, el("span", { lang: "te", text: masa }), masa && masaEn ? "  " : "", el("span", { text: masaEn ? `(${masaEn})` : "" })),
      facts,
      el("p", { class: "today-source", text: `From ${t.source.document.replace(/_/g, " ").replace(/\.pdf$/i, "")}, page ${t.source.pages[0]}${t.verified ? "" : " (not verified against the book)"}` }),
    ),
  );
  const pill = $("today-pill");
  pill.replaceChildren(moon(tithi.number, 22), el("span", {}, el("strong", { lang: "te", text: tithi.name }), ` ${tithi.name_en}`, t.nakshatra ? `, ${t.nakshatra.name_en}` : ""));
}

// ------------------------------------------------------------------ thread
function paragraphs(text) {
  return text.split(/\n{2,}/).map((block) =>
    el("p", { lang: langOf(block) }, ...block.split("\n").flatMap((line, i) => (i ? [el("br"), line] : [line]))));
}

const METHOD_NOTE = {
  template: "Read directly from the panchangam tables",
  llm: "Written from the passages below",
};

function seal(verified) {
  const s = svg("svg", { viewBox: "0 0 16 16", class: "seal", "aria-hidden": "true" });
  if (verified === true) {
    s.classList.add("seal-verified");
    s.append(svg("path", { d: "M8 1l1.9 1.6 2.5-.2.6 2.4 2.1 1.4-1 2.3 1 2.3-2.1 1.4-.6 2.4-2.5-.2L8 15l-1.9-1.6-2.5.2-.6-2.4L.9 9.8 1.9 7.5.9 5.2 3 3.8l.6-2.4 2.5.2z" }));
  } else if (verified === false) {
    s.classList.add("seal-unverified");
    s.append(svg("circle", { cx: 8, cy: 8, r: 6 }));
  } else {
    s.classList.add("seal-prose");
    s.append(svg("rect", { x: 3, y: 2, width: 10, height: 12, rx: 1.5 }));
  }
  return s;
}

function docLabel(c) {
  const name = c.document.replace(/_/g, " ").replace(/\.pdf$/i, "");
  const pages = c.pages.length ? `, page ${c.pages.slice(0, 3).join(", ")}` : "";
  return `${name}${pages}`;
}

function renderAnswer(a, { animate = false } = {}) {
  const lang = a.language;
  const li = el("li", { class: `turn-a${a.abstained ? " abstained" : ""}` });
  if (a.standalone_question && a.standalone_question !== a.question) {
    li.append(el("p", { class: "understood" }, "Answered as ", el("q", { lang: langOf(a.standalone_question), text: a.standalone_question })));
  }
  li.append(el("div", { class: "answer-text", lang }, ...paragraphs(a.answer)));
  if (a.citations?.length) {
    const list = el("ul", { class: "sources", "aria-label": "Sources" });
    for (const c of a.citations) {
      const label = c.date ? `${docLabel(c)} (${c.date})` : docLabel(c);
      const title = c.verified === true ? "Verified against the book" : c.verified === false ? "Not verified against the book" : "Passage from the book";
      list.append(el("li", {}, el("button", { class: "source", type: "button", title, onclick: () => openSource(c) }, seal(c.verified), label)));
    }
    li.append(list);
  }
  const note = a.abstained ? null : METHOD_NOTE[a.method];
  if (note) li.append(el("p", { class: "answer-meta", text: a.latency_ms ? `${note}, in ${duration(a.latency_ms)}` : note }));
  if (animate) li.scrollIntoView?.({ block: "nearest" });
  return li;
}

function duration(ms) {
  return ms < 1000 ? `${Math.max(1, Math.round(ms))} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function pushQuestion(q) {
  $("welcome").hidden = true;
  $("today-pill").hidden = !state.today?.tithi;
  $("thread").append(el("li", { class: "turn-q", lang: langOf(q), text: q }));
}

function thinking() {
  const icon = moon(22, 22);
  return el("li", { class: "thinking", role: "status" }, icon, "Reading the panchangam…");
}

function scrollToEnd() {
  const s = $("scroll");
  s.scrollTop = s.scrollHeight;
}

// ------------------------------------------------------------------ asking
async function ask(question) {
  question = question.trim();
  if (!question || state.busy) return;
  state.busy = true;
  $("ask").disabled = true;
  pushQuestion(question);
  const wait = thinking();
  $("thread").append(wait);
  scrollToEnd();
  try {
    const a = await api("/v1/chat", { method: "POST", body: { question, session_id: state.sessionId } });
    const isNew = state.sessionId !== a.session_id;
    state.sessionId = a.session_id;
    wait.replaceWith(renderAnswer(a));
    if (isNew) loadHistory();
  } catch (err) {
    if (err.status === 401) return signOut(explain(err));
    if (err.status === 404) state.sessionId = null;
    wait.replaceWith(el("li", { class: "turn-a error", role: "alert", text: explain(err) }));
  } finally {
    state.busy = false;
    $("ask").disabled = false;
    scrollToEnd();
    $("question").focus();
  }
}

// ------------------------------------------------------------------ history
async function loadHistory() {
  const nav = $("history");
  let sessions = [];
  try { sessions = await api("/v1/sessions"); } catch { return; }
  nav.replaceChildren();
  if (!sessions.length) {
    nav.append(el("p", { class: "history-empty", text: "Your conversations will appear here." }));
    return;
  }
  for (const s of sessions) {
    const del = el("button", { class: "history-delete", type: "button", "aria-label": "Delete conversation", title: "Delete conversation",
      onclick: async (e) => {
        e.stopPropagation();
        try { await api(`/v1/sessions/${s.id}`, { method: "DELETE" }); } catch { /* already gone */ }
        if (state.sessionId === s.id) newConversation();
        loadHistory();
      } });
    del.append(svg("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" }, svg("path", { d: "M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12" })));
    nav.append(el("div", { class: "history-item", "aria-current": s.id === state.sessionId ? "true" : "false" },
      el("button", { class: "history-open", type: "button", onclick: () => openConversation(s.id) },
        el("span", { class: "history-title", lang: langOf(s.title), text: s.title }),
        el("span", { class: "history-when", text: when(s.updated_at) })),
      del));
  }
}

async function openConversation(id) {
  closeSidebar();
  try {
    const s = await api(`/v1/sessions/${id}`);
    state.sessionId = id;
    $("thread").replaceChildren();
    $("welcome").hidden = true;
    for (const t of s.turns) {
      pushQuestion(t.question);
      $("thread").append(renderAnswer({ ...t, citations: t.citations }));
    }
    scrollToEnd();
    loadHistory();
  } catch (err) {
    if (err.status === 401) return signOut(explain(err));
    loadHistory();
  }
}

function newConversation() {
  state.sessionId = null;
  $("thread").replaceChildren();
  $("welcome").hidden = false;
  $("today-pill").hidden = true;
  closeSidebar();
  loadHistory();
  $("question").focus();
}

// ------------------------------------------------------------------ source drawer
const pageCache = new Map();

async function openSource(c) {
  $("drawer-doc").textContent = docLabel(c);
  const status = c.verified === true ? "Verified against the book" : c.verified === false ? "Not verified against the book" : "Passage from the book";
  $("drawer-meta").textContent = c.date ? `${c.date}. ${status}.` : `${status}.`;
  const quotes = $("drawer-quotes");
  quotes.replaceChildren();
  if (c.quotes?.length) {
    quotes.append(el("p", { class: "quotes-label", text: "Supporting text, checked word for word against this page" }));
    for (const q of c.quotes) quotes.append(el("blockquote", { lang: langOf(q), text: q }));
  }
  const pages = $("drawer-pages");
  pages.replaceChildren();
  $("drawer").hidden = false;
  $("drawer-close").focus();
  for (const [i, url] of (c.page_image_urls || []).entries()) {
    const fig = el("figure", {}, el("p", { class: "page-missing", text: "Loading page…" }));
    pages.append(fig);
    try {
      if (!pageCache.has(url)) pageCache.set(url, URL.createObjectURL(await api(url)));
      const src = pageCache.get(url);
      fig.replaceChildren(
        el("a", { href: src, target: "_blank", rel: "noopener", title: "Open the page at full size" },
          el("img", { src, alt: `Page ${c.pages[i]} of ${c.document}` })),
        el("figcaption", {}, `Page ${c.pages[i]}. `, el("a", { href: src, target: "_blank", rel: "noopener", text: "Open full size" })));
    } catch {
      fig.replaceChildren(el("p", { class: "page-missing", text: `The image of page ${c.pages[i]} is not available.` }));
    }
  }
}

function closeDrawer() { $("drawer").hidden = true; }

// ------------------------------------------------------------------ sidebar (small screens)
function closeSidebar() {
  $("sidebar").classList.remove("open");
  $("menu").setAttribute("aria-expanded", "false");
}

// ------------------------------------------------------------------ key gate
function signOut(message) {
  localStorage.removeItem(KEY_STORE);
  state.key = null;
  state.sessionId = null;
  $("app").hidden = true;
  $("gate").hidden = false;
  const errBox = $("gate-error");
  errBox.hidden = !message;
  errBox.textContent = message || "";
  $("gate-key").value = "";
  $("gate-key").focus();
}

async function start(key) {
  state.key = key;
  const today = await api("/v1/today"); // also proves the key works
  localStorage.setItem(KEY_STORE, key);
  state.today = today;
  $("gate").hidden = true;
  $("app").hidden = false;
  renderToday(today);
  $("today-pill").hidden = true;
  api("/v1/whoami").then((me) => { $("monitoring").hidden = !me.is_admin; }).catch(() => {});
  loadHistory();
  $("question").focus();
}

// ------------------------------------------------------------------ wiring
function wire() {
  const starters = $("starters");
  for (const q of STARTERS) starters.append(el("li", {}, el("button", { type: "button", lang: langOf(q), text: q, onclick: () => ask(q) })));

  const input = $("question");
  const counter = $("counter");
  const grow = () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
    const left = 500 - input.value.length;
    counter.hidden = left > 60;
    counter.textContent = `${left} left`;
  };
  input.addEventListener("input", grow);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value;
    input.value = "";
    grow();
    ask(q);
  });

  $("new-chat").addEventListener("click", newConversation);
  $("sign-out").addEventListener("click", () => signOut());
  $("drawer-close").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeDrawer(); closeSidebar(); } });
  $("menu").addEventListener("click", () => {
    const open = $("sidebar").classList.toggle("open");
    $("menu").setAttribute("aria-expanded", String(open));
  });

  $("gate-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const key = $("gate-key").value.trim();
    const btn = e.submitter;
    if (btn) btn.disabled = true;
    try {
      await start(key);
    } catch (err) {
      signOut(explain(err));
    } finally {
      if (btn) btn.disabled = false;
    }
  });
}

wire();
let saved = null;
try { saved = localStorage.getItem(KEY_STORE); } catch { /* storage blocked */ }
if (saved) start(saved).catch((err) => signOut(explain(err)));
else signOut();
