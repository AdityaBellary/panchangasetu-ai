// PanchangaSetu monitoring. Plain DOM + hand-built SVG (no external scripts: strict CSP).
// Every value from the server is inserted as text, never as HTML.

const KEY_STORE = "psetu.adminKey";
const SVG_NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const state = { key: null, hours: 24, tab: "requests", method: "", errorsOnly: false, offset: 0, timer: null };

// ------------------------------------------------------------------ helpers
function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k === "style") n.style.cssText = v; // CSSOM: allowed by the CSP, unlike a style attribute
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) n.append(c);
  return n;
}
function svg(tag, attrs = {}, ...children) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  for (const c of children) n.append(c);
  return n;
}
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmtInt = (n) => (n ?? 0).toLocaleString("en-IN");
const fmtInr = (x) => (x == null ? "–" : `₹${x < 1 ? x.toFixed(4) : x.toFixed(2)}`);
const fmtMs = (x) => (x == null ? "–" : x < 1000 ? `${Math.round(x)} ms` : `${(x / 1000).toFixed(1)} s`);
const fmtTok = (n) => (n >= 10000 ? `${(n / 1000).toFixed(1)}K` : fmtInt(n));
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const isTelugu = (s) => /[ఀ-౿]/.test(s || "");

async function api(path) {
  const res = await fetch(path, { headers: { Authorization: `Bearer ${state.key}` } });
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail; } catch { /* not JSON */ }
    const err = new Error(detail || `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

function timeLabel(iso, hourly) {
  const d = new Date(iso);
  return hourly ? d.toLocaleTimeString([], { hour: "numeric" }) : d.toLocaleDateString([], { day: "numeric", month: "short" });
}
function when(iso) {
  const d = new Date(iso);
  return d.toLocaleString([], { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });
}

// ------------------------------------------------------------------ tooltip
const tip = () => $("tip");
function showTip(evt, rows, title) {
  const t = tip();
  t.replaceChildren(
    title ? el("div", {}, el("b", { text: title })) : null,
    ...rows.map(([color, label, value]) =>
      el("div", { class: "row" }, color ? el("i", { style: `background:${color}` }) : null, `${label}: `, el("b", { text: value }))),
  );
  t.hidden = false;
  const x = Math.min(evt.clientX + 14, window.innerWidth - t.offsetWidth - 8);
  const y = Math.min(evt.clientY + 14, window.innerHeight - t.offsetHeight - 8);
  t.style.left = `${x}px`;
  t.style.top = `${y}px`;
}
const hideTip = () => { tip().hidden = true; };

// ------------------------------------------------------------------ charts
const W = 560, H = 200, PAD = { l: 36, r: 10, t: 10, b: 24 };

function axes(root, maxY, n, labels, fmtY) {
  const plotH = H - PAD.t - PAD.b;
  const ticks = 5; // divides the nice maxima (1, 2, 2.5, 5, 10 × 10ⁿ) into round steps
  for (let i = 0; i <= ticks; i++) {
    const v = (maxY / ticks) * i;
    const y = PAD.t + plotH - (plotH * i) / ticks;
    root.append(svg("line", { class: "gridline", x1: PAD.l, x2: W - PAD.r, y1: y, y2: y }));
    root.append(svg("text", { x: PAD.l - 6, y: y + 3.5, "text-anchor": "end" }, document.createTextNode(fmtY(v))));
  }
  const every = Math.max(1, Math.ceil(n / 8));
  const band = (W - PAD.l - PAD.r) / n;
  labels.forEach((lab, i) => {
    if (i % every) return;
    root.append(svg("text", { x: PAD.l + band * i + band / 2, y: H - 6, "text-anchor": "middle" }, document.createTextNode(lab)));
  });
  return { plotH, band };
}

function niceMax(v) {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v);
}

function columnChart(points, series, { hourly, fmtY = (v) => fmtInt(Math.round(v)) }) {
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img" });
  const totals = points.map((p) => series.reduce((s, k) => s + (p[k.key] || 0), 0));
  const maxY = niceMax(Math.max(...totals, 0));
  const { plotH, band } = axes(root, maxY, points.length, points.map((p) => timeLabel(p.t, hourly)), fmtY);
  const barW = Math.min(24, band * 0.7);
  points.forEach((p, i) => {
    const x = PAD.l + band * i + (band - barW) / 2;
    let y0 = PAD.t + plotH;
    const segs = series.filter((s) => p[s.key] > 0);
    segs.forEach((s, j) => {
      const h = (p[s.key] / maxY) * plotH;
      const top = j === segs.length - 1;
      const gap = j > 0 ? 2 : 0; // surface gap between stacked segments
      const y = y0 - h;
      const r = top ? Math.min(4, barW / 2, h) : 0;
      // rounded data-end (top), square at the baseline
      const d = `M${x},${y0 - gap} V${y + r} Q${x},${y} ${x + r},${y} H${x + barW - r} Q${x + barW},${y} ${x + barW},${y + r} V${y0 - gap} Z`;
      root.append(svg("path", { d, fill: s.color }));
      y0 = y;
    });
    const hit = svg("rect", { class: "hit", x: PAD.l + band * i, y: PAD.t, width: band, height: plotH });
    hit.addEventListener("mousemove", (e) => showTip(e, series.map((s) => [s.color, s.label, s.fmt ? s.fmt(p[s.key]) : fmtInt(p[s.key])]), timeLabel(p.t, hourly)));
    hit.addEventListener("mouseleave", hideTip);
    root.append(hit);
  });
  return root;
}

function lineChart(points, series, { hourly }) {
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img" });
  const vals = points.flatMap((p) => series.map((s) => p[s.key]).filter((v) => v != null));
  const maxY = niceMax(Math.max(...vals, 0));
  const { plotH, band } = axes(root, maxY, points.length, points.map((p) => timeLabel(p.t, hourly)), (v) => fmtMs(v));
  const xy = (i, v) => [PAD.l + band * i + band / 2, PAD.t + plotH - (v / maxY) * plotH];
  for (const s of series) {
    let d = "";
    let pen = false;
    points.forEach((p, i) => {
      if (p[s.key] == null) { pen = false; return; }
      const [x, y] = xy(i, p[s.key]);
      d += `${pen ? "L" : "M"}${x},${y} `;
      pen = true;
    });
    if (d) root.append(svg("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    points.forEach((p, i) => {
      if (p[s.key] == null) return;
      const [x, y] = xy(i, p[s.key]);
      root.append(svg("circle", { cx: x, cy: y, r: 3, fill: s.color, stroke: css("--card"), "stroke-width": 2 }));
    });
  }
  const cross = svg("line", { x1: 0, x2: 0, y1: PAD.t, y2: PAD.t + plotH, stroke: css("--slate"), "stroke-width": 1, opacity: 0 });
  root.append(cross);
  points.forEach((p, i) => {
    const hit = svg("rect", { class: "hit", x: PAD.l + band * i, y: PAD.t, width: band, height: plotH });
    hit.addEventListener("mousemove", (e) => {
      const [x] = xy(i, 0);
      cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("opacity", 0.35);
      showTip(e, series.map((s) => [s.color, s.label, fmtMs(p[s.key])]), timeLabel(p.t, hourly));
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
    root.append(hit);
  });
  return root;
}

function legend(items) {
  return el("ul", { class: "legend" }, items.map(([color, label, value, line]) =>
    el("li", {}, el("i", { class: line ? "line" : "", style: `background:${color}` }), label, value != null ? el("b", { text: value }) : null)));
}

function tableView(headers, rows) {
  return el("details", { class: "table-view" }, el("summary", { text: "Show as table" }),
    el("div", { class: "tbl-wrap" }, el("table", { class: "tbl" },
      el("thead", {}, el("tr", {}, headers.map((h, i) => el("th", { class: i ? "num" : "", text: h })))),
      el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c, i) => el("td", { class: i ? "num" : "", text: c }))))))));
}

function hbars(entries, fmt = fmtInt) {
  if (!entries.length) return el("p", { class: "empty", text: "Nothing in this period." });
  const max = Math.max(...entries.map(([, v]) => v), 1);
  return el("ul", { class: "hbars" }, entries.map(([label, v]) =>
    el("li", {}, el("span", { class: "label", title: label, text: label }),
      el("span", { class: "track" }, el("span", { class: "bar", style: `width:${(v / max) * 100}%` })),
      el("span", { class: "val", text: fmt(v) }))));
}

function card(id, title, note, ...body) {
  const parts = [el("h2", { text: title }), note ? el("p", { class: "note", text: note }) : null, ...body];
  $(id).replaceChildren(...parts.filter((x) => x != null)); // replaceChildren would print "null"
}

// ------------------------------------------------------------------ overview
const REASONS = {
  no_evidence: "Nothing relevant found", date_not_covered: "Date outside the documents",
  insufficient_sources: "Sources judged insufficient", verification_failed: "Failed fact check",
  llm_unavailable: "LLM unavailable", budget: "Daily budget reached", no_llm: "No LLM configured", other: "Other",
  unrecorded: "Not recorded (older answers)",
};
const METHOD = () => [
  { key: "template", label: "From tables", color: css("--series-1") },
  { key: "llm", label: "LLM answers", color: css("--series-2") },
  { key: "abstain", label: "Refused", color: css("--series-3") },
];

function renderOverview(o) {
  const k = o.kpis;
  const hourly = o.window_hours <= 48;
  const period = o.window_hours <= 24 ? "last 24 hours" : `last ${Math.round(o.window_hours / 24)} days`;
  const b = o.budget;
  const used = b.limit_inr > 0 ? b.spent_today_inr / b.limit_inr : null;
  const tile = (label, value, sub, extra) => el("dl", { class: "tile" }, el("dt", { text: label }), el("dd", {}, value, sub ? el("span", { class: "sub", text: sub }) : null), extra || null);
  $("tiles").replaceChildren(
    tile("Questions", fmtInt(k.questions), period),
    tile("Answered", pct(k.answer_rate), `${fmtInt(k.answered)} answered, ${fmtInt(k.abstained)} refused`),
    tile("LLM spend", fmtInr(k.cost_inr), k.avg_cost_per_llm_answer_inr != null ? `${fmtInr(k.avg_cost_per_llm_answer_inr)} per LLM answer` : "no LLM answers"),
    tile("Spent today", fmtInr(b.spent_today_inr), b.limit_inr > 0 ? `of ${fmtInr(b.limit_inr)} daily budget (${pct(used)})` : "no daily limit",
      b.limit_inr > 0 ? el("div", { class: `meter${used >= 1 ? " over" : ""}`, role: "meter", "aria-valuenow": Math.round(used * 100), "aria-valuemin": 0, "aria-valuemax": 100, "aria-label": "Daily budget used" }, el("span", { style: `width:${Math.min(100, used * 100)}%` })) : null),
    tile("LLM calls", fmtInt(k.llm_calls), k.llm_errors ? `${fmtInt(k.llm_errors)} failed` : "none failed"),
    tile("Tokens", fmtTok(k.prompt_tokens + k.completion_tokens), `${fmtTok(k.prompt_tokens)} in, ${fmtTok(k.completion_tokens)} out`),
    tile("Latency", fmtMs(k.latency_p50_ms), `median; p95 ${fmtMs(k.latency_p95_ms)}`),
  );

  const methods = METHOD();
  const total = (key) => o.series.reduce((s, p) => s + (p[key] || 0), 0);
  card("card-mix", "Questions by answer type", hourly ? "Per hour" : "Per day",
    legend(methods.map((m) => [m.color, m.label, fmtInt(total(m.key))])),
    o.series.length ? columnChart(o.series, methods, { hourly }) : el("p", { class: "empty", text: "No questions yet." }),
    tableView(["Time", ...methods.map((m) => m.label)], o.series.map((p) => [timeLabel(p.t, hourly), ...methods.map((m) => fmtInt(p[m.key]))])));

  const blue = css("--series-1");
  card("card-cost", "LLM spend", `${hourly ? "Per hour" : "Per day"}, ₹; template answers cost nothing`,
    columnChart(o.series, [{ key: "cost_inr", label: "Spend", color: blue, fmt: fmtInr }], { hourly, fmtY: (v) => (v < 1 ? `₹${v.toFixed(2)}` : `₹${Math.round(v)}`) }),
    tableView(["Time", "Spend"], o.series.map((p) => [timeLabel(p.t, hourly), fmtInr(p.cost_inr)])));

  const lat = [{ key: "p50_ms", label: "Median", color: css("--series-1") }, { key: "p95_ms", label: "p95", color: css("--series-2") }];
  card("card-latency", "Answer latency", "Question in to answer out",
    legend(lat.map((s) => [s.color, s.label, null, true])),
    lineChart(o.series, lat, { hourly }),
    tableView(["Time", "Median", "p95"], o.series.map((p) => [timeLabel(p.t, hourly), fmtMs(p.p50_ms), fmtMs(p.p95_ms)])));

  const order = ["rewrite", "retrieve", "llm", "total"];
  const names = { rewrite: "Follow-up rewrite", retrieve: "Routing + search", llm: "LLM (all calls)", total: "End to end" };
  const stages = order.filter((s) => o.stages[s]).map((s) => [names[s], o.stages[s]]);
  const smax = Math.max(...stages.map(([, v]) => v.p95_ms || 0), 1);
  card("card-stages", "Where the time goes", "Bar = median, line = p95, per stage",
    stages.length ? el("ul", { class: "hbars" }, stages.map(([label, v]) =>
      el("li", {}, el("span", { class: "label", text: label }),
        el("span", { class: "track" },
          el("span", { class: "bar", style: `width:${((v.p50_ms || 0) / smax) * 100}%` }),
          el("span", { class: "bar p95", style: `width:${((v.p95_ms || 0) / smax) * 100}%` })),
        el("span", { class: "val", text: `${fmtMs(v.p50_ms)} / ${fmtMs(v.p95_ms)}` })))) : el("p", { class: "empty", text: "No timings yet." }));

  const reasons = Object.entries(o.abstention_reasons).sort((a, b) => b[1] - a[1]).map(([k2, v]) => [REASONS[k2] || k2, v]);
  card("card-reasons", "Why questions were refused", "Refusals are by design when the documents don't support an answer",
    hbars(reasons));

  const langs = Object.entries(o.languages).map(([k2, v]) => [k2 === "te" ? "Telugu" : k2 === "en" ? "English" : k2, v]);
  const rew = Object.entries(o.rewrites).map(([k2, v]) => [{ none: "Standalone", rule: "Follow-up (rules)", llm: "Follow-up (LLM)" }[k2] || k2, v]);
  card("card-mixes", "Who asks what", null,
    el("div", { class: "split" },
      el("div", {}, el("h3", { text: "Language" }), hbars(langs)),
      el("div", {}, el("h3", { text: "Question type" }), hbars(rew))),
    el("h3", { class: "routes-h", text: "Routes" }), hbars(Object.entries(o.routes)));

  const mt = ["Provider", "Model", "Calls", "Failed", "Tokens in", "Tokens out", "Spend", "Per call", "Median", "p95"];
  $("card-models").replaceChildren(el("h2", { text: "LLM providers and models" }),
    el("p", { class: "note", text: "Every HTTP call, retries and fallbacks included. Thinking tokens bill as output." }),
    o.models.length ? el("div", { class: "tbl-wrap" }, el("table", { class: "tbl" },
      el("thead", {}, el("tr", {}, mt.map((h, i) => el("th", { class: i > 1 ? "num" : "", text: h })))),
      el("tbody", {}, o.models.map((m) => el("tr", {},
        el("td", { text: m.provider }), el("td", { text: m.model }), el("td", { class: "num", text: fmtInt(m.calls) }),
        el("td", { class: `num ${m.errors ? "status-bad" : ""}`, text: fmtInt(m.errors) }),
        el("td", { class: "num", text: fmtTok(m.prompt_tokens) }), el("td", { class: "num", text: fmtTok(m.completion_tokens) }),
        el("td", { class: "num", text: fmtInr(m.cost_inr) }), el("td", { class: "num", text: fmtInr(m.avg_cost_inr) }),
        el("td", { class: "num", text: fmtMs(m.p50_ms) }), el("td", { class: "num", text: fmtMs(m.p95_ms) }))))))
      : el("p", { class: "empty", text: "No LLM calls in this period." }));

  $("card-clients").replaceChildren(el("h2", { text: "API clients" }),
    o.clients.length ? el("div", { class: "tbl-wrap" }, el("table", { class: "tbl" },
      el("thead", {}, el("tr", {}, ["Key", "Questions", "Spend"].map((h, i) => el("th", { class: i ? "num" : "", text: h })))),
      el("tbody", {}, o.clients.map((c) => el("tr", {}, el("td", { text: c.name }), el("td", { class: "num", text: fmtInt(c.questions) }), el("td", { class: "num", text: fmtInr(c.cost_inr) }))))))
      : el("p", { class: "empty", text: "No API usage in this period." }));

  $("updated").textContent = `Updated ${new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" })}`;
}

// ------------------------------------------------------------------ request log and calls
function methodChip(m) {
  const map = Object.fromEntries(METHOD().map((x) => [x.key, x]));
  const x = map[m] || map.abstain;
  return el("span", { class: "chip" }, el("i", { style: `background:${x.color}` }), x.label);
}

function renderFilters() {
  const f = $("filters");
  if (state.tab === "requests") {
    const sel = el("select", { "aria-label": "Answer type", onchange: (e) => { state.method = e.target.value; state.offset = 0; loadLog(); } },
      [["", "All answers"], ["template", "From tables"], ["llm", "LLM answers"], ["abstain", "Refused"]].map(([v, l]) =>
        el("option", { value: v, selected: state.method === v, text: l })));
    f.replaceChildren(sel);
  } else {
    const box = el("input", { type: "checkbox", checked: state.errorsOnly, onchange: (e) => { state.errorsOnly = e.target.checked; loadLog(); } });
    f.replaceChildren(el("label", {}, box, " Failed calls only"));
  }
}

async function loadLog(append = false) {
  const box = $("log");
  try {
    if (state.tab === "requests") {
      const rows = await api(`/v1/admin/requests?limit=50&offset=${state.offset}${state.method ? `&method=${state.method}` : ""}`);
      const head = ["Time", "Key", "Question", "Answer", "Model", "Calls", "Tokens", "Spend", "Latency"];
      const body = rows.map((r) => el("tr", { class: "clickable", tabindex: 0, onclick: () => openTrace(r.id), onkeydown: (e) => { if (e.key === "Enter") openTrace(r.id); } },
        el("td", { text: when(r.asked_at) }), el("td", { text: r.client }),
        el("td", { class: "q" }, el("span", { lang: isTelugu(r.question) ? "te" : "en", text: r.question }), r.standalone_question ? el("small", { text: `as: ${r.standalone_question}` }) : null),
        el("td", {}, methodChip(r.method), r.reason ? el("small", { class: "note", text: ` ${REASONS[r.reason] || r.reason}` }) : null),
        el("td", { text: r.model || "–" }), el("td", { class: "num", text: fmtInt(r.llm_calls) }),
        el("td", { class: "num", text: r.prompt_tokens + r.completion_tokens ? fmtTok(r.prompt_tokens + r.completion_tokens) : "–" }),
        el("td", { class: "num", text: r.cost_inr ? fmtInr(r.cost_inr) : "–" }), el("td", { class: "num", text: fmtMs(r.latency_ms) })));
      const table = append && box.querySelector("tbody") ? null : el("div", { class: "tbl-wrap" }, el("table", { class: "tbl" },
        el("thead", {}, el("tr", {}, head.map((h, i) => el("th", { class: i >= 5 ? "num" : "", text: h })))), el("tbody", {})));
      if (table) box.replaceChildren(table);
      box.querySelector("tbody").append(...body);
      box.querySelector(".more")?.remove();
      if (!rows.length && !append) box.replaceChildren(el("p", { class: "empty", text: "No requests yet." }));
      else if (rows.length === 50) box.append(el("button", { type: "button", class: "more", text: "Load older requests", onclick: () => { state.offset += 50; loadLog(true); } }));
    } else {
      const rows = await api(`/v1/admin/calls?limit=200${state.errorsOnly ? "&errors=true" : ""}`);
      const head = ["Time", "Purpose", "Provider", "Model", "Status", "Tokens in", "Tokens out", "Spend", "Latency"];
      box.replaceChildren(rows.length ? el("div", { class: "tbl-wrap" }, el("table", { class: "tbl" },
        el("thead", {}, el("tr", {}, head.map((h, i) => el("th", { class: i >= 5 ? "num" : "", text: h })))),
        el("tbody", {}, rows.map((c) => el("tr", { class: c.turn_id ? "clickable" : "", onclick: c.turn_id ? () => openTrace(c.turn_id) : null },
          el("td", { text: when(c.at) }), el("td", { text: c.purpose.replace("_", " ") }), el("td", { text: c.provider }), el("td", { text: c.model }),
          el("td", { class: c.status === "ok" ? "status-ok" : "status-bad", title: c.error || "", text: c.status === "ok" ? "OK" : c.status === "quota" ? "Out of credits" : "Failed" }),
          el("td", { class: "num", text: fmtInt(c.prompt_tokens) }), el("td", { class: "num", text: fmtInt(c.completion_tokens) }),
          el("td", { class: "num", text: fmtInr(c.cost_inr) }), el("td", { class: "num", text: fmtMs(c.latency_ms) }))))))
        : el("p", { class: "empty", text: state.errorsOnly ? "No failed calls." : "No LLM calls yet." }));
    }
  } catch (err) {
    box.replaceChildren(el("p", { class: "empty", text: `Couldn't load: ${err.message}` }));
  }
}

// ------------------------------------------------------------------ trace drawer
async function openTrace(id) {
  const body = $("trace-body");
  body.replaceChildren(el("p", { class: "empty", text: "Loading…" }));
  $("trace").hidden = false;
  $("trace-close").focus();
  let t;
  try { t = await api(`/v1/admin/requests/${id}`); } catch (err) {
    body.replaceChildren(el("p", { class: "empty", text: `Couldn't load: ${err.message}` }));
    return;
  }
  $("trace-meta").textContent = `${when(t.asked_at)}, key ${t.client}${t.request_id ? `, request ${t.request_id.slice(0, 12)}` : ""}`;
  const kv = (pairs) => el("dl", { class: "kv" }, pairs.filter(([, v]) => v !== null && v !== undefined && v !== "").flatMap(([k2, v]) => [el("dt", { text: k2 }), el("dd", { text: String(v) })]));

  const timings = t.timings || {};
  const total = timings.total || t.latency_ms || 1;
  const stages = [
    ["Follow-up rewrite", timings.rewrite, false],
    ["Routing", timings.retrieve_route, false],
    ["Calendar search", timings.retrieve_panchanga, false],
    ["Text search", timings.retrieve_general, false],
    ...Object.keys(timings).filter((k2) => k2.startsWith("llm_")).sort().map((k2) => [`LLM attempt ${k2.slice(4)}`, timings[k2], true]),
    ["End to end", total, false],
  ].filter(([, v]) => v != null);

  body.replaceChildren(
    el("section", {}, el("div", { class: "qa" },
      el("p", {}, el("b", { text: "Q: " }), el("span", { lang: isTelugu(t.question) ? "te" : "en", text: t.question })),
      t.standalone_question && t.standalone_question !== t.question ? el("p", { class: "note", text: `Answered as (${t.rewrite_method}): ${t.standalone_question}` }) : null,
      el("p", { lang: isTelugu(t.answer) ? "te" : "en", text: t.answer }))),
    el("section", {}, el("h3", { text: "Outcome" }), kv([
      ["Answer type", { template: "From tables (no LLM)", llm: "LLM answer", abstain: "Refused" }[t.method] || t.method],
      ["Refusal reason", t.abstained ? t.reason : null], ["Route", t.route], ["Language", t.language],
      ["Model", t.model], ["Tokens", t.prompt_tokens + t.completion_tokens ? `${fmtInt(t.prompt_tokens)} in, ${fmtInt(t.completion_tokens)} out` : null],
      ["Spend", t.cost_inr ? fmtInr(t.cost_inr) : "₹0"], ["Latency", fmtMs(t.latency_ms)]])),
    el("section", {}, el("h3", { text: "Timeline" }), el("ul", { class: "waterfall" }, stages.map(([label, v, isLlm]) =>
      el("li", {}, el("span", { text: label }), el("span", { class: "track" }, el("span", { class: `seg-bar${isLlm ? " llm" : ""}`, style: `width:${Math.min(100, (v / total) * 100)}%` })),
        el("span", { class: "val", text: fmtMs(v) }))))),
    el("section", {}, el("h3", { text: `LLM calls (${t.llm_calls.length})` }), t.llm_calls.length ? el("div", { class: "tbl-wrap" }, el("table", { class: "tbl" },
      el("thead", {}, el("tr", {}, ["Purpose", "Model", "Status", "In", "Out", "Spend", "Latency"].map((h, i) => el("th", { class: i >= 3 ? "num" : "", text: h })))),
      el("tbody", {}, t.llm_calls.map((c) => el("tr", {},
        el("td", { text: c.purpose.replace("_", " ") }), el("td", { text: `${c.provider}/${c.model}` }),
        el("td", { class: c.status === "ok" ? "status-ok" : "status-bad", title: c.error || "", text: c.status === "ok" ? "OK" : c.error || c.status }),
        el("td", { class: "num", text: fmtInt(c.prompt_tokens) }), el("td", { class: "num", text: fmtInt(c.completion_tokens) }),
        el("td", { class: "num", text: fmtInr(c.cost_inr) }), el("td", { class: "num", text: fmtMs(c.latency_ms) }))))))
      : el("p", { class: "note", text: "None: answered without an LLM." })),
    el("section", {}, el("h3", { text: `Sources given to the model (${t.sources.length})` }), t.sources.length ? el("div", { class: "tbl-wrap" }, el("table", { class: "tbl" },
      el("thead", {}, el("tr", {}, ["Kind", "Section", "Pages", "Date", "Score"].map((h, i) => el("th", { class: i >= 2 ? "num" : "", text: h })))),
      el("tbody", {}, t.sources.map((s) => el("tr", {},
        el("td", { text: `${s.collection}/${s.kind}` }), el("td", { title: s.heading || "", text: s.heading || s.section || "–" }),
        el("td", { class: "num", text: (s.pages || []).join(", ") }), el("td", { class: "num", text: s.date || "–" }),
        el("td", { class: "num", text: s.score != null ? s.score.toFixed(3) : "–" }))))))
      : el("p", { class: "note", text: "No sources were found." })),
  );
}

// ------------------------------------------------------------------ loading & wiring
async function refresh() {
  try {
    renderOverview(await api(`/v1/admin/overview?hours=${state.hours}`));
    if (state.tab === "calls" || state.offset === 0) await loadLog();
  } catch (err) {
    if (err.status === 401 || err.status === 403) return signOut(err.status === 403 ? "This key can chat but can't open monitoring. Use an admin key." : "This key was not accepted.");
    $("updated").textContent = `Couldn't refresh: ${err.message}`;
  }
}

function schedule() {
  clearInterval(state.timer);
  if ($("auto").checked) state.timer = setInterval(() => { if (!document.hidden) refresh(); }, 30000);
}

function signOut(message) {
  try { localStorage.removeItem(KEY_STORE); } catch { /* storage blocked */ }
  clearInterval(state.timer);
  state.key = null;
  $("dash").hidden = true;
  $("gate").hidden = false;
  $("gate-error").hidden = !message;
  $("gate-error").textContent = message || "";
  $("gate-key").focus();
}

async function start(key) {
  state.key = key;
  const me = await api("/v1/whoami");
  if (!me.is_admin) {
    const err = new Error("not admin");
    err.status = 403;
    throw err;
  }
  try { localStorage.setItem(KEY_STORE, key); } catch { /* storage blocked */ }
  $("gate").hidden = true;
  $("dash").hidden = false;
  renderFilters();
  await refresh();
  schedule();
}

function wire() {
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    state.hours = Number(b.dataset.hours);
    refresh();
  }));
  $("auto").addEventListener("change", schedule);
  $("sign-out").addEventListener("click", () => signOut());
  $("trace-close").addEventListener("click", () => { $("trace").hidden = true; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("trace").hidden = true; });
  for (const [id, tab] of [["tab-requests", "requests"], ["tab-calls", "calls"]]) {
    $(id).addEventListener("click", () => {
      state.tab = tab;
      state.offset = 0;
      $("tab-requests").setAttribute("aria-selected", String(tab === "requests"));
      $("tab-calls").setAttribute("aria-selected", String(tab === "calls"));
      renderFilters();
      loadLog();
    });
  }
  $("gate-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try { await start($("gate-key").value.trim()); } catch (err) {
      signOut(err.status === 403 ? "This key can chat but can't open monitoring. Use an admin key." : "This key was not accepted.");
    }
  });
}

wire();
let saved = null;
try { saved = localStorage.getItem(KEY_STORE); } catch { /* storage blocked */ }
if (saved) start(saved).catch(() => signOut());
else signOut();
