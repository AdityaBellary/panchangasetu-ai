# Phase 8: Operations (Observability, Budget, Retention, Docker, CI)

Status: **implemented, awaiting approval**  ·  Date: 2026-11-07

## 1. Observability
**Metrics:** `GET /metrics` (Prometheus format). Without `PSETU_API_METRICS_TOKEN` only loopback clients may read it
(403 otherwise); with it, scrapers must send the token as a bearer token (compared in constant time).

| Metric | Labels | Use |
|---|---|---|
| `psetu_http_requests_total`, `psetu_http_request_seconds` | method, route *template*, status | traffic, errors, latency per endpoint |
| `psetu_answers_total`, `psetu_answer_seconds` | method (template/llm/abstain), language | answer mix and speed |
| `psetu_abstentions_total` | reason (no_evidence, date_not_covered, insufficient_sources, verification_failed, llm_unavailable, budget, …) | why questions go unanswered |
| `psetu_llm_tokens_total`, `psetu_llm_cost_inr_total` | prompt/completion | spend |
| `psetu_followup_rewrites_total` | none/rule/llm | how often follow-ups need context |

Labels are low-cardinality by construction: route templates (`/v1/sessions/{session_id}`), never raw paths, session
ids or question text.

**Logs:** JSON lines via structlog. Each answer writes one `answer.done` event: session, method, route, language,
abstained and its reason, rewrite method, model, cost, latency and the question's *length*. The question text and
API keys are never logged. Each request gets an `x-request-id`, which appears in the response header and in its log line.

**Health:** `/healthz` (process up) and `/readyz` (database, vector index, LLM configured; 503 if any check fails).

## 2. Daily LLM budget
`PSETU_LLM_DAILY_BUDGET_INR` (default **₹25 per local day**, all users together; `0` = unlimited). Spend is read
from an append-only `llm_spend` ledger, written in the same transaction as each turn, so deleting or pruning
conversations can't lower it (an earlier version summed stored turns, which let a client reset the day's spend by
deleting its sessions). Requests already in flight when the limit is crossed can still finish, so the overshoot is
bounded by concurrent requests × ~₹0.1. Once it's reached:
- **calendar answers keep working** (tithi, nakshatram, rahukalam, next Ekadashi: templates cost ₹0);
- LLM questions get a clear message in the question's language ("Today's limit for detailed answers has been
  reached… ask this one again tomorrow"), not the misleading "not in the documents";
- follow-up rewriting falls back to rules only.

At ~₹0.05 per LLM answer, ₹25 is about 500 detailed answers a day.

## 3. Session retention
Conversations with no activity for `PSETU_SESSION_RETENTION_DAYS` (default 180; `0` keeps them forever) are deleted by
`psetu sessions prune [--days N]`, which also runs each time the server starts.

## 4. Docker
```bash
docker compose up -d --build api      # Postgres + Qdrant + API/web chat at http://127.0.0.1:8000
```
- One image for the API and the CLI: ingestion, indexing, `api-keys`. It contains code and config only; `./data`
  (PDFs, page images, OCR cache, embedding models) is mounted, and secrets come from `.env`.
- It runs as a non-root user (uid 10001), applies migrations on start, has a health check, and includes Tesseract
  Telugu for ingestion.
- **Memory: give Docker at least 6 GB.** The multilingual-e5-large model needs about 2.5 GB on its own. On this Mac,
  Docker Desktop has 2 GB in total, so the container was killed while loading the model (exit 137). Migrations,
  database access, the CLI and Tesseract were verified inside the container; serving needs the memory raised
  (Docker Desktop → Settings → Resources).
- It runs as a single process, so rate limits and metrics stay correct. Running several replicas would need a shared
  rate-limit store (Redis) and per-replica metric scraping. That isn't needed at this scale.

## 5. CI (GitHub Actions, `.github/workflows/ci.yml`)
On every push to `main` and every pull request, with no secrets:
1. **quality:** ruff lint, ruff format check, mypy;
2. **tests:** pytest against a Postgres service container with Tesseract Telugu installed. Sarvam-calling tests are
   excluded; tests needing local PDFs or models skip themselves;
3. **docker:** the image builds (not pushed).

Simulated locally in a copy of the repo with no `data/` and no `.env`: 230 passed, 4 skipped (data-dependent).

## 6. Tests
234 pass locally (83% coverage). New: budget behaviour (LLM blocked, templates unaffected, resets the next day),
spend tracking and pruning on both stores, answer metrics without question text, route-template metric labels, and
the metrics token.

## 7. Outstanding
1. Raise Docker Desktop memory to ≥ 6 GB to run the full stack in containers on this machine.
2. No dashboards or alerts are defined yet. Suggested alerts: `readyz` failing, 5xx rate, LLM spend near the budget,
   and a rising abstention rate for `verification_failed`.
3. Tracing (OpenTelemetry) was left out. With one process and request ids in every log line, logs plus metrics
   cover debugging for now.
