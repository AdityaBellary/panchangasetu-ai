# Monitoring and observability

Three layers, from "what is this one request doing?" up to "how is the service doing this month?".

## 1. Per-request and per-call records (always on)
Every answer served through the API or the chat UI is stored with its full trace (`chat_turns`), and every
LLM HTTP call with its own row (`llm_calls`). That includes retries, fallbacks to another provider, follow-up
rewrites and escalations.

| Recorded | Where |
|---|---|
| Question, standalone rewrite, answer, language, route, answer type, refusal reason | `chat_turns` |
| Stage timings: rewrite, routing, calendar search, text search, each LLM attempt, end to end | `chat_turns.timings` |
| Sources shown to the model (section, pages, date, similarity score) and citations | `chat_turns.sources`, `.citations` |
| Request id (also in the `x-request-id` response header and every log line) | `chat_turns.request_id`, `llm_calls.request_id` |
| Per LLM call: purpose (answer / answer_retry / escalation / followup_rewrite), provider, model, prompt and completion tokens, ₹ cost, latency, ok / error / out of credits, error text | `llm_calls` |

- **Cost** per call = tokens × the model's price in `config/llm_pricing.yaml` (₹, or USD × `usd_to_inr`).
  Reasoning ("thinking") tokens bill as output, and calls that fail after being billed still count.
- **Spend history is append-only.** Deleting or pruning a conversation keeps its LLM calls, so cost history
  and the daily budget can't be reset that way.

## 2. Monitoring dashboard (in the app)
```bash
uv run psetu api-keys create ops --admin      # an admin key: chat + monitoring
uv run psetu serve                            # then open http://127.0.0.1:8000/dashboard.html
```
(Admins also see a "Monitoring" link in the chat sidebar.)

- **Headline numbers:** questions, answer rate, LLM spend and spend per LLM answer, today's spend against the
  daily budget, LLM calls and failures, tokens in/out, latency median and p95.
- **Charts** (24 h / 7 d / 30 d, auto-refresh): questions by answer type, spend over time, latency over time,
  where the time goes per stage, why questions were refused, language and follow-up mix, routes.
- **Tables:** every provider/model (calls, failures, tokens, spend, cost per call, median and p95 latency) and
  every API key (questions, spend).
- **Request log** with filters. Click any request for its **trace**: question and answer, outcome, a timeline
  of every stage, each LLM call, and the sources the model saw.
- **LLM call log**, with a "failed calls only" filter (out of credits, outages, rate limits, empty replies).

API behind it (admin keys only): `/v1/admin/overview?hours=`, `/v1/admin/requests`,
`/v1/admin/requests/{id}`, `/v1/admin/calls?errors=true`.

## 3. Prometheus + Grafana (time series and alerts)
```bash
echo "PSETU_API_METRICS_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
docker compose --profile monitoring up -d       # Prometheus 127.0.0.1:9090, Grafana 127.0.0.1:3000 (admin/admin)
```
Prometheus scrapes `/metrics` from the API container, or from the host when you run `psetu serve --host 0.0.0.0`.
Grafana opens with the **PanchangaSetu** dashboard (16 panels): questions and answer rate, spend, budget used,
failed calls, answers per minute by type, latency median and p95, spend per hour and tokens per minute by
model, LLM call latency by model, calls by purpose and outcome, stage latency, refusals by reason, and HTTP
traffic and latency by route. If ports 9090/3000 are taken, set `PSETU_PROMETHEUS_PORT` / `PSETU_GRAFANA_PORT`.

Metrics (all labels low-cardinality; no question text, session ids or keys):
`psetu_http_requests_total`, `psetu_http_request_seconds`, `psetu_answers_total`, `psetu_answer_seconds`,
`psetu_abstentions_total{reason}`, `psetu_llm_calls_total{provider,model,purpose,status}`,
`psetu_llm_call_seconds`, `psetu_llm_tokens_total{provider,model,kind}`, `psetu_llm_cost_inr_total`,
`psetu_stage_seconds{stage}`, `psetu_llm_budget_inr`, `psetu_llm_spent_today_inr`, `psetu_followup_rewrites_total`.

**Alert rules** (`monitoring/alerts.yml`, shown under Alerts): API down, 5xx errors, over 80% of the daily
budget spent, over 20% of LLM calls failing, p95 answer latency above 8 s, more than half of questions refused.
Connect a Grafana contact point or an Alertmanager to be notified.

## 4. Logs and health
JSON logs (structlog): one `api.request` line per API call and one `answer.done` line per answer. The latter
carries method, route, language, refusal reason, model, LLM calls, tokens, cost and latency, plus the
question's length but never its text. `/healthz` is liveness; `/readyz` checks the database, vector index and LLM.

## Verified
On 2026-11-08: the dashboard rendered against live traffic, including a trace showing an answer, its
fact-check failure and the paid retry. Prometheus scraped the API with the token, all 6 alert rules evaluated,
and Grafana auto-provisioned the dashboard.
