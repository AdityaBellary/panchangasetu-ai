# Phase 7: HTTP API, API Keys and Web Chat

Status: **implemented, awaiting approval**  ·  Date: 2026-11-07

## 1. Run it
```bash
uv run psetu db-upgrade                  # adds the api_keys table
uv run psetu api-keys create web         # prints a key once; only its SHA-256 is stored
uv run psetu serve                       # http://127.0.0.1:8000/  (API docs: /docs)
```
Open the page, paste the key once (it stays in that browser's local storage), and chat.
`psetu api-keys list` / `psetu api-keys revoke <prefix>` manage keys.

## 2. Web chat
- **Today panel:** tonight's moon drawn from today's actual tithi, the tithi in large Telugu with masa and paksha,
  nakshatra, rahukalam and sunrise, read from the same tables the answers use. It needs no LLM call and costs nothing.
- **Conversation:** Telugu and English, with follow-ups ("and tomorrow?"). When a follow-up was rewritten the
  answer says "Answered as …".
- **Sources under every answer:** a turmeric seal marks a calendar entry verified against the book, a red ring
  marks wall-calendar-only data, and a plain mark is a prose passage. Clicking a source opens the **actual book
  page** (images saved at ingestion; no PDF is touched per request) and the quotes the verifier matched word for
  word against it.
- "Not in the documents" answers are marked and suggest what the documents do cover.
- Past conversations are listed in the sidebar and can be reopened or deleted. "New conversation" starts fresh.
- Light and dark themes, phone layout, keyboard focus, reduced motion. No external fonts, scripts or CDNs; a
  strict Content-Security-Policy, and all server text inserted as text (never as HTML).

Design: lapis night for structure, a cool moonlit page, turmeric for "verified", and kumkum red only where the
documents could not answer. Telugu text uses a Telugu-first font stack (Kohinoor/Noto Sans Telugu): falling
back glyph by glyph from a Latin face mis-shapes vowel signs.

## 3. API
| Endpoint | Purpose |
|---|---|
| `POST /v1/chat` `{question, session_id?}` | Answer; starts or continues a session. Returns answer, language, standalone question, citations (with page-image URLs and verified quotes), method, model, cost, latency |
| `GET /v1/sessions`, `GET/DELETE /v1/sessions/{id}` | This key's conversations |
| `GET /v1/pages/{document}/{page}` | Source page image |
| `GET /v1/today` | Today at a glance |
| `GET /healthz`, `GET /readyz` | Liveness; readiness (database, vector index, LLM configured) |

- **Auth:** `Authorization: Bearer psk_…` or `X-API-Key`. Keys are 256-bit random values; only a SHA-256 is
  stored, and each key is shown once at creation. Revoked keys stop working immediately.
- **Isolation:** a session belongs to the key that created it; any other key gets 404 (not 403), so session ids
  don't reveal that a conversation exists.
- **Limits:** 30 requests per minute per key by default (`PSETU_API_RATE_LIMIT_PER_MINUTE`, or per key with
  `api-keys create --rate-limit`); questions up to 500 characters.
- Every response has an `x-request-id`; each `/v1` request is logged with status and latency (never the key).

## 4. Tests
229 tests pass (82% coverage): 401 without or with a bad key, both header styles, session continuation, cross-key
404s on read/continue/delete, per-key rate limits, validation, page-image access, readiness, CSP header, hashed key
storage and revocation, and the tithi-to-moon-phase mapping. The UI was checked in a browser: sign-in, template
and LLM answers, a follow-up, a refusal, the source drawer with page images, light and dark themes, and phone width.

Found while testing: with `from __future__ import annotations`, FastAPI could not resolve the auth dependency
declared inside the app factory and treated it as a query parameter (every call returned 422). The app module no
longer postpones annotations.

## 5. Outstanding
1. **Highlight the row:** for a calendar answer, the drawer shows the whole daily-table page. The day's row
   bounding box is already stored, so the next step is to crop or highlight it.
2. **Rate limits are per process** (in memory). Running several API processes needs a shared store (Phase 8).
3. **Streaming answers:** answers arrive whole (template ~5 ms, LLM ~0.5–2 s), so streaming isn't needed yet.
4. **Admin UI for keys:** keys are managed from the CLI only.
5. **Session retention:** turns are kept until deleted; a retention job belongs in Phase 8.
