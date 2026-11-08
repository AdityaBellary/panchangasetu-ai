# LLM providers: Sarvam vs Gemini (measured 2026-11-08)

The answer chain is `PSETU_LLM_PROVIDERS` (default `["sarvam","gemini","openai"]`). Each provider is tried in
order, and running out of credits/quota or an outage falls through to the next. Every call is recorded
(see `docs/monitoring.md`), so all numbers below come from those records.

## Held-out prose questions (answers grounded in the book, checked by the fact checker)
| | Sarvam `sarvam-105b-conversations` | Gemini (`gemini-3.6-flash`, falling back to `3.5-flash-lite`) |
|---|---|---|
| prose v1 (19 answerable + 4 to refuse) | 13/19, refusals 4/4 | **16/19**, refusals 4/4 |
| prose v2 (12) | 8/12 | **10/12** |
| Invented answers | 0 | 0 |
| Cost, prose v1 run | ₹1.41 | ₹6.39 (~4.5×) |
| Median latency | ~0.5 s | ~4.2 s |

The Sarvam runs predate three fact-checker fixes (number words, ISO dates read as times, "November 2026"
read as 20 November), so its true score is somewhat higher. All three caused false refusals regardless of
model. Gemini's remaining misses: three retrieval gaps both models share (Sanskrit terms asked in English), and
two answers where it added Gregorian dates the book doesn't print, which the checker blocked.

## Your questions (`eval/datasets/user_questions.jsonl`)
| Question | Sarvam | Gemini |
|---|---|---|
| ఉగాది పండుగ ఎప్పుడు? | ✅ 19 Mar 2026 / 7 Apr 2027, ₹0.08, 0.9 s | ✅ same, ₹0.33, 3.9 s |
| ఏలినాటి శని ప్రభావం… | ✅ ₹0.07, 1.1 s | ✅ fuller, ₹0.18, 2.3 s |
| Kumbha rasi Nov & Dec 2026 | ✅ after the checker fix (pp.39–40), ₹0.05 | refused before the fix (same bug) |
| Tomorrow good for a new business? | refused | refused |

The book has no rule for starting a business (only the Śukra ghaḍiya and Gowri panchangam tables mention
trade), so the refusal is defensible. A friendlier reply would give tomorrow's panchangam and say so.

## Gemini operational notes (2026-11-08)
- **`gemini-3.8-flash` was overloaded** (503 "high demand") all day, and one retry hung for the full 60 s timeout.
  The default is now `gemini-3.6-flash` (same price) with `gemini-3.5-flash-lite` as fallback, and one quick
  retry per model. Switch back with `PSETU_GEMINI_CHAT_MODEL=gemini-3.8-flash` when it's stable.
- **Quota:** after about 15 answers, the key returned 429 "You exceeded your current quota, please check your
  plan and billing details" for `gemini-3.6-flash`. Only Flash-Lite kept working. Check the key's plan and
  limits in AI Studio. These errors now fail over to the next model immediately instead of being retried.
- **Thinking tokens:** Gemini 3 models always think. Thinking tokens are billed as output but left out of
  `completion_tokens`, so cost is computed from `total_tokens − prompt_tokens`. Gemini also gets 2,048 tokens
  of extra output allowance so thinking can't crowd out the answer.
- **Error bodies** arrive as a JSON list (`[{"error": …}]`); this crashed the error handler until it was fixed.
- **Prices** (`config/llm_pricing.yaml`): Flash $0.75 in / $3.75 out per 1M tokens until 31 Dec 2026, then
  doubling; Flash-Lite $0.30 / $2.50. At ₹88 to the dollar a typical answer costs ₹0.2–0.4, against Sarvam's ₹0.05–0.1.

## Recommendation
Keep **Sarvam as the primary model** (4–5× cheaper, ~8× faster, and now close in accuracy after the checker
fixes), and use **Gemini where Sarvam can't produce a verified answer although relevant sources exist**
(escalation). That gets Gemini's quality on the hard questions while paying its price only for them.

This is now the default (`PSETU_ESCALATION_PROVIDER=gemini`): when Sarvam's answer fails the checks or it
judges the sources insufficient, the question is asked once more of the Gemini chain (`gemini_chat_model`,
then `gemini_fallback_models`). The extra call is recorded with purpose `escalation`, so the dashboard shows
how often it happens and what it costs. It is skipped when Sarvam was unavailable and Gemini already
answered as the fallback. Set `PSETU_ESCALATION_PROVIDER=""` to turn it off, or `sarvam` to use
`PSETU_ESCALATION_MODEL` (sarvam-105b) as before.
