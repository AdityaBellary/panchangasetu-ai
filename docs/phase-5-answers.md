# Phase 5: Grounded Multilingual Answers

Status: **implemented, awaiting approval**  ·  Date: 2026-11-07

## 1. Results (`eval/answer_eval.py`, 22 questions, Telugu + English)

| Metric | Value |
|---|---|
| Answerable questions answered correctly (expected facts present) | **15 / 17 (88%)** |
| Unanswerable questions correctly refused | **5 / 5 (100%)**: 0 invented answers |
| Answer language matches the question | **100%** |
| Cost for the whole run | ₹0.83 (templates and early abstentions cost ₹0) |
| Latency median / p95 | 0.42 s / 1.1 s |

Answer paths, cheapest and safest first:

| Path | When | LLM? | Cost | Example |
|---|---|---|---|---|
| **Abstain** | no evidence, or date outside every source | no | ₹0 | "tithi on 15 Aug 2030" |
| **Template** | verified day lookup, or next tithi/nakshatra | no | ₹0, ~5–60 ms | "ఈరోజు తిథి" → *త్రయోదశి ఉదయం 10:29 వరకు, తరువాత చతుర్దశి* |
| **LLM** | everything else (rules, predictions, festivals) | `sarvam-105b-conversations` | ~₹0.03–0.11 | Kuja dosha, Kanya rasi phalalu, Deepavali |

## 2. How answers are kept honest
1. **Closed-book prompt** with numbered sources `[S1..Sn]`: answer only from them, cite them, keep every
   condition exactly as stated (translate rule sentences one by one, don't summarise), else return
   `sufficient: false`.
2. **Mechanical verifier** on every LLM answer (no second LLM call):
   - every **time** and **date** must appear in the cited sources (in any notation: 10:29 AM, ఉ.10-29,
     "డిసెంబర్ 5వ తేదీ"; month-first Telugu dates included);
   - every **number or ordinal** must appear as digits **or** a Telugu ordinal word (దశమ = 10th);
   - **1–4 verbatim quotes** must be found in the sources (spacing- and OCR-variant tolerant, ≥ 85%
     character 4-gram containment). Quotes are returned with each citation so users can check them;
   - failure → one corrective retry → otherwise the fixed abstention message (English or Telugu).
3. **Wrong-entity guard at retrieval:** if the question names a rasi, only that rasi's chunks are used.
4. **Cross-language anchors:** for English questions, glossary hits ("chin = గడ్డము") are shown to the model.

## 3. Model choice (measured, 2026-11-07)
| Model | Latency | Cost/answer | Faithfulness on hard list/rule lookups |
|---|---|---|---|
| **`sarvam-105b-conversations`** (default) | 0.3–2 s | ₹0.03–0.11 | Good on prose/rules; **misreads dense lists sometimes** (e.g. right eye → left eye's entry) |
| `sarvam-105b` (reasoning) | ~20 s | ~₹0.3 | Read the lizard table correctly, but **needs > 4k reasoning tokens**, and often returned empty output at 4k. Available as `PSETU_ESCALATION_MODEL`, off by default |
| `gpt-4.1-mini` (OpenAI fallback) | n/a | n/a | **Not tested: the OpenAI account still reports no credits.** Configured as fallback and worth benchmarking as primary |

## 4. Bugs and data issues found by testing answers (all fixed)
- **Phase 2:** Sarvam's `section-title` tag (165 blocks) was not treated as a heading, so all rasi sections
  lost their titles. Fixed, and 69 pages were rebuilt **from stored OCR payloads at no cost** (`psetu ingest --rederive`).
- Generated picture descriptions under non-image tags ("This image contains no discernible text…") are now dropped.
- **Two-column lists (p89 omen tables)** are now OCR'd column by column with a full-width title band (curator rule
  `columns: 2`). This **recovered a whole table that whole-page OCR had silently skipped** (head positions).
- Chunks shown to the LLM didn't include their section heading, so the model couldn't tell which rasi a chunk was about.
- A "next Ekadashi" LLM answer skipped the actual next date; FIND questions now use an exact template.
- Rasi mix-ups (Tula facts in a Kanya answer) are now prevented by entity filtering at retrieval.
- The verifier rejected correct answers: Telugu ordinal words, month-first dates, quotes attributed to the
  wrong source id, and "(S2)" read as the number 2. All fixed.
- An over-eager prompt rule ("only use sources whose section matches") made the model reject a correct but
  mis-headed chunk. Removed; the rasi guard now lives in retrieval.

## 5. Try it
```bash
uv run psetu chat -v            # terminal chat (Telugu or English); -v shows method, cost, latency
uv run psetu ask "ఈరోజు నక్షత్రం ఏమిటి?" "What is Kuja dosha?"
uv run python eval/answer_eval.py   # ~₹1 of Sarvam credits
```
Settings: `PSETU_LLM_PROVIDERS=["sarvam","openai"]`, `PSETU_SARVAM_CHAT_MODEL`, `PSETU_OPENAI_CHAT_MODEL`,
`PSETU_ESCALATION_MODEL`.

## 6. Outstanding
1. **Faithfulness on dense lists** is the main quality gap: 1 wrong answer in the eval, plus partial mixing in
   another. Options: benchmark `gpt-4.1-mini` as primary once OpenAI credits work; or add an LLM judge (adds cost).
2. **OCR completeness:** whole-page OCR can silently drop regions (found on p89). A glyph-count ratio is too coarse
   to detect it (p89 moved only 0.76 → 0.80). A line-level comparison against the exact text-layer numbers is the
   likely detector.
3. **Rule-change detection:** changing `collections.yaml` OCR rules needs `psetu ingest --rechunk`; a plain `ingest`
   skips unchanged files before checking page rules.
4. **No conversation memory yet** ("when is the next one?"): Phase 6. **No HTTP API or UI yet**: Phase 7.
5. The evaluation set is small (22 questions) and was built while debugging. It needs a larger held-out set (Phase 9).
