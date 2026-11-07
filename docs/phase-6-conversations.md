# Phase 6: Conversations and Sessions

Status: **implemented, awaiting approval**  ·  Date: 2026-11-07

## 1. Results

Multi-turn eval (`eval/conversation_eval.py`, 10 conversations, 22 turns, 12 follow-ups, Telugu + English):

| Metric | Value |
|---|---|
| Follow-ups rewritten into the right standalone question | **12 / 12 (100%)**, all by free rules (no LLM call) |
| Follow-ups answered correctly | **10 / 12 (83%)** |
| All turns answered correctly | 19 / 22 (86%) |
| Off-topic question inside a conversation ("capital of France?") | refused |
| Cost of the whole run | ₹0.88 (rewrites ₹0) |

Single-turn regression (`eval/answer_eval.py`) after this phase's fixes: **15/17 answered correctly, 5/5 refusals,
0 invented answers**, same as Phase 5. Two Phase 5 failures (griha pravesham months, head omen) are now fixed;
Kanya yearly predictions needed a glossary fix (below).

## 2. How it works
```
question ─▶ follow-up resolver ─▶ standalone question ─▶ AnswerService (Phase 5, unchanged) ─▶ answer
              ▲  rules first, LLM only if needed                                │
              └──────── last 3 turns of this session (Postgres) ◀───────────────┘ stored
```
- **Follow-ups are rewritten into standalone questions** by editing the previous question. Retrieval,
  verification and citations stay single-turn. Earlier *answers* are never shown to the answering model,
  so one wrong answer cannot leak into the next.

| Follow-up | Previous question | Answered as |
|---|---|---|
| "and tomorrow?" | What is today's tithi? | What is tomorrow's tithi? |
| "రేపు?" | ఈరోజు తిథి ఏమిటి? | రేపు తిథి ఏమిటి? |
| "and the nakshatram?" | What is tomorrow's tithi? | What is tomorrow's nakshatram? |
| "and Purnima?" | When is the next Ekadashi? | When is the next Purnima? |
| "దాని ప్రాముఖ్యత ఏమిటి?" | వచ్చే ఏకాదశి ఎప్పుడు? | ఏకాదశి ప్రాముఖ్యత ఏమిటి? |
| "కన్యా రాశికి?" | మకర రాశి వారికి ఉద్యోగ ఫలితాలు…? | కన్యా రాశి వారికి ఉద్యోగ ఫలితాలు…? |
| "what is the tithi on that day?" | When is Deepavali? (answer cited 8 Nov) | what is the tithi on 8 November 2026? |

- **"That day"** becomes the date the previous question named, or else the date its answer cited.
- **LLM fallback** (`sarvam-105b-conversations`, ~₹0.01) only for follow-ups that rules can't place
  (e.g. "what are its remedies?" after a long prose question). Its output is rejected if it switches language.
  `PSETU_FOLLOWUP_LLM_REWRITE=false` turns it off.
- **Answer language** follows what the user typed in *this* turn, even when the rewrite borrowed words from an
  earlier turn in another language.
- **Context expires** after 60 idle minutes (`PSETU_CONVERSATION_IDLE_MINUTES`); only the last 3 turns are used
  (`PSETU_CONVERSATION_HISTORY_TURNS`). A question that names its own subject is never rewritten.
- **Storage:** `chat_sessions` (with an `owner` column for Phase 7 API clients) and `chat_turns`. Each turn stores
  the question, its standalone form, how it was rewritten (none/rule/llm), the answer, citations, model, cost
  and latency. Migration `65221725010b`.

## 3. Try it
```bash
uv run psetu db-upgrade
uv run psetu chat -v                 # shows "(understood as: …)" when a follow-up was rewritten; /new resets
uv run psetu chat --session <id>     # resume a conversation
uv run python eval/conversation_eval.py            # rewrite check: free
uv run python eval/conversation_eval.py --answer   # end to end: ~₹1
```

## 4. Problems found by testing conversations (all fixed)
Follow-ups such as "and Tula?" exposed retrieval gaps the single-turn eval had missed:
- **Tula and Mithuna sections had no heading:** OCR tagged their titles as plain paragraphs. Short
  "…రాశి ఫలితములు" title blocks are now promoted to headings. All 12 signs are now headed.
- **Monthly results lost their sign:** every sign's "నెలవారీ ఫలితములు" replaced the sign's title, so the rasi
  filter threw them away. They now nest under the sign ("కన్యారాశి ఫలితములు > నెలవారీ ఫలితములు").
  Karkataka's title is OCR'd as "కర్నాటక"; it was added as an alias. Rechunked (`chunk-7`) and reindexed with
  0 OCR calls.
- **"Next Purnima" skipped 24 Nov:** the book's row for that day (Karthika Deepotsavam) is unreadable, and an
  empty book row still beat the readable wall-calendar row. A row with no data now loses, and the template
  lists such days with "(wall calendar only; not verified against the book)" instead of handing them to the LLM.
- **Rules crowded out by dated lists:** for "which months are good for griha pravesham?" the 5 source slots
  filled up with muhurtham lists, and the rule on p.92 never reached the model. Sources now alternate between
  the two collections.
- **Glossary hint misled the model:** "yearly predictions = సంవత్సర ఫలితములు", but each sign's section is titled
  "<sign>రాశి ఫలితములు", so the model concluded there was no yearly prediction. Fixed.
- "According to [S1], …" left a dangling "According to," after citation markers were stripped. Fixed.

## 5. Outstanding
1. **Rahukalam/yamagandam on a date:** the weekday table (p.83–84) has three side-by-side column groups that the
   fast model sometimes misreads or refuses. Both depend only on the weekday, so a small template from that
   table would make them exact and free. **Recommended next fix.**
2. **Run-to-run variance of the fast model** on long prose: the same question can be answered in one run and
   refused in the next (Kuja dosha "when is it not a problem"). Benchmarking `gpt-4.1-mini` (needs OpenAI
   credits) or self-consistency would address this.
3. **Not supported yet:** "the one after that" (needs the next-but-one occurrence), comparisons across turns
   ("which of those is earlier?"), and relative dates in a conversation that crosses midnight (they follow the
   new day).
4. **Session retention:** turns are kept indefinitely; a prune job (and per-owner deletion) belongs with the
   Phase 7 API and Phase 8 operations work.
5. The eval sets are still small and were written while building (22 + 22 turns). Phase 9 needs held-out sets.
