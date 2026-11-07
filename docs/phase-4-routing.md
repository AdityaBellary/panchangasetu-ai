# Phase 4: Query Routing & Retrieval

Status: **implemented, awaiting approval**  ·  Date: 2026-11-07  ·  Provider spend: **₹0 / $0**

A question becomes **evidence** (no answer generation yet; that's Phase 5):

```
question ─► language (te / en / romanised te) ─► dates + slots ─► router ─► retrieval ─► evidence
                                                     │                 │
                         "today", "రేపు", "15 Dec",   │   rules first,  │  PANCHANGA/DAY    exact day facts (SQL)
                         "next Friday", "ఈ నెల" ...   │   then semantic │  PANCHANGA/FIND   next tithi/nakshatra (SQL)
                         tithi / nakshatra names,     │   (e5 kNN);     │  PANCHANGA/SEARCH dated events + vector search
                         attributes, when / why       │   unsure→HYBRID │  GENERAL          vector search (+ expansion)
```

## 1. Results

| Metric | Value |
|---|---|
| Routing accuracy, intent (59 Te/En/romanised questions) | **96.6%** (rules alone: 88.1%, and every rules-only miss falls back to HYBRID) |
| Routing accuracy, intent + panchanga mode | **96.6%** |
| Retrieval recall@5 (64 questions, production index, with expansion) | **0.953** (EN 0.938, TE 0.969, Panchanga 1.0) |
| Retrieval MRR@10 | 0.765 |
| End-to-end retrieval latency (route + SQL + vector), typical | 20–65 ms |

Retrieval recall@5 went from 0.922 to **0.953** in this phase. The gains came from query expansion
and from two chunk-metadata fixes (headings keep the section title).

**Caveat:** I tuned the rules while looking at these evaluation sets, so the numbers are optimistic. A held-out set is planned for Phase 9.

### Example traces (`psetu retrieve --now 2026-11-07T09:00 …`)
| Question | Route | Top evidence |
|---|---|---|
| What is today's nakshatram? | panchanga/day | 7 Nov (p131): Trayodashi until 10:29, Chitta until 06:12 (8 Nov), **verified** |
| వచ్చే ఏకాదశి ఎప్పుడు? | panchanga/find | **20 Nov** Kartika Shukla Ekadashi (ends 21 Nov 02:04), then 4 Dec, verified |
| Festivals this week | panchanga/search | 7–8 Nov festival-table and utsavam events with dates |
| Lizard falls on the head? | general | బల్లిపాటు ఫలములు, p88–89 |
| Rahu kalam next Friday | panchanga/day | 13 Nov day facts **and** the weekday Rahukalam rule (p83–84) |
| Tithi on 15 August 2030? | panchanga/day | **no evidence + note "no panchanga data"**, so Phase 5 abstains |

## 2. Components (all local, deterministic, free)

| Module | Role |
|---|---|
| `query/language.py` | Answer language: Telugu script → te; English, even with Telugu terms like "nakshatram" → en; romanised Telugu ("ee roju tithi enti") → te |
| `query/dates.py` | today/tomorrow/yesterday (en, te, romanised), ISO and day-first numeric dates, English and Telugu month names (boundary-safe), weekdays, week/month ranges. "Today" is Asia/Kolkata via an injected clock |
| `query/slots.py` | Attributes (tithi … rahukalam, festival, muhurtham, karthe, transit), named tithi/nakshatra, when/why markers, General topics, residual words (festival vs. bare tithi) |
| `query/router.py` | Rules for certain cases, an e5 nearest-example router (`config/router_examples.yaml`) for the rest; below 0.65 confidence → HYBRID |
| `query/rewrite.py` | English → Telugu term expansion from `config/query_glossary.yaml` and the lexicon |
| `retrieval/structured.py` | Exact day facts (book preferred, wall calendar marked UNVERIFIED), next occurrences, events by range, coverage |
| `retrieval/orchestrator.py` | Route → evidence with provenance (document, page, date, verified flag), notes such as "no data", timings |
| `retrieval/evidence.py` | The evidence model Phase 5 will answer from, with citations |

## 3. Bugs found and fixed along the way
- **"మే" (May) matched inside "మేషరాశి"**, turning Mesha rasi questions into May date lookups. Telugu month names now need word boundaries (case suffixes like "మేలో" are still allowed).
- **"దుర్ముహూర్తం" contains "ముహూర్తం"**, so durmuhurtam lookups went to muhurtham search.
- English plurals ("muhurthams", "festivals") didn't match.
- **Festivals named after tithis** (రథసప్తమి, Sri Rama Navami, Vaikuntha Ekadasi) were treated as "next tithi" lookups. Leftover content words now mark them as festivals.
- **Header blocks lost their title.** A rasi section's chunks kept only the nakshatra-pada and aaya/vyaya lines, so all twelve rasi sections looked alike to search (`chunk-4`/`chunk-5`). Re-chunking was free.
- **Rahukalam questions** retrieved Rahu's *planetary transit* predictions. Attributes the daily table doesn't print (rahukalam, yamagandam) now trigger a targeted search for the weekday rule.

## 4. Outstanding
1. **3 retrieval misses:** Makara monthly (Telugu), tarpanam order (English), Shukla/Krishna paksha (English).
   LLM query rewriting in Phase 5/6 is the natural next lever.
2. **Romanised festival names** ("krishnashtami eppudu") fall to HYBRID at low confidence. That's safe but costs an extra search.
3. **Genuinely ambiguous questions** ("which months are good for gruhapravesam": rules or dates?) go to one side. HYBRID may be preferable.
4. **Follow-up resolution** ("when is the next one?") needs conversation context: Phase 6.
5. **Rahukalam/yamagandam are answered from the weekday rule text**, not a structured table. The wall calendars print the weekday Rahukalam and could seed one.

## 5. Try it
```bash
uv run psetu retrieve "ఈరోజు నక్షత్రం ఏమిటి?" "When is the next Ekadashi?" "What is Kuja dosha?"
uv run python eval/routing_eval.py
uv run python eval/retrieval_eval.py --expand
```
