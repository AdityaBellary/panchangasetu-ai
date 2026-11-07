# Phase 2b: Structured Calendar Facts (free methods)

Status: **implemented, awaiting approval**  ·  Date: 2026-11-07  ·  Provider spend this phase: **₹0**

Calendar questions ("today's nakshatram", "next Ekadashi", "when is Deepavali") are answered by
**exact date lookups** in two Postgres tables, never by vector search:

| Table | Rows | Source |
|---|---|---|
| `panchanga_days`: one day per source, every value with its verification status | 385 book days + 730 calendar days | book pp. 100–151; both wall calendars |
| `panchanga_events`: dated festivals and utsavams | 276 festival rows + 309 utsavam rows | book pp. 152–163 and 164–176 |

## 1. The key technique: the "broken" text layer still has exact numbers

The book's legacy fonts scramble Telugu letters, but **digits and punctuation sit at ASCII + 0xF000**
(`U+F036 U+F02D U+F032 U+F032` → `6-22`). So every time, date and ghati value in the PDF is
recoverable **exactly, without OCR** (`structured/legacy_text.py`). That makes three things possible:

1. **Row segmentation.** Each day row is anchored on its exact sunrise or date number, and text-layer
   lines are assigned to rows by vertical centre. Main text uses a larger font than the date column, so
   assigning by line top was off by one line; centring fixed it. The odd-page lagna-table rule is drawn
   as two half segments, which are merged before the full-width test.
2. **Date assignment by continuity**, validated against the printed day: **385 rows from 19 Mar 2026
   to 7 Apr 2027, no gaps, 383 confirmed against the printed date, 0 mismatches**. Separately, the
   weekday OCR'd from the row text matches the computed weekday on 380 of 385 rows, with 0 contradictions.
3. **Verification of every OCR'd value**, two independent ways:
   - *ghati check:* sunrise + ghatis × 24 min must equal the printed end time (±4 min);
   - *digits check:* the OCR'd numbers must appear verbatim among the exact text-layer numbers.

## 2. Book daily table → `panchanga_days` (source_kind `book_daily_table`)

Pipeline (`structured/book_daily.py`, `row_ocr.py`, `panchanga_parse.py`, `service.py`):
segment rows → crop each day → Tesseract (`tel`, psm 6, 300 dpi; **26 s for all 385 rows**) → trim
to `<weekday> … అహః NN-NN` → parse → retry weak rows (400 dpi / psm 4 / 250 dpi; 24 of 34 improved)
→ cycle pass.

Each day stores tithis, nakshatras, yogas and karanas (with end times converted to civil datetimes),
plus varjyam, durmuhurtam and amrutha-kalam windows, the notes text (festivals, planetary ingresses),
sunrise/sunset, page and bbox, and the raw OCR text.

**Time conversion** follows the book's own convention: `ఉ` morning, `ప` daytime, `మ` midday, `సా`
evening, `రా` night (12-xx and 1–5 → after midnight, next civil date), `తె` pre-dawn (next civil
date). `పూర్తి` = continues past the next sunrise.

**Cycle pass** (`resolve_sequences`): tithis and nakshatras follow strict cycles.
- An unreadable name is filled from the cycle (2 cases); its timing is still digit-verified.
- An OCR'd name that contradicts the cycle is flagged `break` and **not** verified.
- Masa and paksha are tracked through the year. Printed markers override inference, so
  **అధిక జ్యేష్ఠ (17 May–15 Jun 2026)** and **నిజ జ్యేష్ఠ** come out correctly.

### Results (all 385 days)

| | Days present | Verified (both checks + cycle) |
|---|---|---|
| Tithi at sunrise | 378 | **363 (94%)** |
| Nakshatra at sunrise | 373 | **341 (89%)** |
| Yoga | 357 | 316 |
| Karana | 378 | 366 |
| Timing windows | 1,188 | 1,178 digit-verified (99%) |

Spot check, 7 Nov 2026 vs. the page image: Trayodashi until 10:29 AM, Chitta until 6:12 AM (8 Nov),
varjyam 1:47–3:26 PM, durmuhurtam 6:13–7:44 AM, amrutham 11:38 PM–1:16 AM. **All match exactly.**

### Safety issues found and fixed during development
- A fuzzy fallback matched the karana `విస్త` (విష్టి) to the nakshatra `హస్త`, and the cycle then
  "corrected" it to another nakshatra. That would have **invented a nakshatra with a karana's
  timing**. Fixed: observed karana variants were added to the lexicon, out-of-order matches need a
  score ≥ 0.75, and tithi/nakshatra names appearing after the yoga/karanas (planet-ingress text) are
  never day elements. Regression tests guard all three.
- `regex`'s `\X` predates Unicode 15.1 (GB9c) and splits Telugu conjuncts. `graphemes()` re-joins them.

## 3. Festival table and utsavam schedules → `panchanga_events`

`structured/festivals.py` parses the Sarvam-OCR'd tables (already near-verbatim).
- **Festival rows** print `day | tamil day | weekday | events`; the month appears only on a month's first
  row, and is sometimes lost. Each row is placed on the **next date matching both the printed day
  and the printed weekday**. Rows that can't be placed are reported, never guessed. Wrapped second
  rows for the same date are kept on that date. **276 rows, 0 unresolved.**
- **Utsavam schedules** carry full dates; the schedule title is the line above each table header
  (running headers excluded). One **source typo** (`31-08-2026` printed between 30-07 and 01-08) was
  corrected to 31 Jul by weekday + continuity, and recorded. **309 rows.**
- **Cross-check between independently extracted tables:** Ugadi 19 Mar, Krishnashtami 4 Sep,
  Vaikuntha Ekadashi 20 Dec 2026 and Sankranti 15 Jan 2027 fall on the same dates in the festival table
  (Sarvam) and the daily table (geometry + Tesseract).

## 4. Wall calendars (free OCR) → `panchanga_days` (source_kind `wall_calendar`), UNVERIFIED

`structured/wall_calendar.py`: cells sit on a weekday × week grid, so each cell's **date comes from
calendar arithmetic**. Big date numbers (detected by Tesseract) only fit and validate the grid
(Theil–Sen fit, robust to misreads). Pages whose fit is implausible use that calendar's median grid
(11 of 24 pages). Each cell is OCR'd separately.

Measured against the book on all **385 overlapping days**:

| Free Tesseract per cell | Tithi | Nakshatra |
|---|---|---|
| Cells read | 290 (75%) | 253 (66%) |
| Name correct, of those read | 94.5% | 94.5% |
| Time exactly right, of those read | 88% | 89% |
| Agrees with the book (stored `cross_check`) | 248 agree / 35 disagree | 224 / 17 |

Low resolution (902 px wide for 2026, with a watermark) causes digit confusions such as 6 vs 8, and
calendars offer no ghati or text-layer check. So **calendar values are stored with `verified=false`**
unless the book agrees for that date. The book is authoritative wherever it exists.

### Coverage, 1 Jan 2026 – 31 Dec 2027 (730 days)

| Best source for the day's tithi | Days |
|---|---|
| Book, verified | 364 |
| Book, unverified or missing | 21 |
| Wall calendar only (unverified OCR) | 252 |
| Nothing readable | 93 |

## 5. Operations

- `psetu ingest` runs the structured stage automatically; `psetu extract-structured [--force]` runs it alone.
- Sources and page ranges live in `config/structured_sources.yaml`. Extraction is skipped when the document
  version and extractor version are unchanged (a no-op ingest takes about 1 s).
- Everything in this phase is free and local (PyMuPDF + Tesseract + cached page OCR).

## 6. Tests
94 tests pass (unit + integration), ruff and `mypy --strict` clean, 85% line coverage. New tests cover:
legacy digit recovery, every part-of-day time rule, real-row parsing with both verifications, tampered
digits being caught, OCR name variants (including the విస్త/హస్త trap), ghati repair, cycle
fill/break/adhika tracking, planet-ingress exclusion, festival placement (year-only cells, month
rollover, wrapped rows, unplaceable rows reported), utsavam typo correction, calendar grid positions and
robust grid fitting, and integration tests for segmenting the real book (385 days) and the extraction
service end to end, including idempotency.

## 7. Open decision: calendar-only dates
About 345 days (1 Jan–18 Mar 2026, 8 Apr–31 Dec 2027) exist **only** in the wall calendars, where free
OCR is unverified (and 93 days are unread). Options:
- **A, keep free (default):** answer those dates with an explicit "unverified OCR of the TTD wall
  calendar" caveat, or abstain on times. ₹0.
- **B, dual OCR (~₹12):** pack each month's cell crops into one page for Sarvam (24 pages × ₹0.50) and
  mark a value verified only where **Sarvam and Tesseract agree**. Unread cells get a second chance too.
