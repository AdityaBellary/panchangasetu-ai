"""Row segmentation against the real TTD 2026-27 book (skipped when the PDF is absent)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pymupdf
import pytest

from panchangasetu.structured.book_daily import assign_dates, iter_daily_rows

pytestmark = pytest.mark.integration
PDF = Path(__file__).resolve().parents[2] / "data/raw/TTD_panchangam.pdf"


@pytest.mark.skipif(not PDF.exists(), reason="sample PDF not present")
def test_every_day_of_the_year_is_segmented_and_date_verified() -> None:
    with pymupdf.open(PDF) as doc:
        rows = assign_dates(list(iter_daily_rows(doc, list(range(100, 152)))), date(2026, 3, 19))
    assert len(rows) == 385
    assert rows[0].date == date(2026, 3, 19) and rows[-1].date == date(2027, 4, 7)
    assert sum(r.date_check == "ok" for r in rows) >= 380
    assert not [r for r in rows if r.date_check == "mismatch"]
    assert all(r.sunrise and r.sunset for r in rows)
    # Ugadi row: exact numbers from the legacy text layer.
    assert rows[0].sunrise == "6-22" and "7-06" in rows[0].numbers_in_row
