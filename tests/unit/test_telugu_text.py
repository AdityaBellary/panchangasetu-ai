import unicodedata

from panchangasetu.telugu.text import (
    graphemes,
    normalize_telugu,
    script_stats,
    split_sentences,
    truncate_graphemes,
)


def test_normalize_converts_telugu_digits_to_ascii() -> None:
    assert normalize_telugu("౧౫-౦౧-౨౦౨౬") == "15-01-2026"


def test_normalize_can_keep_telugu_digits() -> None:
    assert normalize_telugu("౧౫", ascii_digits=False) == "౧౫"


def test_normalize_applies_nfc_and_strips_zero_width_but_keeps_zwnj() -> None:
    decomposed = unicodedata.normalize("NFD", "కై")  # vowel sign ై decomposes to ె + ౖ
    text = f"﻿{decomposed}​ జూన్‌"
    out = normalize_telugu(text)
    assert out == "కై జూన్‌"
    assert unicodedata.is_normalized("NFC", out)


def test_normalize_collapses_whitespace_and_blank_lines() -> None:
    assert normalize_telugu("  ఈ   రాశి \t వారికి \n\n\n\nగురుడు  ") == "ఈ రాశి వారికి\n\nగురుడు"


def test_graphemes_never_split_consonant_from_vowel_sign_or_virama() -> None:
    # "స్వామి" = స + ్ + వ + ా + మ + ి  -> 2 grapheme clusters: "స్వా", "మి"
    assert graphemes("స్వామి") == ["స్వా", "మి"]


def test_truncate_graphemes_keeps_whole_syllables() -> None:
    assert truncate_graphemes("శ్రీనివాస", 2) == "శ్రీని"


def test_script_stats_flags_legacy_font_private_use_text() -> None:
    legacy = " "
    st = script_stats(legacy)
    assert st.pua == 6 and st.telugu == 0
    assert st.valid_ratio == 0.0


def test_script_stats_counts_telugu_and_latin() -> None:
    st = script_stats("నక్షత్రం today 15")
    assert st.telugu > 0 and st.latin == 5 and st.digits == 2
    assert st.valid_ratio == 1.0
    assert 0 < st.telugu_ratio < 1


def test_split_sentences_handles_danda_and_full_stop() -> None:
    text = "శ్రీ వేంకటేశాయ మంగళం। శ్రీనివాసాయ మంగళం॥ ఇది వాక్యం. మరొకటి?"
    assert split_sentences(text) == ["శ్రీ వేంకటేశాయ మంగళం।", "శ్రీనివాసాయ మంగళం॥", "ఇది వాక్యం.", "మరొకటి?"]
