"""Recover exact ASCII digits/punctuation from legacy Telugu DTP fonts.

The TTD panchangam is set in legacy fonts (Gowthami, Priyaanka, TeluguNumbers, ...) that place
glyphs in the Private Use Area. Telugu letters are scrambled beyond use, but digits and common
punctuation sit at their ASCII code point + 0xF000 (U+F036 U+F02D U+F032 U+F032 -> "6-22").
That makes every printed number, time and date in the PDF exactly recoverable without OCR,
which we use to segment table rows and to validate OCR'd times.
"""

from __future__ import annotations

_RECOVERABLE = frozenset("0123456789-.:,/() ")
_PUA_ASCII_OFFSET = 0xF000


def decode_legacy_ascii(text: str, placeholder: str = "·") -> str:
    """Map PUA-shifted ASCII digits/punctuation back to ASCII; other PUA glyphs -> placeholder.
    Regular Unicode characters pass through unchanged."""
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if 0xF020 <= cp <= 0xF07E and chr(cp - _PUA_ASCII_OFFSET) in _RECOVERABLE:
            out.append(chr(cp - _PUA_ASCII_OFFSET))
        elif 0xE000 <= cp <= 0xF8FF:
            out.append(placeholder)
        else:
            out.append(ch)
    return "".join(out)


def digits_only(text: str) -> str:
    return "".join(ch for ch in decode_legacy_ascii(text, placeholder="") if ch.isdigit())
