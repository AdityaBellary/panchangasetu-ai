"""Prompts for grounded answering. The model sees numbered sources [S1..Sn] and must answer
only from them, cite them, and say so when they are insufficient."""

from __future__ import annotations

from datetime import datetime

from panchangasetu.core.interfaces import ChatMessage
from panchangasetu.retrieval.evidence import Evidence

# Refusals say what happened and what the app *can* answer, in the user's language.
ABSTAIN_EN = (
    "I couldn't find this in the TTD panchangam, so I'd rather not guess.\n\n"
    "I can tell you: the panchangam for any day (tithi, nakshatram, rahukalam), when the next Ekadashi "
    "or Pournami is, festival dates, muhurthams, and this year's rasi phalalu."
)
ABSTAIN_TE = (
    "ఈ విషయం టీటీడీ పంచాంగంలో దొరకలేదు, అందుకే ఊహించి చెప్పలేను.\n\n"
    "నేను చెప్పగలిగేవి: ఏ రోజుకైనా పంచాంగం (తిథి, నక్షత్రం, రాహుకాలం), తదుపరి ఏకాదశి లేదా పౌర్ణమి "
    "ఎప్పుడు, పండుగ తేదీలు, ముహూర్తాలు, ఈ సంవత్సరం రాశి ఫలాలు."
)
# A question about a specific day the book has no rule for ("is tomorrow good for a new business?"):
# the day's panchangam follows, so the user (or their priest) can judge.
DAY_INTRO_EN = (
    "The panchangam doesn't answer this directly, so I won't guess. "
    "Here is that day's panchangam to help you, or your priest, decide:"
)
DAY_INTRO_TE = (
    "ఈ ప్రశ్నకు పంచాంగంలో నేరుగా సమాధానం లేదు, అందుకే ఊహించి చెప్పలేను. మీరు లేదా మీ పురోహితులు నిర్ణయించుకోవడానికి ఆ రోజు పంచాంగం ఇదిగో:"
)
UNAVAILABLE_EN = "Sorry, I couldn't reach the answering service just now. Please ask again in a minute."
UNAVAILABLE_TE = "క్షమించండి, ఇప్పుడు సమాధాన సేవ అందుబాటులో లేదు. ఒక నిమిషం తరువాత మళ్ళీ అడగండి."
# The daily LLM budget is used up: calendar lookups (templates) still work.
BUDGET_EN = (
    "Today's limit for detailed answers has been reached. Calendar questions (tithi, nakshatram, "
    "rahukalam, the next Ekadashi) still work now; ask this one again tomorrow."
)
BUDGET_TE = (
    "ఈరోజు వివరణాత్మక సమాధానాల పరిమితి పూర్తయింది. తిథి, నక్షత్రం, రాహుకాలం, తదుపరి ఏకాదశి వంటి "
    "క్యాలెండర్ ప్రశ్నలకు ఇప్పుడూ సమాధానం లభిస్తుంది; ఈ ప్రశ్నను రేపు మళ్ళీ అడగండి."
)

SYSTEM = """You are PanchangaSetu, an assistant that answers questions about the TTD (Tirumala Tirupati \
Devasthanams) panchangam and calendar documents.

STRICT RULES:
1. Use ONLY the numbered sources provided. Never use outside knowledge, even if you know the answer.
2. Never invent or compute dates, tithis, nakshatrams, times or names that are not written in the sources.
   Copy times and dates exactly as they appear in the sources.
3. If the sources do not contain enough information to answer, set "sufficient" to false.
4. Cite every source you used by its id (e.g. "S2") in "citations".
5. If a source is marked UNVERIFIED, say in the answer that the value comes from an unverified reading
   of the wall calendar.
6. Write the answer in {language}. Keep Telugu names of tithis, nakshatrams and festivals; in English
   answers you may add the common English spelling in brackets.
7. Keep every condition, exception and qualifier exactly as the source states it. Do not simplify,
   generalise or merge cases. When the source states rules or conditions, translate those source
   sentences one by one (as a list) instead of summarising them; keep each pairing intact
   (e.g. "12th house in Vrishabha or Tula" must not become "in Vrishabha"). Every number you write
   (houses, counts, dates, times) must appear in a cited source.
8. Be concise: answer the question directly, in at most 6 sentences or a short list.

Today is {today} (India time).

9. In "quotes", copy 1-4 short passages (each under 25 words) VERBATIM from the sources, in the
    source's own language and spelling, that directly support your answer.

Respond with JSON only:
{{"sufficient": true|false, "answer": "...", "citations": ["S1", ...],
  "quotes": [{{"source": "S1", "text": "exact words copied from S1"}}]}}"""


def evidence_block(evidence: list[Evidence], max_chars_each: int = 2400) -> str:
    parts = []
    for i, e in enumerate(evidence, 1):
        status = {True: "verified", False: "UNVERIFIED", None: "document text"}[e.verified]
        text = e.text if len(e.text) <= max_chars_each else e.text[:max_chars_each] + " …"
        parts.append(f"[S{i}] ({e.citation()}; {status})\n{text}")
    return "\n\n".join(parts)


def build_messages(
    question: str,
    evidence: list[Evidence],
    *,
    language: str,
    now: datetime,
    notes: list[str] | None = None,
    term_hints: list[tuple[str, list[str]]] | None = None,
) -> list[ChatMessage]:
    lang = "Telugu (తెలుగు లిపిలో)" if language == "te" else "English"
    system = SYSTEM.format(language=lang, today=now.strftime("%A, %d %B %Y, %H:%M"))
    user = f"SOURCES:\n{evidence_block(evidence)}\n\n"
    if notes:
        user += "RETRIEVAL NOTES: " + "; ".join(notes) + "\n\n"
    if term_hints:
        # Cross-language anchors: the sources are Telugu, the question may not be.
        user += (
            "TELUGU TERMS FOR THIS QUESTION: "
            + "; ".join(f"{en} = {' / '.join(te)}" for en, te in term_hints)
            + "\n\n"
        )
    user += f"QUESTION: {question}"
    return [ChatMessage("system", system), ChatMessage("user", user)]


def retry_feedback(problems: list[str]) -> ChatMessage:
    return ChatMessage(
        "user",
        "Your previous answer had problems: "
        + "; ".join(problems)
        + ". Answer again using ONLY the sources, copying any times/dates exactly, or set sufficient=false.",
    )
