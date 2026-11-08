"""Answer generation: verification, templates, abstention and provider fallback (fake LLMs)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime

import httpx
import pytest

from panchangasetu.core.interfaces import ChatMessage, LLMError, LLMResult
from panchangasetu.generation.answerer import AnswerService
from panchangasetu.generation.prompts import ABSTAIN_EN, ABSTAIN_TE
from panchangasetu.generation.templates import (
    can_template_occurrences,
    render_day_answer,
    render_occurrence_answer,
)
from panchangasetu.generation.verify import check_answer, check_quotes, quote_support, strip_citation_markers
from panchangasetu.providers.llm.openai_compatible import FallbackLLM, OpenAICompatibleChat
from panchangasetu.query.dates import DateSpec
from panchangasetu.query.language import Language
from panchangasetu.query.router import Intent, PanchangaMode, Route
from panchangasetu.query.slots import Slots
from panchangasetu.retrieval.evidence import Evidence
from panchangasetu.retrieval.orchestrator import RetrievalResult

NOW = datetime(2026, 11, 7, 9, 0)
SRC = "[Panchanga for 2026-11-07] Tithi: త్రయోదశి; ends 2026-11-07 10:29 (printed ఉ.10-29) [verified]"


# ------------------------------------------------------------------ mechanical verification
def test_times_in_any_notation_must_be_in_the_source() -> None:
    assert check_answer("Trayodashi until 10:29 AM.", [SRC], default_year=2026) == []
    assert check_answer("త్రయోదశి ఉదయం 10:29 వరకు.", [SRC], default_year=2026) == []
    assert check_answer("Trayodashi until 11:29 AM.", [SRC], default_year=2026)  # invented time


def test_dates_including_telugu_month_first_form_are_checked() -> None:
    src = "గురుడు 2026 జూన్ 1 వరకు దశమ స్థానమందు, డిసెంబర్ 5వ తేదీ వరకు"
    assert check_answer("Until 1 June and until December 5.", [src], default_year=2026) == []
    assert check_answer("Until 3 June.", [src], default_year=2026)


def test_numbers_must_appear_as_digits_or_telugu_ordinals() -> None:
    src = "దశమ స్థానమందు ... ఏకాదశ స్థానమందు; వధూవరులకు 2-4-7-8-12 స్థానములు"
    assert (
        check_answer("Jupiter in the 10th, then the 11th; houses 2, 4, 7, 8, 12.", [src], default_year=2026)
        == []
    )
    problems = check_answer("Mars in the 9th house.", [src], default_year=2026)
    assert problems == ["number 9 is not in the cited sources"]


def test_citation_markers_are_not_numbers_and_are_removed() -> None:
    assert strip_citation_markers("Ekadashi is on 20 Nov (S2) [S3].") == "Ekadashi is on 20 Nov."
    assert check_answer("See (S2) and S3.", ["no numbers here"], default_year=2026) == []


def test_quotes_must_exist_in_some_source_allowing_ocr_variants() -> None:
    sources = {"S1": "ముక్కుమీద కార్యహాని\nముక్కుకొనయందు పరాజయము", "S2": "కొత్త ఉద్యోగం పొందవచ్చు."}
    problems, supported = check_quotes([("S1", "కొత్త ఉద్యోగం పొందవచ్చు.")], sources)  # wrong id, real quote
    assert problems == [] and supported == {"S2": ["కొత్త ఉద్యోగం పొందవచ్చు."]}
    assert quote_support("ముక్కుమీద కార్యహాని", "ముక్కు మీద కార్యహాని")  # spacing variant
    problems, _ = check_quotes([("S1", "తలమీద రాజయోగము కలుగును")], sources)  # invented
    assert problems and "not in any source" in problems[0]
    assert check_quotes([], sources)[0] == ["no supporting quotes"]


# ------------------------------------------------------------------ templates
def day_ev() -> Evidence:
    elements = [
        {
            "kind": "tithi",
            "name": "త్రయోదశి",
            "ends_at": "2026-11-07T10:29",
            "masa": "ఆశ్వయుజం",
            "paksha": "బహుళ",
            "adhika": False,
            "verified": True,
        },
        {"kind": "nakshatra", "name": "చిత్త", "ends_at": "2026-11-08T06:12", "verified": True},
    ]
    return Evidence(
        "d1",
        "day",
        "panchanga",
        SRC,
        "book.pdf",
        [131],
        date="2026-11-07",
        verified=True,
        source_kind="book_daily_table",
        data={
            "elements": elements,
            "windows": [],
            "sunrise": "6-13",
            "sunset": "5-41",
            "following": [{"kind": "tithi", "name": "చతుర్దశి"}, {"kind": "nakshatra", "name": "స్వాతి"}],
        },
    )


def test_day_template_in_both_languages() -> None:
    en = render_day_answer(day_ev(), ["tithi", "nakshatra"], "en")
    assert "Trayodasi until 10:29 AM; then Chaturdasi" in en and "Chitta until 6:12 AM on 08 Nov" in en
    te = render_day_answer(day_ev(), ["tithi", "masa"], "te")
    assert "త్రయోదశి ఉదయం 10:29 వరకు, తరువాత చతుర్దశి" in te and "ఆశ్వయుజం" in te


def test_occurrence_template_lists_next_two() -> None:
    occ = []
    for d, ends in (("2026-11-20", "2026-11-21T02:04"), ("2026-12-04", "2026-12-05T01:36")):
        occ.append(
            Evidence(
                d,
                "occurrence",
                "panchanga",
                "",
                "book.pdf",
                [132],
                date=d,
                verified=True,
                data={
                    "match": {
                        "kind": "tithi",
                        "name": "ఏకాదశి",
                        "ends_at": ends,
                        "masa": "కార్తీకం",
                        "paksha": "శుద్ధ",
                    }
                },
            )
        )
    out = render_occurrence_answer(occ, "en")
    assert out.splitlines()[0].startswith("Next Ekadasi: Friday, 20 November 2026")
    assert "until 2:04 AM on 21 Nov" in out and "After that" in out
    assert "not verified" not in out
    # A day only the wall calendar could read is still listed, and says so.
    occ[0].verified = False
    assert can_template_occurrences(occ)
    first, second = render_occurrence_answer(occ, "en").splitlines()
    assert "not verified against the book" in first and "not verified" not in second
    assert "ధృవీకరించబడలేదు" in render_occurrence_answer(occ, "te").splitlines()[0]


# ------------------------------------------------------------------ answer service
class FakeRetriever:
    def __init__(
        self,
        evidence: list[Evidence],
        mode: PanchangaMode,
        intent: Intent,
        attrs: list[str],
        lang: str = "en",
        notes: list[str] | None = None,
    ) -> None:
        self.evidence, self.mode, self.intent, self.attrs, self.lang = evidence, mode, intent, attrs, lang
        self.notes = notes or []
        self.clock = lambda: NOW
        from panchangasetu.query.rewrite import Glossary

        self.glossary = Glossary({})

    def retrieve(
        self, question: str, *, now: datetime | None = None, language: Language | None = None
    ) -> RetrievalResult:
        self.last_question = question
        route = Route(
            self.intent,
            self.mode,
            language or Language(self.lang, "Latn", False, 0.0),
            [DateSpec(NOW.date(), NOW.date(), "today", "day")],
            Slots(attributes=self.attrs),
            0.9,
        )
        return RetrievalResult(
            question, route, question, list(self.evidence), list(self.notes), {"total": 1.0}
        )


class FakeLLM:
    name = "fake"

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = replies
        self.calls = 0

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 600,
        temperature: float = 0.1,
        json: bool = False,
    ) -> LLMResult:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(reply, "fake-model", 1000, 100, 5.0)


PROSE = Evidence("c1", "chunk", "general", "కుజుడు బృహస్పతితో చేరినపుడు కుజదోషము లేదు.", "book.pdf", [74])


def reply(answer: str, quote: str = "కుజుడు బృహస్పతితో చేరినపుడు కుజదోషము లేదు.", sufficient: bool = True) -> str:
    return json.dumps(
        {
            "sufficient": sufficient,
            "answer": answer,
            "citations": ["S1"],
            "quotes": [{"source": "S1", "text": quote}],
        },
        ensure_ascii=False,
    )


def service(
    evidence: list[Evidence],
    llm: FakeLLM | None,
    mode: PanchangaMode = PanchangaMode.NONE,
    intent: Intent = Intent.GENERAL,
    attrs: list[str] | None = None,
    **kw: object,
) -> AnswerService:
    return AnswerService(FakeRetriever(evidence, mode, intent, attrs or [], **kw), llm)  # type: ignore[arg-type]


def test_no_evidence_abstains_without_calling_the_llm() -> None:
    llm = FakeLLM([reply("x")])
    a = service([], llm, notes=["no panchanga data for 2030-08-15..2030-08-15"]).ask("tithi on 15 Aug 2030?")
    assert a.abstained and a.answer == ABSTAIN_EN and llm.calls == 0 and "2030" in a.reason


def test_verified_day_lookup_uses_the_template_not_the_llm() -> None:
    llm = FakeLLM([reply("x")])
    a = service([day_ev()], llm, PanchangaMode.DAY, Intent.PANCHANGA, ["tithi"]).ask("today's tithi?")
    assert a.method == "template" and llm.calls == 0 and a.cost_inr == 0 and "10:29" in a.answer


def test_grounded_llm_answer_with_verified_quote() -> None:
    a = service([PROSE], FakeLLM([reply("There is no Kuja dosha when Mars is with Jupiter (S1).")])).ask(
        "Kuja?"
    )
    assert a.method == "llm" and not a.abstained and "(S1)" not in a.answer
    assert a.citations[0].pages == [74] and a.citations[0].quotes and a.cost_inr > 0


def test_model_saying_insufficient_means_abstain_in_question_language() -> None:
    a = service([PROSE], FakeLLM([reply("", sufficient=False)]), lang="te").ask("బల్లి తల మీద?")
    assert a.abstained and a.answer == ABSTAIN_TE


def test_invented_number_triggers_retry_then_abstain() -> None:
    llm = FakeLLM([reply("No dosha in the 11th house.")])
    a = service([PROSE], llm).ask("Kuja?")
    assert a.abstained and llm.calls == 2 and "number 11" in a.reason


def test_retry_can_recover() -> None:
    llm = FakeLLM([reply("No dosha in the 11th house."), reply("No dosha when Mars is with Jupiter.")])
    a = service([PROSE], llm).ask("Kuja?")
    assert not a.abstained and llm.calls == 2


def test_invented_quote_is_rejected() -> None:
    a = service([PROSE], FakeLLM([reply("Mars brings a royal yoga.", quote="కుజుడు రాజయోగమును ఇచ్చును")])).ask(
        "?"
    )
    assert a.abstained and "quote" in a.reason


def test_provider_outage_abstains_with_reason() -> None:
    a = service([PROSE], FakeLLM([LLMError("down", retryable=True)])).ask("Kuja?")
    assert a.abstained and "unavailable" in a.reason


# ------------------------------------------------------------------ providers
def _chat(status: int, body: dict) -> OpenAICompatibleChat:  # type: ignore[type-arg]
    http = httpx.Client(
        base_url="https://x.test", transport=httpx.MockTransport(lambda r: httpx.Response(status, json=body))
    )
    return OpenAICompatibleChat(
        name="t", model="m", base_url="https://x.test", headers={}, http=http, max_retries=0
    )


def test_out_of_credits_is_classified_as_quota() -> None:
    with pytest.raises(LLMError) as exc:
        _chat(402, {"error": {"message": "no credits"}}).complete([ChatMessage("user", "hi")])
    assert exc.value.quota and not exc.value.retryable
    with pytest.raises(LLMError) as exc2:
        _chat(429, {"error": {"code": "credit_balance_exhausted"}}).complete([ChatMessage("user", "hi")])
    assert exc2.value.quota


def test_reasoning_that_exhausts_max_tokens_is_an_error_not_an_empty_answer() -> None:
    body = {"choices": [{"finish_reason": "length", "message": {"content": ""}}], "usage": {}}
    with pytest.raises(LLMError, match="empty completion"):
        _chat(200, body).complete([ChatMessage("user", "hi")])


def test_fallback_moves_to_next_provider_on_quota() -> None:
    ok = {
        "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
        "model": "m2",
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }
    llm = FallbackLLM([_chat(402, {}), _chat(200, ok)])
    assert llm.complete([ChatMessage("user", "hi")]).model == "m2"


def test_fallback_does_not_hide_our_own_bad_requests() -> None:
    llm = FallbackLLM([_chat(400, {"error": {"message": "bad field"}}), _chat(200, {})])
    with pytest.raises(LLMError, match="rejected"):
        llm.complete([ChatMessage("user", "hi")])


def test_evidence_selection_keeps_exact_facts_and_mixes_collections() -> None:
    from panchangasetu.generation.answerer import select_evidence

    lists = [Evidence(f"p{i}", "chunk", "panchanga", "list", "book.pdf", [47 + i]) for i in range(5)]
    rule = Evidence("g1", "chunk", "general", "rule", "book.pdf", [92])
    day = Evidence("d1", "day", "panchanga", "day", "book.pdf", [132], date="2026-11-07", verified=True)
    picked = select_evidence([day, *lists, rule], 5)
    assert picked[0] is day and rule in picked and len(picked) == 5


def test_citation_lead_ins_are_removed_with_their_markers() -> None:
    assert strip_citation_markers("According to [S1], the months are Magha.") == "The months are Magha."
    assert strip_citation_markers("Magha is good (S2).") == "Magha is good."


def test_dates_are_not_read_as_times_and_number_words_count() -> None:
    src = "తేది. 15-01-2027 శుక్రవారం మకర సంక్రాంతి. బ్రహ్మణః పరమాయుః వర్ష శతమేవ"
    assert check_answer("Makara Sankranti is on 2027-01-15 (15-01-2027).", [src], default_year=2027) == []
    assert check_answer("Brahma's lifespan is 100 years.", [src], default_year=2027) == []
    assert check_answer("Brahma's lifespan is 120 years.", [src], default_year=2027) != []
