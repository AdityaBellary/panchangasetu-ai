"""Panchanga vocabulary (Telugu canonical forms + English/romanised aliases).

Used for page classification during ingestion and, later, for query routing, slot extraction
and cross-language query expansion. Spellings follow the TTD publications; common OCR/spelling
variants are listed so matching is tolerant.
"""

from __future__ import annotations

import regex

TITHIS: dict[str, list[str]] = {
    "పాడ్యమి": ["padyami", "prathama", "pratipada", "పాడ్యమి", "ప్రతిపద"],
    "విదియ": ["vidiya", "dwitiya", "dvitiya", "విదియ"],
    "తదియ": ["tadiya", "tritiya", "తదియ"],
    "చవితి": ["chavithi", "chaturthi", "చవితి", "చతుర్థి"],
    "పంచమి": ["panchami", "పంచమి"],
    "షష్ఠి": ["shashti", "sashti", "షష్ఠి", "షష్టి"],
    "సప్తమి": ["saptami", "సప్తమి"],
    "అష్టమి": ["ashtami", "అష్టమి"],
    "నవమి": ["navami", "నవమి"],
    "దశమి": ["dasami", "dashami", "దశమి"],
    "ఏకాదశి": ["ekadasi", "ekadashi", "ఏకాదశి"],
    "ద్వాదశి": ["dwadasi", "dvadashi", "dwadashi", "ద్వాదశి"],
    "త్రయోదశి": ["trayodasi", "trayodashi", "త్రయోదశి"],
    "చతుర్దశి": ["chaturdasi", "chaturdashi", "చతుర్దశి"],
    "పూర్ణిమ": ["purnima", "pournami", "poornima", "full moon", "పూర్ణిమ", "పౌర్ణమి"],
    "అమావాస్య": ["amavasya", "amavasya", "new moon", "అమావాస్య"],
}

NAKSHATRAS: dict[str, list[str]] = {
    "అశ్విని": ["ashwini", "aswini"],
    "భరణి": ["bharani"],
    "కృత్తిక": ["krittika", "krithika"],
    "రోహిణి": ["rohini"],
    "మృగశిర": ["mrigasira", "mrigashira"],
    "ఆర్ద్ర": ["ardra", "arudra", "ఆర్ర"],
    "పునర్వసు": ["punarvasu"],
    "పుష్యమి": ["pushyami", "pushya", "పుష్యమి"],
    "ఆశ్లేష": ["ashlesha", "aslesha", "ఆశ్రేష"],
    "మఖ": ["makha", "magha", "మఘ"],
    "పుబ్బ": ["pubba", "purva phalguni", "పూర్వఫల్గుణి"],
    "ఉత్తర": ["uttara", "uttara phalguni"],
    "హస్త": ["hasta"],
    "చిత్త": ["chitta", "chitra", "చిత్ర"],
    "స్వాతి": ["swati", "swathi"],
    "విశాఖ": ["visakha", "vishakha"],
    "అనూరాధ": ["anuradha", "అనురాధ"],
    "జ్యేష్ఠ": ["jyeshta", "jyeshtha", "జ్యేష్ట"],
    "మూల": ["moola", "mula"],
    "పూర్వాషాఢ": ["purvashadha", "poorvashada"],
    "ఉత్తరాషాఢ": ["uttarashadha", "uttarashada"],
    "శ్రవణం": ["shravana", "sravanam", "శ్రవణ"],
    "ధనిష్ఠ": ["dhanishta", "dhanishtha", "ధనిష్ట"],
    "శతభిషం": ["shatabhisha", "satabhisham", "శతభిష"],
    "పూర్వాభాద్ర": ["purvabhadra", "poorvabhadra"],
    "ఉత్తరాభాద్ర": ["uttarabhadra"],
    "రేవతి": ["revati", "revathi"],
}

MASAS: dict[str, list[str]] = {
    "చైత్రం": ["chaitra", "చైత్ర"],
    "వైశాఖం": ["vaisakha", "vaishakha", "వైశాఖ"],
    "జ్యేష్ఠం": ["jyeshta", "జ్యేష్ఠ", "జ్యేష్ట"],
    "ఆషాఢం": ["ashadha", "ashada", "ఆషాఢ"],
    "శ్రావణం": ["shravana", "sravana", "శ్రావణ"],
    "భాద్రపదం": ["bhadrapada", "భాద్రపద"],
    "ఆశ్వయుజం": ["ashwayuja", "aswayuja", "ఆశ్వయుజ"],
    "కార్తీకం": ["kartika", "karthika", "కార్తీక"],
    "మార్గశిరం": ["margashira", "margasira", "మార్గశిర"],
    "పుష్యం": ["pushya", "pushyam"],
    "మాఘం": ["magha", "మాఘ"],
    "ఫాల్గుణం": ["phalguna", "ఫాల్గుణ"],
}

VARAS: dict[str, list[str]] = {
    "ఆదివారం": ["sunday", "adivaram"],
    "సోమవారం": ["monday", "somavaram"],
    "మంగళవారం": ["tuesday", "mangalavaram"],
    "బుధవారం": ["wednesday", "budhavaram"],
    "గురువారం": ["thursday", "guruvaram"],
    "శుక్రవారం": ["friday", "sukravaram", "shukravaram"],
    "శనివారం": ["saturday", "shanivaram", "sanivaram"],
}

YOGAS: dict[str, list[str]] = {
    "విష్కంభం": ["vishkambha"],
    "ప్రీతి": ["priti", "preeti"],
    "ఆయుష్మాన్": ["ayushman"],
    "సౌభాగ్యం": ["saubhagya"],
    "శోభనం": ["shobhana", "sobhana"],
    "అతిగండం": ["atiganda"],
    "సుకర్మ": ["sukarma"],
    "ధృతి": ["dhriti"],
    "శూలం": ["shula", "soola"],
    "గండం": ["ganda"],
    "వృద్ధి": ["vriddhi", "వృద్ది"],
    "ధ్రువం": ["dhruva", "ధృవం"],
    "వ్యాఘాతం": ["vyaghata"],
    "హర్షణం": ["harshana"],
    "వజ్రం": ["vajra"],
    "సిద్ధి": ["siddhi", "సిద్ది"],
    "వ్యతీపాతం": ["vyatipata", "వ్యతీపాత్"],
    "వరీయాన్": ["variyan", "variyana"],
    "పరిఘం": ["parigha", "పరిఘ"],
    "శివం": ["shiva", "siva"],
    "సిద్ధం": ["siddha", "సిద్దం"],
    "సాధ్యం": ["sadhya"],
    "శుభం": ["shubha", "subha"],
    "శుక్లం": ["shukla yoga", "శుభ్రం"],
    "బ్రహ్మం": ["brahma", "బ్రాహ్మం"],
    "ఐంద్రం": ["indra", "aindra"],
    "వైధృతి": ["vaidhriti"],
}

KARANAS: dict[str, list[str]] = {
    "బవ": ["bava"],
    "బాలవ": ["balava", "బాలువ"],
    "కౌలవ": ["kaulava", "కౌలువ", "కలువ"],
    "తైతుల": ["taitila", "taitula"],
    "గరజి": ["garaja", "gara", "గర", "గరణజి", "గరణి", "గరజణి"],
    "వణిజ": ["vanija", "వణి", "వణజి"],
    # OCR variants observed in the TTD 2026-27 book (Tesseract tel): విస్త, విస్థి, విప్లి, భృద్ర
    "విష్టి": ["vishti", "bhadra", "భద్ర", "విస్త", "విస్థి", "విప్లి", "భృద్ర"],
    "శకుని": ["shakuni", "sakuni"],
    "చతుష్పాత్": ["chatushpada", "చతుష్పాద", "చతుప్పాత్"],
    "నాగవం": ["naga", "nagava"],
    "కింస్తుఘ్నం": ["kimstughna"],
}

# Planet ingress entries in the daily tables ("కుజః 37-30 రా.9-21 శతభిషం 3").
PLANETS: dict[str, list[str]] = {
    "అర్కః": ["sun", "surya", "రవి", "సూర్యః"],
    "చంద్రః": ["moon", "chandra"],
    "కుజః": ["mars", "kuja", "అంగారకః"],
    "బుధః": ["mercury", "budha"],
    "గురుః": ["jupiter", "guru", "బృహస్పతిః"],
    "శుక్రః": ["venus", "shukra"],
    "శనిః": ["saturn", "shani"],
    "రాహుః": ["rahu"],
    "కేతుః": ["ketu"],
}


RASIS: dict[str, list[str]] = {
    "మేష": ["mesha", "mesham", "aries"],
    "వృషభ": ["vrishabha", "vrushabha", "taurus"],
    "మిథున": ["mithuna", "midhuna", "gemini"],
    "కర్కాటక": ["karkataka", "kataka", "cancer", "కటక", "కర్నాటక"],
    "సింహ": ["simha", "simham", "leo"],
    "కన్యా": ["kanya", "virgo", "కన్య"],
    "తులా": ["tula", "thula", "libra", "తుల"],
    "వృశ్చిక": ["vrischika", "vrushchika", "scorpio"],
    "ధనుస్సు": ["dhanus", "dhanussu", "dhanu", "sagittarius", "ధనూ", "ధనుర్"],
    "మకర": ["makara", "makaram", "capricorn"],
    "కుంభ": ["kumbha", "kumbham", "aquarius"],
    "మీన": ["meena", "meenam", "pisces"],
}


def english_name(table: dict[str, list[str]], canonical: str) -> str:
    """First Latin alias, title-cased (used for English answers)."""
    for alias in table.get(canonical, []):
        if alias.isascii():
            return alias.title()
    return canonical


PANCHANGA_TERMS: dict[str, list[str]] = {
    "తిథి": ["tithi", "thithi", "thidhi"],
    "నక్షత్రం": ["nakshatram", "nakshatra", "star", "నక్షత్ర"],
    "మాసం": ["masam", "month"],
    "పక్షం": ["paksham", "paksha", "fortnight"],
    "శుద్ధ": ["shukla", "sukla", "suddha", "waxing"],
    "బహుళ": ["krishna", "bahula", "waning"],
    "రాహుకాలం": ["rahukalam", "rahu kalam", "rahukaal"],
    "వర్జ్యం": ["varjyam", "varjyam"],
    "దుర్ముహూర్తం": ["durmuhurtam", "durmuhurtham"],
    "యమగండం": ["yamagandam", "yamaganda"],
    "అమృతకాలం": ["amrutha kalam", "amritakalam"],
    "సూర్యోదయం": ["sunrise"],
    "సూర్యాస్తమయం": ["sunset"],
    "యోగం": ["yogam", "yoga"],
    "కరణం": ["karanam", "karana"],
    "పండుగ": ["festival", "panduga"],
}

# "ఉ.9-43", "సా.6-08", "తె.5-47", "రా.11-49", "ప.12-30", "మ.1-30": time-of-day prefix + h-mm.
TIME_RE = regex.compile(r"(?:ఉ|ప|మ|సా|రా|తె)\.\s?\d{1,2}[-.:]\d{2}")


def _all_telugu_forms() -> list[str]:
    forms: set[str] = set()
    for table in (TITHIS, NAKSHATRAS, MASAS, VARAS, YOGAS, KARANAS, PANCHANGA_TERMS):
        for canonical, aliases in table.items():
            forms.add(canonical)
            forms.update(a for a in aliases if regex.search(r"\p{Telugu}", a))
    return sorted(forms, key=len, reverse=True)


CALENDAR_TERM_RE = regex.compile("|".join(regex.escape(f) for f in _all_telugu_forms()))


GREGORIAN_MONTHS_TE = (
    "జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్", "జూలై", "జులై",
    "ఆగష్టు", "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్",
)  # fmt: skip

GREGORIAN_MONTH_NUMBER: dict[str, int] = {
    "జనవరి": 1, "ఫిబ్రవరి": 2, "మార్చి": 3, "ఏప్రిల్": 4, "మే": 5, "జూన్": 6, "జూలై": 7, "జులై": 7,
    "ఆగష్టు": 8, "ఆగస్టు": 8, "సెప్టెంబర్": 9, "అక్టోబర్": 10, "నవంబర్": 11, "డిసెంబర్": 12,
}  # fmt: skip
assert set(GREGORIAN_MONTH_NUMBER) == set(GREGORIAN_MONTHS_TE)

# "19-03-2026", "13-8-2026". A 4-digit year is required so number lists such as tithi sets
# "2-3-5-7" are not mistaken for dates.
DATE_RE = regex.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.](?:19|20)\d{2}\b")
# "ఏప్రిల్ 12", "అక్టోబర్ 14." Weaker evidence: prose also says "జూన్ 1 వరకు".
MONTH_DAY_RE = regex.compile(r"(?:" + "|".join(GREGORIAN_MONTHS_TE) + r")\s?\d{1,2}\b")


def calendar_signal(text: str) -> dict[str, float]:
    """Calendar evidence in a text: densities per 1000 characters plus an absolute count.

    Vocabulary alone does not make a page calendar data: pages *explaining* what a paksham or
    nakshatra pada is use the vocabulary heavily but answer general questions. Dated facts
    (time stamps, full dates; month-day mentions count a third) are what distinguish them.
    """
    n = max(len(text), 1)
    times = len(TIME_RE.findall(text))
    dates = len(DATE_RE.findall(text)) + len(MONTH_DAY_RE.findall(text)) / 3
    return {
        "terms_per_k": 1000 * len(CALENDAR_TERM_RE.findall(text)) / n,
        "times_per_k": 1000 * times / n,
        "dates_per_k": 1000 * dates / n,
        "dated_facts": times + dates,
    }
