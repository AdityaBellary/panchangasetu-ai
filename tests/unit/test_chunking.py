import regex

from panchangasetu.core.types import Collection
from panchangasetu.ingestion.chunking import (
    PageText,
    TeluguChunker,
    approx_token_counter,
    find_boilerplate_lines,
    split_block,
)
from panchangasetu.telugu.text import graphemes

COUNT = approx_token_counter(3.0)
SENTENCE = "ఈ రాశివారికి గురుడు దశమ స్థానమందు సామాన్యఫలితములను ఇచ్చును."


def chunker(target: int = 60, maximum: int = 80, overlap: int = 0, minimum: int = 0) -> TeluguChunker:
    return TeluguChunker(
        target_tokens=target,
        max_tokens=maximum,
        overlap_units=overlap,
        count_tokens=COUNT,
        min_tokens=minimum,
    )


def page(n: int, text: str, coll: Collection = Collection.GENERAL, section: str = "prose") -> PageText:
    return PageText(n, text, coll, section)


def test_running_headers_are_detected_and_removed() -> None:
    pages = [page(i, f"## {30 + i} శ్రీ పరాభవనామ సంవత్సర పంచాంగం\n\n{SENTENCE} {i}") for i in range(1, 6)]
    assert "# శ్రీ పరాభవనామ సంవత్సర పంచాంగం" in find_boilerplate_lines(pages)
    chunks = chunker().chunk(pages)
    assert all("పరాభవనామ" not in c.text for c in chunks)
    assert all(c.heading_path == [] for c in chunks)


def test_headings_start_new_chunks_and_become_heading_path() -> None:
    text = f"## కన్యారాశి ఫలితములు\n\n{SENTENCE}\n\n## తులారాశి ఫలితములు\n\n{SENTENCE}"
    chunks = chunker(target=500, maximum=600).chunk([page(31, text)])
    assert [c.heading_path for c in chunks] == [["కన్యారాశి ఫలితములు"], ["తులారాశి ఫలితములు"]]


def test_chunks_never_mix_collections_and_track_page_spans() -> None:
    pages = [
        page(1, SENTENCE),
        page(2, SENTENCE),
        page(3, "నవమి ఉ.9-43 విశాఖ ఉ.7-14", Collection.PANCHANGA, "daily"),
    ]
    chunks = chunker(target=500, maximum=600).chunk(pages)
    assert [(c.collection, c.page_start, c.page_end) for c in chunks] == [
        (Collection.GENERAL, 1, 2),
        (Collection.PANCHANGA, 3, 3),
    ]


def test_skip_pages_are_not_chunked() -> None:
    chunks = chunker().chunk([page(1, "cover", Collection.SKIP, "cover"), page(2, SENTENCE)])
    assert [c.page_start for c in chunks] == [2]


def test_long_text_respects_max_tokens_and_keeps_graphemes_whole() -> None:
    text = " ".join([SENTENCE] * 30)
    chunks = chunker(target=40, maximum=50).chunk([page(1, text)])
    assert len(chunks) > 1
    assert all(c.token_count <= 50 for c in chunks)
    # Rejoined chunks reproduce the original grapheme sequence (no syllable was cut).
    original = [g for g in graphemes(text) if not g.isspace()]
    rebuilt = [g for c in chunks for g in graphemes(c.text) if not g.isspace()]
    assert rebuilt == original


def test_unbreakable_token_is_split_on_grapheme_boundaries() -> None:
    word = "శ్రీనివాస" * 40
    pieces = split_block(word, 10, COUNT)
    assert len(pieces) > 1
    assert "".join(pieces) == word
    assert all(not regex.match(r"\p{M}", p) for p in pieces)  # no piece starts with a combining mark


def test_table_blocks_split_on_rows() -> None:
    rows = "\n".join(f"6-2{i} | 6-20 | మార్చి {19 + i} | నవమి ఉ.9-43" for i in range(10))
    pieces = split_block(rows, 15, COUNT)
    assert all(p.count(" | ") == 3 for p in pieces)


def test_overlap_carries_last_unit_into_next_chunk() -> None:
    text = "\n\n".join(f"{SENTENCE} {i}" for i in range(6))
    chunks = chunker(target=40, maximum=60, overlap=1).chunk([page(1, text)])
    assert len(chunks) >= 2
    assert chunks[0].text.split("\n\n")[-1] == chunks[1].text.split("\n\n")[0]


def test_alternating_recto_verso_headers_are_stripped_but_table_data_survives() -> None:
    pages = []
    for n in range(1, 21):
        header = f"తిరుమల తిరుపతి దేవస్థానములు {n}" if n % 2 == 0 else f"{n} శ్రీ పరాభవనామ సంవత్సర సిద్ధాన్త పంచాంగం"
        # "6-20" and "భో.3-34 అహః 29-55" recur on every page but are data, not boilerplate.
        pages.append(page(n, f"{header}\n6-20\nభో.3-34 అహః 29-55\n{SENTENCE} {n}"))
    text = "\n".join(c.text for c in chunker(target=2000, maximum=3000).chunk(pages))
    assert "దేవస్థానములు" not in text and "సిద్ధాన్త పంచాంగం" not in text
    assert text.count("6-20") == 20 and text.count("అహః 29-55") == 20


def test_numeric_only_header_is_not_a_heading() -> None:
    chunks = chunker(target=500, maximum=600).chunk(
        [page(1, f"## జాతివర్గః\n\n{SENTENCE}\n\n## 156\n\n{SENTENCE}")]
    )
    assert len(chunks) == 1 and chunks[0].heading_path == ["జాతివర్గః"]


def test_small_chunks_merge_into_following_chunk_of_same_section() -> None:
    pages = [page(1, "శ్రీ పరాభవనామ సంవత్సర వ్రతదిన పట్టిక"), page(2, "## కార్తెలు\n\n" + SENTENCE * 3)]
    chunks = chunker(target=500, maximum=600, minimum=30).chunk(pages)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("శ్రీ పరాభవనామ") and (chunks[0].page_start, chunks[0].page_end) == (1, 2)
    assert chunks[0].heading_path == ["కార్తెలు"]


def test_small_chunks_never_merge_across_collections() -> None:
    pages = [page(1, "చిన్న వాక్యం"), page(2, SENTENCE * 3, Collection.PANCHANGA, "daily")]
    assert len(chunker(target=500, maximum=600, minimum=30).chunk(pages)) == 2


def test_overlap_never_pushes_a_chunk_past_max_tokens() -> None:
    units = [SENTENCE * 3 for _ in range(8)]  # each unit ~ 60 tokens
    chunks = chunker(target=70, maximum=80, overlap=1).chunk([page(1, "\n\n".join(units))])
    assert chunks and all(c.token_count <= 80 for c in chunks)
