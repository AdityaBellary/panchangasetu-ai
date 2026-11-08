"""Application settings.

Every tunable lives here and is read from the environment (or `.env`). Provider credentials
(`SARVAM_API_KEY`) are kept separate from anything consumer-facing and are never logged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

OCREngineName = Literal["sarvam", "tesseract"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PSETU_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["dev", "test", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    # --- storage -------------------------------------------------------------------------
    database_url: str = "postgresql+psycopg://psetu:psetu@localhost:55432/psetu"
    data_dir: Path = Path("data")
    input_dir: Path = Path("data/raw")
    object_store_dir: Path = Path("data/objects")
    collections_rules_path: Path = Path("config/collections.yaml")
    structured_sources_path: Path = Path("config/structured_sources.yaml")
    # Postgres advisory-lock key guarding ingestion; distinct per deployment sharing a database.
    ingestion_lock_key: int = 0x50534554

    # --- page preparation ----------------------------------------------------------------
    render_dpi: int = Field(300, ge=72, le=600)
    # Image-only pages are extracted at native resolution and upscaled so the short side
    # reaches at least this many pixels (low-res scans OCR better with some upscaling).
    min_image_short_side_px: int = Field(1800, ge=0)
    # Fraction of text-layer letters that must be real Unicode Telugu/Latin for the text
    # layer to be trusted instead of OCR (legacy-font PDFs fail this check).
    text_layer_min_valid_ratio: float = Field(0.9, ge=0, le=1)
    text_layer_min_chars: int = 40

    # --- OCR -----------------------------------------------------------------------------
    ocr_engine: OCREngineName = "sarvam"
    ocr_fallback_engine: OCREngineName | None = "tesseract"
    ocr_batch_pages: int = Field(10, ge=1, le=20)

    sarvam_api_key: SecretStr | None = Field(default=None, validation_alias="SARVAM_API_KEY")
    # Google AI Studio key (Gemini); GOOGLE_API_KEY is accepted too.
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    sarvam_doc_language: str = "te-IN"
    sarvam_doc_content_type: Literal["printed", "handwritten", "mixed"] = "printed"
    sarvam_poll_interval_s: float = Field(3.0, gt=0)
    sarvam_job_timeout_s: float = Field(900.0, gt=0)
    sarvam_request_timeout_s: float = Field(120.0, gt=0)
    sarvam_max_retries: int = Field(4, ge=0)

    tesseract_cmd: str = "tesseract"
    tesseract_langs: str = "tel+eng"
    tesseract_data_dir: Path | None = Path("data/tessdata")
    tesseract_timeout_s: float = 300.0
    # Day-row OCR for structured extraction (Telugu only: the rows contain no English words).
    structured_row_langs: str = "tel"
    structured_ocr_workers: int = Field(6, ge=1, le=32)

    # --- vector index & retrieval ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    index_prefix: str = "psetu"
    # Chosen by the Phase 3 bake-off: best cross-lingual recall.
    embedding_model: str = "intfloat/multilingual-e5-large"
    model_cache_dir: Path = Path("data/models")
    index_section_gloss: bool = True
    retrieval_top_k: int = Field(8, ge=1, le=50)
    retrieval_candidates: int = Field(30, ge=5, le=200)
    # Lexical (Telugu sparse) fusion only for queries that are mostly Telugu script; the
    # bake-off showed it degrades English queries.
    hybrid_min_telugu_ratio: float = Field(0.5, ge=0, le=1)
    keep_previous_index_builds: int = Field(1, ge=0)
    timezone: str = "Asia/Kolkata"  # "today" is evaluated here, not in the server's timezone
    router_examples_path: Path = Path("config/router_examples.yaml")
    router_min_confidence: float = Field(0.65, ge=0, le=1)
    query_glossary_path: Path = Path("config/query_glossary.yaml")
    weekday_periods_path: Path = Path("config/weekday_periods.yaml")  # rahukalam/yamagandam/gulika

    # --- answer generation ---
    # Providers tried in order; out-of-credits or outages fall through to the next one.
    llm_providers: list[Literal["sarvam", "gemini"]] = ["sarvam", "gemini"]
    sarvam_chat_model: str = "sarvam-105b-conversations"  # no reasoning: fast, cheap (Phase 5 probe)
    # gemini-3.8-flash is the newest Flash but was overloaded (503 "high demand") on 2026-11-08; 3.6 is the
    # same price and answered in ~2 s. Switch back with PSETU_GEMINI_CHAT_MODEL once 3.8 is stable.
    gemini_chat_model: str = "gemini-3.6-flash"
    gemini_reasoning_effort: str = "low"  # Gemini 3 always thinks; this keeps it (and its cost) small
    # Tried in order when the main Gemini model is overloaded or failing.
    gemini_fallback_models: list[str] = ["gemini-3.5-flash-lite"]
    llm_pricing_path: Path = Path("config/llm_pricing.yaml")
    # Tried once when the primary cannot produce a verified answer although sources were found (hard
    # list lookups, complex rules). Gemini answered 16/19 held-out questions to Sarvam's 13 at ~4.5x the
    # cost (2026-11-08 benchmark), so only those questions pay for it. "" disables escalation.
    escalation_provider: Literal["", "gemini", "sarvam"] = "gemini"
    # Used when escalation_provider is "sarvam": reasons, ~20 s and ~₹0.3 per answer, and on large
    # prompts needs >4k reasoning tokens (Phase 5 probe).
    escalation_model: str = "sarvam-105b"
    llm_timeout_s: float = Field(60.0, gt=0)

    # --- conversations ---
    conversation_history_turns: int = Field(3, ge=0, le=20)  # earlier questions used to resolve a follow-up
    conversation_idle_minutes: float = Field(60.0, gt=0)  # older turns are no longer context
    # Follow-ups the rules cannot place ("what are its remedies?") are rewritten by the chat LLM
    # (~₹0.01 each). False: such questions are answered as typed.
    followup_llm_rewrite: bool = True

    # --- HTTP API ---
    api_host: str = "127.0.0.1"
    api_port: int = Field(8000, ge=1, le=65535)
    api_rate_limit_per_minute: int = Field(30, ge=1)  # per API key; a key can carry its own limit
    api_max_question_chars: int = Field(500, ge=10)
    api_metrics_token: SecretStr | None = None  # required by /metrics when set

    # --- operations ---
    # LLM spend allowed per local day (INR, all users together); 0 = unlimited. Calendar lookups are
    # answered by template for free and keep working when it is used up.
    llm_daily_budget_inr: float = Field(25.0, ge=0)
    # Conversations with no activity for this many days are deleted by `psetu sessions prune` (also run
    # when the server starts). 0 keeps them forever.
    session_retention_days: int = Field(180, ge=0)

    # --- chunking ------------------------------------------------------------------------
    chunk_min_tokens: int = Field(30, ge=0)
    chunk_target_tokens: int = Field(350, ge=50)
    chunk_max_tokens: int = Field(512, ge=64)
    chunk_overlap_sentences: int = Field(1, ge=0)
    # Approximation until the embedder's tokenizer is pinned in Phase 3.
    approx_chars_per_token: float = Field(3.0, gt=0)

    # Bump these when the corresponding logic changes; the pipeline uses them to decide
    # what to recompute (OCR is the expensive step and is only redone on extraction bumps).
    extraction_version: str = "ext-1"
    chunker_version: str = "chunk-7"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
