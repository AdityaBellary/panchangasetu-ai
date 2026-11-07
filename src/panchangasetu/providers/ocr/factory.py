"""Build OCR engines from settings."""

from __future__ import annotations

from panchangasetu.config import OCREngineName, Settings
from panchangasetu.core.interfaces import OCREngine
from panchangasetu.providers.ocr.sarvam import SarvamOCREngine
from panchangasetu.providers.ocr.tesseract import TesseractOCREngine


class ProviderConfigError(RuntimeError):
    pass


def build_ocr_engine(name: OCREngineName, settings: Settings) -> OCREngine:
    if name == "sarvam":
        if settings.sarvam_api_key is None or not settings.sarvam_api_key.get_secret_value():
            raise ProviderConfigError("SARVAM_API_KEY is not set (add it to .env)")
        return SarvamOCREngine.from_api_key(
            settings.sarvam_api_key.get_secret_value(),
            request_timeout_s=settings.sarvam_request_timeout_s,
            language=settings.sarvam_doc_language,
            content_type=settings.sarvam_doc_content_type,
            poll_interval_s=settings.sarvam_poll_interval_s,
            job_timeout_s=settings.sarvam_job_timeout_s,
            max_retries=settings.sarvam_max_retries,
        )
    if name == "tesseract":
        return TesseractOCREngine(
            cmd=settings.tesseract_cmd,
            langs=settings.tesseract_langs,
            tessdata_dir=settings.tesseract_data_dir,
            timeout_s=settings.tesseract_timeout_s,
        )
    raise ProviderConfigError(f"unknown OCR engine: {name}")
