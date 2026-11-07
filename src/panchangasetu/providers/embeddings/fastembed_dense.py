"""Local dense embedders via FastEmbed (ONNX Runtime; no PyTorch, no network at query time).

Models run on CPU inside the worker/API process. Weights are downloaded once into
`cache_dir` (keep it on a persistent volume in production).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from panchangasetu.observability.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DenseModelSpec:
    name: str
    dim: int
    query_prefix: str = ""
    document_prefix: str = ""
    # For models FastEmbed does not ship: ONNX export on the Hugging Face Hub.
    custom_hf_repo: str | None = None
    custom_model_file: str | None = None
    custom_additional_files: tuple[str, ...] = ()
    custom_pooling: str = "cls"  # cls | mean


KNOWN_MODELS: dict[str, DenseModelSpec] = {
    "intfloat/multilingual-e5-large": DenseModelSpec(
        "intfloat/multilingual-e5-large", 1024, query_prefix="query: ", document_prefix="passage: "
    ),
    "BAAI/bge-m3": DenseModelSpec(
        "BAAI/bge-m3",
        1024,
        custom_hf_repo="BAAI/bge-m3",
        custom_model_file="onnx/model.onnx",
        custom_additional_files=("onnx/model.onnx_data",),
        custom_pooling="cls",
    ),
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": DenseModelSpec(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 768
    ),
}


def _register_custom(spec: DenseModelSpec) -> None:
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    if any(m["model"] == spec.name for m in TextEmbedding.list_supported_models()):
        return
    TextEmbedding.add_custom_model(
        model=spec.name,
        pooling=PoolingType.CLS if spec.custom_pooling == "cls" else PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=spec.custom_hf_repo),
        dim=spec.dim,
        model_file=spec.custom_model_file or "model.onnx",
        additional_files=list(spec.custom_additional_files),
    )


class FastEmbedDense:
    def __init__(
        self, model: str, *, cache_dir: Path | None = None, batch_size: int = 16, threads: int | None = None
    ):
        from fastembed import TextEmbedding

        self.spec = KNOWN_MODELS.get(model) or DenseModelSpec(model, dim=0)
        if self.spec.custom_hf_repo:
            _register_custom(self.spec)
        self.name = model
        self._batch = batch_size
        self._model = TextEmbedding(
            model_name=model, cache_dir=str(cache_dir) if cache_dir else None, threads=threads
        )
        probe = next(iter(self._model.embed(["probe"])))
        self.dim = len(probe)
        log.info("embedder.loaded", model=model, dim=self.dim)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [self.spec.document_prefix + t for t in texts]
        return [_unit(v) for v in self._model.embed(prefixed, batch_size=self._batch)]

    def embed_query(self, text: str) -> list[float]:
        return _unit(next(iter(self._model.embed([self.spec.query_prefix + text]))))


def _unit(vector: Any) -> list[float]:
    """L2-normalise: some FastEmbed models return unnormalised vectors, which silently turns
    cosine search into dot-product search (observed with multilingual-e5-large)."""
    norm = float(np.linalg.norm(vector))
    unit: list[float] = (vector / norm).tolist() if norm else vector.tolist()
    return unit
