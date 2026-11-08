"""Download the embedding model into the image at build time, without loading it.

Loading a 2 GB ONNX model needs several GB of memory, more than an image build can count on; the files
are all the image needs. FastEmbed finds them in the same cache at runtime.
"""

from __future__ import annotations

import sys

from fastembed import TextEmbedding

from panchangasetu.config import get_settings

settings = get_settings()
for cls in TextEmbedding.EMBEDDINGS_REGISTRY:  # the class that serves this model knows its files
    try:
        description = cls._get_model_description(settings.embedding_model)
    except ValueError:
        continue
    path = cls.download_model(description, cache_dir=str(settings.model_cache_dir))
    print(f"embedding model files in {path}")
    break
else:
    sys.exit(f"FastEmbed does not know {settings.embedding_model}")
