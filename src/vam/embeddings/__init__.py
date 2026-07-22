"""Optional foundation-model embedding adapters."""

from .esm2 import extract_esm2_embeddings
from .scgpt import extract_scgpt_embeddings

__all__ = ["extract_esm2_embeddings", "extract_scgpt_embeddings"]
