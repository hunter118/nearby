from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SimilarityConfig:
    positive_similarity_only: bool = True
    similarity_floor: float = 0.0


@dataclass
class EmbeddingConfig:
    backend: str = "hashing"  # hashing | sentence_transformers
    hashing_n_features: int = 2**12
    st_model_name: str = "BAAI/bge-large-en-v1.5"
    st_device: str = "auto"  # auto | cpu | mps | cuda
    st_batch_size: int = 64
    st_normalize_embeddings: bool = True
    st_trust_remote_code: bool = False
    st_cache_chunk_size: int = 4096


class MarketEmbedder:
    """
    Deterministic text embedding via hashing vectorizer.
    Suitable for backtests where reproducibility matters.
    """

    def __init__(self, n_features: int = 2**12) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.vectorizer.n_features), dtype=float)
        return self.vectorizer.transform(texts).toarray()

    def pairwise_similarity(
        self,
        query_vec: np.ndarray,
        history_vecs: np.ndarray,
        config: SimilarityConfig,
    ) -> np.ndarray:
        if history_vecs.size == 0:
            return np.array([], dtype=float)
        sim = cosine_similarity(query_vec.reshape(1, -1), history_vecs).reshape(-1)
        if config.positive_similarity_only:
            sim = np.maximum(sim, config.similarity_floor)
        return sim


class SentenceTransformerEmbedder:
    """
    Local LLM-style semantic embedding via sentence-transformers.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for LLM embedding backend. "
                "Install with: pip install sentence-transformers"
            ) from exc
        resolved_device = None if device == "auto" else device
        self.model = SentenceTransformer(
            model_name,
            device=resolved_device,
            trust_remote_code=trust_remote_code,
        )
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=float)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def pairwise_similarity(
        self,
        query_vec: np.ndarray,
        history_vecs: np.ndarray,
        config: SimilarityConfig,
    ) -> np.ndarray:
        if history_vecs.size == 0:
            return np.array([], dtype=float)
        sim = cosine_similarity(query_vec.reshape(1, -1), history_vecs).reshape(-1)
        if config.positive_similarity_only:
            sim = np.maximum(sim, config.similarity_floor)
        return sim


def build_embedder(config: EmbeddingConfig):
    backend = config.backend.lower()
    if backend == "hashing":
        return MarketEmbedder(n_features=config.hashing_n_features)
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(
            model_name=config.st_model_name,
            device=config.st_device,
            batch_size=config.st_batch_size,
            normalize_embeddings=config.st_normalize_embeddings,
            trust_remote_code=config.st_trust_remote_code,
            show_progress_bar=False,
        )
    raise ValueError(f"Unsupported embedding backend: {config.backend}")
