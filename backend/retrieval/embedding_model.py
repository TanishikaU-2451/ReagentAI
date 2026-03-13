"""Embedding Model -- Sentence-Transformer Wrapper.

Provides a thin, production-grade wrapper around the
``sentence-transformers/all-mpnet-base-v2`` model (or any model specified in
``settings.embedding_model``) for encoding text into dense vector
representations.

Key design decisions:
    * **Lazy loading** -- the heavyweight model is downloaded and initialised
      only on the first call to :meth:`encode`, keeping import-time fast.
    * **Automatic batching** -- callers can pass arbitrarily long text lists;
      the encoder splits them into configurable batches so GPU/CPU memory
      stays bounded.
    * **Normalisation** -- embeddings are L2-normalised by default so that
      cosine similarity reduces to a simple dot product, which FAISS
      ``IndexFlatIP`` exploits for speed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from backend.config.settings import settings
from backend.utils.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger("retrieval.embedding_model")

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------
_DEFAULT_BATCH_SIZE: int = 64
_DEFAULT_NORMALIZE: bool = True


class EmbeddingModel:
    """Encode text into dense vectors using a SentenceTransformer model.

    The model is loaded lazily on the first call to :meth:`encode` so that
    importing this module has negligible overhead.

    Parameters
    ----------
    model_name : str | None
        HuggingFace model identifier.  Falls back to
        ``settings.embedding_model`` (default:
        ``sentence-transformers/all-mpnet-base-v2``).
    batch_size : int
        Maximum number of texts per forward pass.  Larger values trade
        memory for throughput.
    normalize : bool
        If *True* (default), embeddings are L2-normalised so that inner
        product equals cosine similarity.
    device : str | None
        PyTorch device string (``"cpu"``, ``"cuda"``, etc.).  When *None*
        the ``SentenceTransformer`` default auto-detection is used.

    Examples
    --------
    >>> model = EmbeddingModel()
    >>> vecs = model.encode(["hello world", "how are you"])
    >>> vecs.shape
    (2, 768)
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        normalize: bool = _DEFAULT_NORMALIZE,
        device: str | None = None,
    ) -> None:
        self.model_name: str = model_name or settings.embedding_model
        self.batch_size: int = batch_size
        self.normalize: bool = normalize
        self.device: str | None = device

        # The underlying SentenceTransformer instance (lazy-loaded).
        self._model: SentenceTransformer | None = None

        logger.info(
            "EmbeddingModel configured: model_name={model_name} | "
            "batch_size={batch_size} | normalize={normalize} | device={device}",
            model_name=self.model_name,
            batch_size=self.batch_size,
            normalize=self.normalize,
            device=self.device or "auto",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> "SentenceTransformer":
        """Download / load the SentenceTransformer model on first use.

        Returns
        -------
        SentenceTransformer
            The loaded model instance.

        Raises
        ------
        ImportError
            If ``sentence-transformers`` is not installed.
        RuntimeError
            If model loading fails (network, corrupt cache, etc.).
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            logger.error(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
            raise ImportError(
                "sentence-transformers package is required for EmbeddingModel. "
                "Install with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading SentenceTransformer model: {model}", model=self.model_name)
        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
        except Exception as exc:
            logger.error(
                "Failed to load model '{model}': {err}",
                model=self.model_name,
                err=exc,
            )
            raise RuntimeError(
                f"Could not load SentenceTransformer model '{self.model_name}': {exc}"
            ) from exc

        dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Model loaded successfully. Embedding dimension: {dim}",
            dim=dim,
        )
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality of the loaded model.

        Returns
        -------
        int
            Number of floats per embedding vector (e.g. 768 for all-mpnet-base-v2).
        """
        model = self._load_model()
        dim: int = model.get_sentence_embedding_dimension()  # type: ignore[assignment]
        return dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into dense embeddings.

        Long input lists are automatically split into batches of size
        ``self.batch_size`` and the results are concatenated.

        Parameters
        ----------
        texts : list[str]
            The texts to embed.  Empty strings are permitted but will
            produce near-zero vectors.

        Returns
        -------
        np.ndarray
            A float32 array of shape ``(len(texts), dimension)``.

        Raises
        ------
        ValueError
            If *texts* is empty.
        RuntimeError
            If the underlying model fails to encode.
        """
        if not texts:
            raise ValueError("Cannot encode an empty list of texts.")

        model = self._load_model()
        n_texts = len(texts)
        n_batches = math.ceil(n_texts / self.batch_size)

        logger.debug(
            "Encoding {n} texts in {b} batch(es) (batch_size={bs})",
            n=n_texts,
            b=n_batches,
            bs=self.batch_size,
        )

        all_embeddings: list[np.ndarray] = []

        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, n_texts)
            batch = texts[start:end]

            try:
                embeddings: np.ndarray = model.encode(
                    batch,
                    show_progress_bar=False,
                    normalize_embeddings=self.normalize,
                    convert_to_numpy=True,
                )
            except Exception as exc:
                logger.error(
                    "Encoding failed for batch {idx}/{total}: {err}",
                    idx=batch_idx + 1,
                    total=n_batches,
                    err=exc,
                )
                raise RuntimeError(
                    f"Embedding encode failed on batch {batch_idx + 1}/{n_batches}: {exc}"
                ) from exc

            all_embeddings.append(embeddings)

            if n_batches > 1:
                logger.debug(
                    "Batch {idx}/{total} encoded ({count} texts)",
                    idx=batch_idx + 1,
                    total=n_batches,
                    count=len(batch),
                )

        result = np.vstack(all_embeddings).astype(np.float32)

        logger.debug(
            "Encoding complete. Output shape: {shape}",
            shape=result.shape,
        )
        return result

    def encode_single(self, text: str) -> np.ndarray:
        """Convenience wrapper to encode a single string.

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        np.ndarray
            A float32 array of shape ``(dimension,)``.
        """
        return self.encode([text])[0]

    def __repr__(self) -> str:
        return (
            f"EmbeddingModel(model_name={self.model_name!r}, "
            f"batch_size={self.batch_size}, normalize={self.normalize})"
        )
