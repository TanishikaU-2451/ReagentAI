"""FAISS Vector Store -- Dense Vector Storage and Similarity Search.

Provides a production-grade wrapper around Facebook AI Similarity Search
(FAISS) for storing document embeddings and retrieving the most
semantically similar passages to a query vector.

Key design decisions:
    * **IndexFlatIP** is used by default because the embedding model
      produces L2-normalised vectors, so inner-product search is
      equivalent to cosine similarity while avoiding the overhead of
      building an IVF index for moderate-scale corpora.
    * **Metadata sidecar** -- FAISS stores raw float vectors only; we
      maintain a parallel Python list of metadata dicts (text chunk,
      paper ID, chunk index, etc.) keyed by integer row position.
    * **Atomic persistence** -- :meth:`save` writes the FAISS index and
      the metadata pickle to ``settings.vector_store_path`` in a single
      logical operation, and :meth:`load` restores both.
    * **Thread safety** -- reads are safe to interleave from multiple
      threads.  Writes (``add_documents``, ``save``) should be
      serialised by the caller if concurrent access is needed.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from backend.config.settings import settings
from backend.utils.logging import get_logger

logger = get_logger("retrieval.vector_store")

# ---------------------------------------------------------------------------
# File-name constants used for persistence
# ---------------------------------------------------------------------------
_FAISS_INDEX_FILE = "index.faiss"
_METADATA_FILE = "metadata.pkl"
_STORE_INFO_FILE = "store_info.json"


class VectorStore:
    """FAISS-backed vector store with metadata sidecar.

    Parameters
    ----------
    dimension : int
        Dimensionality of the embedding vectors (e.g. 768 for
        all-mpnet-base-v2).
    store_path : Path | str | None
        Directory where the index and metadata are persisted.
        Defaults to ``settings.vector_store_path``.

    Attributes
    ----------
    dimension : int
        Embedding dimensionality that was set at construction time.
    index : faiss.IndexFlatIP | None
        The underlying FAISS index.  ``None`` until the first call to
        :meth:`add_documents` or :meth:`load`.
    metadata : list[dict[str, Any]]
        Parallel list of metadata dicts, one per stored vector.
    """

    def __init__(
        self,
        dimension: int,
        store_path: Path | str | None = None,
    ) -> None:
        self.dimension: int = dimension
        self.store_path: Path = Path(store_path or settings.vector_store_path)

        # Lazy-import faiss so the rest of the package stays importable
        # even if faiss-cpu is missing.
        self._faiss = _import_faiss()

        # Build an empty inner-product index.
        self.index: Any = self._faiss.IndexFlatIP(self.dimension)
        self.metadata: list[dict[str, Any]] = []

        logger.info(
            "VectorStore initialised: dimension={dim} | store_path={path}",
            dim=self.dimension,
            path=self.store_path,
        )

    # ------------------------------------------------------------------
    # Public API -- mutation
    # ------------------------------------------------------------------

    def add_documents(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """Add document embeddings and their associated metadata.

        Parameters
        ----------
        embeddings : np.ndarray
            Float32 array of shape ``(n, self.dimension)``.
        metadata : list[dict[str, Any]]
            A list of the same length as *embeddings*.  Each dict
            typically contains at least ``{"text": ..., "paper_id": ...,
            "chunk_index": ...}``.

        Raises
        ------
        ValueError
            If shapes are inconsistent or metadata length does not match.
        """
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2-D embeddings array, got shape {embeddings.shape}"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) does not match "
                f"number of metadata entries ({len(metadata)})"
            )

        self.index.add(embeddings)
        self.metadata.extend(metadata)

        logger.info(
            "Added {n} vectors to store (total now: {total})",
            n=embeddings.shape[0],
            total=self.index.ntotal,
        )

    def clear(self) -> None:
        """Remove all vectors and metadata, resetting the index."""
        self.index = self._faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info("Vector store cleared.")

    # ------------------------------------------------------------------
    # Public API -- search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Find the *top_k* most similar vectors to *query_embedding*.

        Parameters
        ----------
        query_embedding : np.ndarray
            A float32 array of shape ``(dimension,)`` or ``(1, dimension)``.
        top_k : int
            Number of nearest neighbours to return.

        Returns
        -------
        list[dict[str, Any]]
            Ranked list of result dicts, each containing:
                - ``"score"`` (float): inner-product similarity score.
                - ``"rank"`` (int): 1-based rank.
                - All keys from the stored metadata dict (typically
                  ``"text"``, ``"paper_id"``, ``"chunk_index"``).

        Raises
        ------
        ValueError
            If the store is empty or the query shape is wrong.
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on empty vector store; returning [].")
            return []

        query_embedding = np.ascontiguousarray(
            query_embedding, dtype=np.float32
        )
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, "
                f"got {query_embedding.shape[1]}"
            )

        # Clamp top_k to the number of stored vectors.
        effective_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, effective_k)

        results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx == -1:
                # FAISS returns -1 for unfilled slots when ntotal < k.
                continue
            entry: dict[str, Any] = {
                "rank": rank,
                "score": float(score),
                **self.metadata[int(idx)],
            }
            results.append(entry)

        logger.debug(
            "Search returned {n} results (requested top_k={k})",
            n=len(results),
            k=top_k,
        )
        return results

    # ------------------------------------------------------------------
    # Public API -- persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path:
        """Persist the FAISS index and metadata to disk.

        Parameters
        ----------
        path : Path | str | None
            Target directory.  Defaults to ``self.store_path``.

        Returns
        -------
        Path
            The directory the files were written to.

        Raises
        ------
        OSError
            If the directory cannot be created or files cannot be written.
        """
        save_dir = Path(path) if path is not None else self.store_path
        save_dir.mkdir(parents=True, exist_ok=True)

        index_path = save_dir / _FAISS_INDEX_FILE
        meta_path = save_dir / _METADATA_FILE
        info_path = save_dir / _STORE_INFO_FILE

        # Write FAISS index
        self._faiss.write_index(self.index, str(index_path))

        # Write metadata sidecar
        with open(meta_path, "wb") as fh:
            pickle.dump(self.metadata, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # Write human-readable store info
        store_info = {
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "metadata_entries": len(self.metadata),
        }
        with open(info_path, "w", encoding="utf-8") as fh:
            json.dump(store_info, fh, indent=2)

        logger.info(
            "Vector store saved to {path} ({n} vectors)",
            path=save_dir,
            n=self.index.ntotal,
        )
        return save_dir

    def load(self, path: Path | str | None = None) -> None:
        """Load a previously saved FAISS index and metadata from disk.

        Parameters
        ----------
        path : Path | str | None
            Source directory.  Defaults to ``self.store_path``.

        Raises
        ------
        FileNotFoundError
            If the index or metadata file does not exist.
        RuntimeError
            If the loaded index dimension does not match ``self.dimension``.
        """
        load_dir = Path(path) if path is not None else self.store_path
        index_path = load_dir / _FAISS_INDEX_FILE
        meta_path = load_dir / _METADATA_FILE

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}"
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {meta_path}"
            )

        loaded_index = self._faiss.read_index(str(index_path))

        # Validate dimension consistency.
        loaded_dim: int = loaded_index.d
        if loaded_dim != self.dimension:
            raise RuntimeError(
                f"Loaded index dimension ({loaded_dim}) does not match "
                f"expected dimension ({self.dimension})"
            )

        with open(meta_path, "rb") as fh:
            loaded_metadata: list[dict[str, Any]] = pickle.load(fh)  # noqa: S301

        if loaded_index.ntotal != len(loaded_metadata):
            logger.warning(
                "Metadata length ({meta}) does not match index size ({idx}). "
                "The store may be corrupted.",
                meta=len(loaded_metadata),
                idx=loaded_index.ntotal,
            )

        self.index = loaded_index
        self.metadata = loaded_metadata

        logger.info(
            "Vector store loaded from {path} ({n} vectors, {m} metadata entries)",
            path=load_dir,
            n=self.index.ntotal,
            m=len(self.metadata),
        )

    def exists_on_disk(self, path: Path | str | None = None) -> bool:
        """Check whether a persisted store exists at the given path.

        Parameters
        ----------
        path : Path | str | None
            Directory to check.  Defaults to ``self.store_path``.

        Returns
        -------
        bool
        """
        check_dir = Path(path) if path is not None else self.store_path
        return (
            (check_dir / _FAISS_INDEX_FILE).exists()
            and (check_dir / _METADATA_FILE).exists()
        )

    # ------------------------------------------------------------------
    # Informational helpers
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Return the number of vectors currently stored."""
        return int(self.index.ntotal)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"VectorStore(dimension={self.dimension}, "
            f"size={self.size}, store_path={str(self.store_path)!r})"
        )


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------

def _import_faiss() -> Any:
    """Import the ``faiss`` package, raising a clear error if missing.

    Returns
    -------
    module
        The ``faiss`` module.
    """
    try:
        import faiss  # type: ignore[import-untyped]

        return faiss
    except ImportError as exc:
        logger.error(
            "faiss-cpu is not installed.  "
            "Install it with: pip install faiss-cpu"
        )
        raise ImportError(
            "faiss-cpu package is required for VectorStore. "
            "Install with: pip install faiss-cpu"
        ) from exc
