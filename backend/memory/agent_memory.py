"""Agent Memory Store -- Vector-Indexed Long-Term Memory for Agents.

Gives every agent in the ReagentAI pipeline the ability to persist
observations, decisions, and findings to a FAISS-backed vector store and
later retrieve the most semantically relevant memories.

Architecture overview::

    +-----------+        embed          +-------+
    |  store()  | ----> EmbeddingModel  | FAISS |
    +-----------+        encode         | Index |
                                        +---+---+
    +-----------+        search             |
    | recall()  | <-------------------------+
    +-----------+

Each memory type (``research_memory``, ``architecture_memory``,
``debug_memory``) is backed by its own FAISS index so that recall
queries are scoped to the relevant domain, and cross-contamination
between unrelated agent contexts is avoided.

Persistence layout (under ``settings.vector_store_path / "agent_memory"``)::

    agent_memory/
        research_memory/
            index.faiss
            metadata.pkl
        architecture_memory/
            index.faiss
            metadata.pkl
        debug_memory/
            index.faiss
            metadata.pkl

Thread safety
-------------
Read operations (:meth:`recall`, :meth:`get_agent_memories`) are safe
to interleave from multiple threads.  Write operations (:meth:`store`,
:meth:`clear`, :meth:`save`, :meth:`load`) must be serialised by the
caller when concurrent access is required.
"""

from __future__ import annotations

import pickle
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from backend.config.settings import settings
from backend.retrieval.embedding_model import EmbeddingModel
from backend.utils.logging import get_logger

logger = get_logger("memory.agent_memory")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MEMORY_TYPES: frozenset[str] = frozenset(
    {
        "research_memory",
        "architecture_memory",
        "debug_memory",
    }
)

_FAISS_INDEX_FILE: str = "index.faiss"
_METADATA_FILE: str = "metadata.pkl"
_DEFAULT_SUBDIRECTORY: str = "agent_memory"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """A single memory record stored and retrieved by the agent memory system.

    Attributes
    ----------
    id : str
        A unique identifier for this memory entry (UUID4 hex string).
    agent_name : str
        Name of the agent that created this memory (e.g. ``"ResearchAgent"``).
    content : str
        The free-text content of the memory.
    memory_type : str
        One of the ``VALID_MEMORY_TYPES`` -- determines which index the
        entry is stored in.
    timestamp : float
        Unix epoch timestamp at which the memory was created.
    metadata : dict[str, Any]
        Arbitrary extra data attached by the caller (paper IDs, error
        hashes, configuration snapshots, etc.).
    relevance_score : float
        Cosine-similarity score assigned during :meth:`AgentMemoryStore.recall`.
        Defaults to ``0.0`` for entries that were not produced by a
        similarity search (e.g. those returned by
        :meth:`AgentMemoryStore.get_agent_memories`).
    """

    id: str
    agent_name: str
    content: str
    memory_type: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the entry to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            All fields as a JSON-friendly dict.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """Reconstruct a ``MemoryEntry`` from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary with keys matching the dataclass fields.

        Returns
        -------
        MemoryEntry
        """
        return cls(
            id=data["id"],
            agent_name=data["agent_name"],
            content=data["content"],
            memory_type=data["memory_type"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
            relevance_score=data.get("relevance_score", 0.0),
        )

    def __repr__(self) -> str:
        content_preview = (
            self.content[:60] + "..." if len(self.content) > 60 else self.content
        )
        return (
            f"MemoryEntry(id={self.id!r}, agent={self.agent_name!r}, "
            f"type={self.memory_type!r}, score={self.relevance_score:.4f}, "
            f"content={content_preview!r})"
        )


# ---------------------------------------------------------------------------
# Internal per-type index container
# ---------------------------------------------------------------------------


class _MemoryIndex:
    """Manages a single FAISS index and its metadata sidecar for one memory type.

    This is an internal helper; external callers should use
    :class:`AgentMemoryStore` exclusively.

    Parameters
    ----------
    memory_type : str
        The memory category this index serves.
    dimension : int
        Embedding vector dimensionality.
    store_dir : Path
        Directory where this index is persisted.
    """

    def __init__(self, memory_type: str, dimension: int, store_dir: Path) -> None:
        self.memory_type = memory_type
        self.dimension = dimension
        self.store_dir = store_dir

        self._faiss = _import_faiss()
        self.index: Any = self._faiss.IndexFlatIP(self.dimension)
        self.entries: list[dict[str, Any]] = []

        logger.debug(
            "MemoryIndex created: type={mtype} | dim={dim} | dir={dir}",
            mtype=self.memory_type,
            dim=self.dimension,
            dir=self.store_dir,
        )

    # -- Mutation ----------------------------------------------------------

    def add(self, embedding: np.ndarray, entry_dict: dict[str, Any]) -> None:
        """Add a single embedding and its associated entry dictionary.

        Parameters
        ----------
        embedding : np.ndarray
            Float32 vector of shape ``(dimension,)``.
        entry_dict : dict[str, Any]
            Serialised :class:`MemoryEntry` dictionary.
        """
        vec = np.ascontiguousarray(embedding, dtype=np.float32).reshape(1, -1)
        self.index.add(vec)
        self.entries.append(entry_dict)

    def clear(self) -> None:
        """Wipe the index and all metadata."""
        self.index = self._faiss.IndexFlatIP(self.dimension)
        self.entries = []
        logger.info(
            "MemoryIndex cleared: type={mtype}",
            mtype=self.memory_type,
        )

    # -- Search ------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        """Return the *top_k* most similar entries.

        Parameters
        ----------
        query_embedding : np.ndarray
            Float32 vector of shape ``(dimension,)``.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[dict[str, Any]]
            Entry dicts augmented with ``"relevance_score"``.
        """
        if self.index.ntotal == 0:
            return []

        vec = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
        effective_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vec, effective_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.entries[int(idx)])
            entry["relevance_score"] = float(score)
            results.append(entry)

        return results

    def get_all_by_agent(self, agent_name: str) -> list[dict[str, Any]]:
        """Return every entry created by *agent_name*, newest first.

        Parameters
        ----------
        agent_name : str
            The agent whose memories to retrieve.

        Returns
        -------
        list[dict[str, Any]]
            Matching entries sorted by descending timestamp.
        """
        matches = [
            dict(e) for e in self.entries if e.get("agent_name") == agent_name
        ]
        matches.sort(key=lambda e: e.get("timestamp", 0.0), reverse=True)
        return matches

    # -- Persistence -------------------------------------------------------

    def save(self) -> None:
        """Write the FAISS index and metadata to ``self.store_dir``."""
        self.store_dir.mkdir(parents=True, exist_ok=True)

        index_path = self.store_dir / _FAISS_INDEX_FILE
        meta_path = self.store_dir / _METADATA_FILE

        self._faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as fh:
            pickle.dump(self.entries, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug(
            "MemoryIndex saved: type={mtype} | entries={n} | path={path}",
            mtype=self.memory_type,
            n=len(self.entries),
            path=self.store_dir,
        )

    def load(self) -> bool:
        """Load a previously saved index from ``self.store_dir``.

        Returns
        -------
        bool
            ``True`` if loading succeeded, ``False`` if the files were
            not found on disk.
        """
        index_path = self.store_dir / _FAISS_INDEX_FILE
        meta_path = self.store_dir / _METADATA_FILE

        if not index_path.exists() or not meta_path.exists():
            logger.debug(
                "No persisted index found for type={mtype} at {path}",
                mtype=self.memory_type,
                path=self.store_dir,
            )
            return False

        loaded_index = self._faiss.read_index(str(index_path))
        loaded_dim: int = loaded_index.d
        if loaded_dim != self.dimension:
            logger.error(
                "Dimension mismatch loading MemoryIndex: expected {exp}, got {got}",
                exp=self.dimension,
                got=loaded_dim,
            )
            raise RuntimeError(
                f"Loaded index dimension ({loaded_dim}) does not match "
                f"expected dimension ({self.dimension}) for memory type "
                f"'{self.memory_type}'"
            )

        with open(meta_path, "rb") as fh:
            loaded_entries: list[dict[str, Any]] = pickle.load(fh)  # noqa: S301

        if loaded_index.ntotal != len(loaded_entries):
            logger.warning(
                "Metadata/index size mismatch for {mtype}: index={idx}, meta={meta}. "
                "Store may be corrupted.",
                mtype=self.memory_type,
                idx=loaded_index.ntotal,
                meta=len(loaded_entries),
            )

        self.index = loaded_index
        self.entries = loaded_entries

        logger.info(
            "MemoryIndex loaded: type={mtype} | entries={n}",
            mtype=self.memory_type,
            n=len(self.entries),
        )
        return True

    @property
    def size(self) -> int:
        """Return the number of stored entries."""
        return int(self.index.ntotal)

    def __repr__(self) -> str:
        return (
            f"_MemoryIndex(type={self.memory_type!r}, "
            f"size={self.size}, dim={self.dimension})"
        )


# ---------------------------------------------------------------------------
# Public API -- AgentMemoryStore
# ---------------------------------------------------------------------------


class AgentMemoryStore:
    """Unified, vector-indexed long-term memory for all ReagentAI agents.

    Maintains one FAISS ``IndexFlatIP`` per memory type so that recall
    queries are scoped to the relevant domain.  All embeddings are
    produced by the shared :class:`~backend.retrieval.embedding_model.EmbeddingModel`
    and are L2-normalised, meaning inner-product search is equivalent to
    cosine similarity.

    Parameters
    ----------
    embedding_model : EmbeddingModel | None
        Pre-configured embedding model instance.  When ``None`` (the
        default), a fresh :class:`EmbeddingModel` is created using the
        model name from ``settings.embedding_model``.
    store_path : Path | str | None
        Root directory under which per-type sub-directories are created.
        Defaults to ``settings.vector_store_path / "agent_memory"``.
    auto_load : bool
        If ``True`` (default), automatically load any previously
        persisted indexes from disk at construction time.
    auto_save : bool
        If ``True`` (default), automatically persist all indexes to disk
        after every :meth:`store` call.  Set to ``False`` for batch
        workloads where explicit :meth:`save` calls are preferred.

    Examples
    --------
    >>> mem = AgentMemoryStore()
    >>> mem.store(
    ...     agent_name="ResearchAgent",
    ...     content="The paper uses a U-Net backbone.",
    ...     memory_type="research_memory",
    ...     metadata={"paper_id": "arxiv:2401.99999"},
    ... )
    >>> results = mem.recall("U-Net architecture", "research_memory", top_k=3)
    >>> results[0].content
    'The paper uses a U-Net backbone.'
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        store_path: Path | str | None = None,
        auto_load: bool = True,
        auto_save: bool = True,
    ) -> None:
        self._embedder: EmbeddingModel = embedding_model or EmbeddingModel()
        self._store_root: Path = Path(
            store_path or (Path(settings.vector_store_path) / _DEFAULT_SUBDIRECTORY)
        )
        self._auto_save: bool = auto_save

        # Lazily resolved on first embed call.
        self._dimension: int | None = None

        # Per-type indexes are built on demand.
        self._indexes: dict[str, _MemoryIndex] = {}

        logger.info(
            "AgentMemoryStore initialising: store_root={root} | "
            "auto_load={al} | auto_save={as_}",
            root=self._store_root,
            al=auto_load,
            as_=auto_save,
        )

        if auto_load:
            self._load_all()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_dimension(self) -> int:
        """Determine the embedding dimensionality (lazy, cached).

        Returns
        -------
        int
            Number of floats per embedding vector.
        """
        if self._dimension is None:
            self._dimension = self._embedder.dimension
            logger.debug(
                "Embedding dimension resolved: {dim}", dim=self._dimension
            )
        return self._dimension

    def _get_index(self, memory_type: str) -> _MemoryIndex:
        """Return (or create) the ``_MemoryIndex`` for *memory_type*.

        Parameters
        ----------
        memory_type : str
            Must be one of ``VALID_MEMORY_TYPES``.

        Returns
        -------
        _MemoryIndex
        """
        if memory_type not in self._indexes:
            dim = self._resolve_dimension()
            sub_dir = self._store_root / memory_type
            idx = _MemoryIndex(
                memory_type=memory_type,
                dimension=dim,
                store_dir=sub_dir,
            )
            self._indexes[memory_type] = idx
            logger.debug(
                "Created new MemoryIndex for type={mtype}", mtype=memory_type
            )
        return self._indexes[memory_type]

    @staticmethod
    def _validate_memory_type(memory_type: str) -> None:
        """Raise ``ValueError`` if *memory_type* is not recognised.

        Parameters
        ----------
        memory_type : str
            The memory type string to validate.

        Raises
        ------
        ValueError
            If *memory_type* is not in ``VALID_MEMORY_TYPES``.
        """
        if memory_type not in VALID_MEMORY_TYPES:
            raise ValueError(
                f"Invalid memory_type '{memory_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_MEMORY_TYPES))}"
            )

    def _embed(self, text: str) -> np.ndarray:
        """Encode a single text string into an embedding vector.

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        np.ndarray
            Float32 vector of shape ``(dimension,)``.
        """
        return self._embedder.encode_single(text)

    def _load_all(self) -> None:
        """Attempt to load persisted indexes for every valid memory type."""
        for mtype in VALID_MEMORY_TYPES:
            sub_dir = self._store_root / mtype
            if (sub_dir / _FAISS_INDEX_FILE).exists():
                idx = self._get_index(mtype)
                try:
                    idx.load()
                except Exception as exc:
                    logger.error(
                        "Failed to load persisted index for {mtype}: {err}",
                        mtype=mtype,
                        err=exc,
                    )

    # ------------------------------------------------------------------
    # Public API -- store
    # ------------------------------------------------------------------

    def store(
        self,
        agent_name: str,
        content: str,
        memory_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Create and persist a new memory entry.

        The *content* string is embedded using the shared
        :class:`EmbeddingModel` and stored in the FAISS index
        corresponding to *memory_type*.

        Parameters
        ----------
        agent_name : str
            The agent creating this memory (e.g. ``"ResearchAgent"``).
        content : str
            Free-text content to store and later recall.
        memory_type : str
            One of ``"research_memory"``, ``"architecture_memory"``,
            ``"debug_memory"``.
        metadata : dict[str, Any] | None
            Optional extra data to attach to the entry.

        Returns
        -------
        MemoryEntry
            The newly created entry, including its generated ``id`` and
            ``timestamp``.

        Raises
        ------
        ValueError
            If *memory_type* is invalid or *content* is empty.
        RuntimeError
            If embedding or FAISS insertion fails.
        """
        self._validate_memory_type(memory_type)

        if not content or not content.strip():
            raise ValueError("Memory content must be a non-empty string.")

        entry = MemoryEntry(
            id=uuid.uuid4().hex,
            agent_name=agent_name,
            content=content,
            memory_type=memory_type,
            timestamp=time.time(),
            metadata=metadata or {},
            relevance_score=0.0,
        )

        logger.info(
            "Storing memory: id={mid} | agent={agent} | type={mtype} | "
            "content_len={clen}",
            mid=entry.id,
            agent=agent_name,
            mtype=memory_type,
            clen=len(content),
        )

        try:
            embedding = self._embed(content)
        except Exception as exc:
            logger.error(
                "Embedding failed for memory {mid}: {err}",
                mid=entry.id,
                err=exc,
            )
            raise RuntimeError(
                f"Failed to embed memory content: {exc}"
            ) from exc

        idx = self._get_index(memory_type)
        idx.add(embedding, entry.to_dict())

        logger.debug(
            "Memory {mid} added to {mtype} index (total: {n})",
            mid=entry.id,
            mtype=memory_type,
            n=idx.size,
        )

        if self._auto_save:
            try:
                idx.save()
            except Exception as exc:
                logger.error(
                    "Auto-save failed for {mtype} after storing {mid}: {err}",
                    mtype=memory_type,
                    mid=entry.id,
                    err=exc,
                )

        return entry

    # ------------------------------------------------------------------
    # Public API -- recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        memory_type: str,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """Retrieve the most semantically relevant memories for *query*.

        The *query* is embedded and compared against all entries in the
        FAISS index for *memory_type* using cosine similarity (via
        inner-product on L2-normalised vectors).

        Parameters
        ----------
        query : str
            Natural-language query describing the desired context.
        memory_type : str
            Which memory index to search.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[MemoryEntry]
            Up to *top_k* entries sorted by descending
            ``relevance_score``.

        Raises
        ------
        ValueError
            If *memory_type* is invalid or *query* is empty.
        """
        self._validate_memory_type(memory_type)

        if not query or not query.strip():
            raise ValueError("Recall query must be a non-empty string.")

        idx = self._get_index(memory_type)

        if idx.size == 0:
            logger.debug(
                "Recall on empty index for type={mtype}; returning [].",
                mtype=memory_type,
            )
            return []

        logger.info(
            "Recalling memories: type={mtype} | top_k={k} | query_len={qlen}",
            mtype=memory_type,
            k=top_k,
            qlen=len(query),
        )

        try:
            query_embedding = self._embed(query)
        except Exception as exc:
            logger.error("Embedding failed during recall: {err}", err=exc)
            raise RuntimeError(
                f"Failed to embed recall query: {exc}"
            ) from exc

        raw_results = idx.search(query_embedding, top_k)

        entries = [MemoryEntry.from_dict(r) for r in raw_results]

        logger.debug(
            "Recall returned {n} results for type={mtype}",
            n=len(entries),
            mtype=memory_type,
        )
        return entries

    # ------------------------------------------------------------------
    # Public API -- agent-scoped retrieval
    # ------------------------------------------------------------------

    def get_agent_memories(
        self,
        agent_name: str,
        memory_type: str | None = None,
    ) -> list[MemoryEntry]:
        """Return every stored memory created by *agent_name*.

        Parameters
        ----------
        agent_name : str
            The agent whose memories to retrieve.
        memory_type : str | None
            If provided, restrict results to this memory type.  When
            ``None``, search across all memory types.

        Returns
        -------
        list[MemoryEntry]
            Matching entries sorted by descending timestamp (newest
            first).  ``relevance_score`` is ``0.0`` for all entries
            because no similarity search is performed.

        Raises
        ------
        ValueError
            If *memory_type* is provided but invalid.
        """
        if memory_type is not None:
            self._validate_memory_type(memory_type)
            types_to_search = [memory_type]
        else:
            types_to_search = list(VALID_MEMORY_TYPES)

        logger.info(
            "Fetching all memories for agent={agent} | types={types}",
            agent=agent_name,
            types=types_to_search,
        )

        all_entries: list[MemoryEntry] = []
        for mtype in types_to_search:
            if mtype not in self._indexes:
                # Index was never created for this type; skip.
                continue
            idx = self._indexes[mtype]
            raw = idx.get_all_by_agent(agent_name)
            all_entries.extend(MemoryEntry.from_dict(r) for r in raw)

        # Sort globally by timestamp descending.
        all_entries.sort(key=lambda e: e.timestamp, reverse=True)

        logger.debug(
            "Found {n} memories for agent={agent}",
            n=len(all_entries),
            agent=agent_name,
        )
        return all_entries

    # ------------------------------------------------------------------
    # Public API -- clear
    # ------------------------------------------------------------------

    def clear(self, memory_type: str) -> None:
        """Delete all entries in the specified memory-type index.

        The on-disk files are also overwritten (emptied) if
        ``auto_save`` is enabled.

        Parameters
        ----------
        memory_type : str
            The memory index to wipe.

        Raises
        ------
        ValueError
            If *memory_type* is invalid.
        """
        self._validate_memory_type(memory_type)

        logger.info(
            "Clearing all memories for type={mtype}", mtype=memory_type
        )

        idx = self._get_index(memory_type)
        idx.clear()

        if self._auto_save:
            try:
                idx.save()
            except Exception as exc:
                logger.error(
                    "Auto-save after clear failed for {mtype}: {err}",
                    mtype=memory_type,
                    err=exc,
                )

    def clear_all(self) -> None:
        """Delete all entries across every memory type.

        Convenience wrapper that calls :meth:`clear` for each type
        in ``VALID_MEMORY_TYPES``.
        """
        logger.info("Clearing ALL agent memory indexes.")
        for mtype in VALID_MEMORY_TYPES:
            self.clear(mtype)

    # ------------------------------------------------------------------
    # Public API -- persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path:
        """Persist all memory-type indexes to disk.

        Parameters
        ----------
        path : Path | str | None
            Override root directory.  Defaults to ``self._store_root``.

        Returns
        -------
        Path
            The root directory that was written to.
        """
        save_root = Path(path) if path is not None else self._store_root
        save_root.mkdir(parents=True, exist_ok=True)

        total_entries = 0
        for mtype, idx in self._indexes.items():
            target_dir = save_root / mtype
            idx.store_dir = target_dir
            idx.save()
            total_entries += idx.size

        logger.info(
            "All memory indexes saved to {root} ({n} total entries)",
            root=save_root,
            n=total_entries,
        )
        return save_root

    def load(self, path: Path | str | None = None) -> None:
        """Load all persisted memory-type indexes from disk.

        Parameters
        ----------
        path : Path | str | None
            Override root directory.  Defaults to ``self._store_root``.
        """
        load_root = Path(path) if path is not None else self._store_root

        logger.info(
            "Loading memory indexes from {root}", root=load_root
        )

        for mtype in VALID_MEMORY_TYPES:
            sub_dir = load_root / mtype
            if (sub_dir / _FAISS_INDEX_FILE).exists():
                idx = self._get_index(mtype)
                idx.store_dir = sub_dir
                try:
                    idx.load()
                except Exception as exc:
                    logger.error(
                        "Failed to load index for {mtype}: {err}",
                        mtype=mtype,
                        err=exc,
                    )

    # ------------------------------------------------------------------
    # Informational helpers
    # ------------------------------------------------------------------

    @property
    def total_memories(self) -> int:
        """Return the total number of stored memories across all types.

        Returns
        -------
        int
        """
        return sum(idx.size for idx in self._indexes.values())

    def memory_counts(self) -> dict[str, int]:
        """Return per-type memory counts.

        Returns
        -------
        dict[str, int]
            Mapping of memory type to entry count.
        """
        return {
            mtype: self._indexes[mtype].size
            for mtype in VALID_MEMORY_TYPES
            if mtype in self._indexes
        }

    def __repr__(self) -> str:
        counts = self.memory_counts()
        count_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        return (
            f"AgentMemoryStore(total={self.total_memories}, "
            f"store_root={str(self._store_root)!r}, {count_str})"
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

    Raises
    ------
    ImportError
        If ``faiss-cpu`` is not installed.
    """
    try:
        import faiss  # type: ignore[import-untyped]
        return faiss
    except ImportError as exc:
        logger.error(
            "faiss-cpu is not installed. "
            "Install it with: pip install faiss-cpu"
        )
        raise ImportError(
            "faiss-cpu package is required for AgentMemoryStore. "
            "Install with: pip install faiss-cpu"
        ) from exc
