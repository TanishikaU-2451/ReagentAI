"""RAG Pipeline -- End-to-End Retrieval-Augmented Generation.

Orchestrates the full RAG workflow for ReagentAI:

1. **Chunking** -- split long documents (research papers) into
   overlapping segments small enough to embed meaningfully.
2. **Embedding** -- convert each chunk into a dense vector using the
   :class:`~backend.retrieval.embedding_model.EmbeddingModel`.
3. **Indexing** -- store vectors and their metadata in the
   :class:`~backend.retrieval.vector_store.VectorStore`.
4. **Retrieval** -- given a natural-language question, find the most
   relevant chunks via semantic similarity search.
5. **Context building** -- concatenate the top-ranked chunks into a
   single string suitable for injection into an LLM prompt.

Typical usage::

    from backend.retrieval.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.index_paper(paper_text, paper_id="arxiv:2301.00001")
    context = rag.build_context("What loss function is used?")
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from backend.config.settings import settings
from backend.utils.logging import get_logger
from backend.retrieval.embedding_model import EmbeddingModel
from backend.retrieval.vector_store import VectorStore

logger = get_logger("retrieval.rag_pipeline")

# ---------------------------------------------------------------------------
# Default chunking hyper-parameters
# ---------------------------------------------------------------------------
_DEFAULT_CHUNK_SIZE: int = 512
_DEFAULT_CHUNK_OVERLAP: int = 50
_DEFAULT_TOP_K: int = 5

# Separator used when concatenating chunks for LLM context.
_CONTEXT_SEPARATOR: str = "\n\n---\n\n"


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    The pipeline owns an :class:`EmbeddingModel` and a
    :class:`VectorStore` and exposes high-level methods for indexing
    documents and answering questions.

    Parameters
    ----------
    chunk_size : int
        Maximum number of **characters** per text chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.
        Overlap helps ensure that sentences split across chunk boundaries
        are still retrievable.
    top_k : int
        Default number of chunks to retrieve per query.
    embedding_model : EmbeddingModel | None
        Pre-configured embedding model.  If *None*, a new one is created
        from ``settings``.
    vector_store : VectorStore | None
        Pre-configured vector store.  If *None*, a new one is created
        once the embedding dimension is known (i.e. on the first call
        that requires the model).
    store_path : Path | str | None
        Persistence directory for the vector store.  Defaults to
        ``settings.vector_store_path``.
    auto_persist : bool
        If *True* (default), the vector store is automatically saved to
        disk after every :meth:`index_paper` call.

    Examples
    --------
    >>> rag = RAGPipeline()
    >>> rag.index_paper("We propose a novel loss ...", paper_id="abc123")
    >>> results = rag.query_paper("What loss is used?")
    >>> context = rag.build_context("Explain the architecture.")
    """

    def __init__(
        self,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        top_k: int = _DEFAULT_TOP_K,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        store_path: Path | str | None = None,
        auto_persist: bool = True,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )

        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.top_k: int = top_k
        self.auto_persist: bool = auto_persist
        self.store_path: Path = Path(store_path or settings.vector_store_path)

        self._embedding_model: EmbeddingModel = embedding_model or EmbeddingModel()
        self._vector_store: VectorStore | None = vector_store

        # Track which papers have been indexed to avoid duplicates.
        self._indexed_paper_ids: set[str] = set()

        logger.info(
            "RAGPipeline initialised: chunk_size={cs} | overlap={co} | "
            "top_k={k} | auto_persist={ap}",
            cs=self.chunk_size,
            co=self.chunk_overlap,
            k=self.top_k,
            ap=self.auto_persist,
        )

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_vector_store(self) -> VectorStore:
        """Return the vector store, creating it lazily if needed.

        The store is created once we know the embedding dimension (which
        requires loading the model), so this method triggers model
        loading on first access.
        """
        if self._vector_store is not None:
            return self._vector_store

        dim = self._embedding_model.dimension
        self._vector_store = VectorStore(
            dimension=dim,
            store_path=self.store_path,
        )

        # If a persisted store already exists, restore it.
        if self._vector_store.exists_on_disk():
            logger.info(
                "Found existing vector store at {path}; loading.",
                path=self.store_path,
            )
            self._vector_store.load()
            # Rebuild the indexed-paper-ids set from metadata.
            self._indexed_paper_ids = {
                meta.get("paper_id", "")
                for meta in self._vector_store.metadata
                if meta.get("paper_id")
            }
            logger.info(
                "Restored {n} indexed paper(s) from disk.",
                n=len(self._indexed_paper_ids),
            )

        return self._vector_store

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        """Collapse runs of whitespace into single spaces and strip."""
        return re.sub(r"\s+", " ", text).strip()

    def chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[str]:
        """Split *text* into overlapping character-level chunks.

        The algorithm walks over the normalised text with a sliding
        window of *chunk_size* characters, advancing by
        ``chunk_size - chunk_overlap`` characters each step.

        Parameters
        ----------
        text : str
            The document text to chunk.
        chunk_size : int | None
            Override the pipeline default.
        chunk_overlap : int | None
            Override the pipeline default.

        Returns
        -------
        list[str]
            Non-empty text chunks.  The last chunk may be shorter than
            *chunk_size*.
        """
        cs = chunk_size or self.chunk_size
        co = chunk_overlap or self.chunk_overlap

        text = self._normalise_whitespace(text)

        if not text:
            logger.warning("chunk_text received empty text; returning [].")
            return []

        if len(text) <= cs:
            return [text]

        step = cs - co
        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + cs
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step

        logger.debug(
            "Chunked text into {n} chunk(s) (chunk_size={cs}, overlap={co})",
            n=len(chunks),
            cs=cs,
            co=co,
        )
        return chunks

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_paper(
        self,
        paper_text: str,
        paper_id: str,
        extra_metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> int:
        """Chunk, embed, and store a research paper.

        Parameters
        ----------
        paper_text : str
            The full text of the paper.
        paper_id : str
            A unique identifier for the paper (e.g. arXiv ID, DOI, or
            internal UUID).  Used to prevent duplicate indexing.
        extra_metadata : dict[str, Any] | None
            Additional key-value pairs to attach to every chunk's
            metadata (e.g. ``{"title": ..., "authors": ...}``).
        force : bool
            If *True*, re-index the paper even if it was already indexed.

        Returns
        -------
        int
            The number of chunks that were indexed.

        Raises
        ------
        ValueError
            If *paper_text* is empty.
        """
        if not paper_text or not paper_text.strip():
            raise ValueError("Cannot index empty paper text.")

        if paper_id in self._indexed_paper_ids and not force:
            logger.info(
                "Paper '{pid}' is already indexed; skipping. "
                "Pass force=True to re-index.",
                pid=paper_id,
            )
            return 0

        logger.info(
            "Indexing paper '{pid}' ({length} chars)...",
            pid=paper_id,
            length=len(paper_text),
        )

        # 1. Chunk
        chunks = self.chunk_text(paper_text)
        if not chunks:
            logger.warning(
                "Paper '{pid}' produced zero chunks after splitting.",
                pid=paper_id,
            )
            return 0

        # 2. Embed
        embeddings: np.ndarray = self._embedding_model.encode(chunks)

        # 3. Build metadata
        base_meta = extra_metadata or {}
        metadata_list: list[dict[str, Any]] = [
            {
                "text": chunk,
                "paper_id": paper_id,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                **base_meta,
            }
            for idx, chunk in enumerate(chunks)
        ]

        # 4. Store
        store = self._get_vector_store()
        store.add_documents(embeddings, metadata_list)
        self._indexed_paper_ids.add(paper_id)

        logger.info(
            "Paper '{pid}' indexed: {n} chunks stored (store total: {t})",
            pid=paper_id,
            n=len(chunks),
            t=store.size,
        )

        # 5. Optionally persist
        if self.auto_persist:
            store.save()

        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query_paper(
        self,
        question: str,
        top_k: int | None = None,
        paper_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant chunks for a question.

        Parameters
        ----------
        question : str
            Natural-language question or search query.
        top_k : int | None
            Number of results to return.  Defaults to ``self.top_k``.
        paper_id : str | None
            If given, post-filter results to only include chunks from
            this paper.

        Returns
        -------
        list[dict[str, Any]]
            Ranked list of dicts, each containing ``"text"``,
            ``"score"``, ``"rank"``, ``"paper_id"``, ``"chunk_index"``,
            and any extra metadata that was stored.
        """
        if not question or not question.strip():
            logger.warning("query_paper received empty question; returning [].")
            return []

        k = top_k or self.top_k
        store = self._get_vector_store()

        if store.size == 0:
            logger.warning("Vector store is empty; cannot answer query.")
            return []

        # Embed the question.
        query_vec: np.ndarray = self._embedding_model.encode_single(question)

        # If filtering by paper_id, request more results then filter.
        search_k = k * 3 if paper_id else k
        raw_results = store.search(query_vec, top_k=search_k)

        # Optional paper-level filter.
        if paper_id:
            raw_results = [
                r for r in raw_results if r.get("paper_id") == paper_id
            ]

        # Trim to requested count and re-rank.
        results = raw_results[:k]
        for new_rank, result in enumerate(results, start=1):
            result["rank"] = new_rank

        logger.info(
            "query_paper: question={q!r} | returned {n}/{k} results",
            q=textwrap.shorten(question, width=80),
            n=len(results),
            k=k,
        )
        return results

    def build_context(
        self,
        question: str,
        top_k: int | None = None,
        paper_id: str | None = None,
        max_context_chars: int = 8000,
    ) -> str:
        """Build a concatenated context string from top retrieved chunks.

        This is the primary method used to inject retrieved knowledge
        into an LLM prompt.  Chunks are separated by a horizontal rule
        and include a brief header showing their source.

        Parameters
        ----------
        question : str
            The user's question.
        top_k : int | None
            Number of chunks to retrieve.  Defaults to ``self.top_k``.
        paper_id : str | None
            Optional paper filter.
        max_context_chars : int
            Hard limit on the total length of the returned context
            string.  Chunks are added in rank order until this budget
            is exhausted.

        Returns
        -------
        str
            A formatted string containing the most relevant chunks,
            or an empty string if no results are found.
        """
        results = self.query_paper(
            question, top_k=top_k, paper_id=paper_id
        )

        if not results:
            logger.info("build_context: no results for question={q!r}", q=question)
            return ""

        context_parts: list[str] = []
        char_budget = max_context_chars

        for result in results:
            chunk_text = result.get("text", "")
            source = result.get("paper_id", "unknown")
            chunk_idx = result.get("chunk_index", "?")
            score = result.get("score", 0.0)

            header = f"[Source: {source} | Chunk {chunk_idx} | Score: {score:.4f}]"
            block = f"{header}\n{chunk_text}"

            if len(block) > char_budget:
                # Truncate the last chunk to fit the budget.
                if char_budget > len(header) + 10:
                    truncated = chunk_text[: char_budget - len(header) - 5] + " ..."
                    block = f"{header}\n{truncated}"
                    context_parts.append(block)
                break

            context_parts.append(block)
            char_budget -= len(block) + len(_CONTEXT_SEPARATOR)

        context = _CONTEXT_SEPARATOR.join(context_parts)

        logger.info(
            "build_context: assembled {n} chunks ({chars} chars) for question={q!r}",
            n=len(context_parts),
            chars=len(context),
            q=textwrap.shorten(question, width=80),
        )
        return context

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path:
        """Explicitly save the vector store to disk.

        Parameters
        ----------
        path : Path | str | None
            Override the default store path.

        Returns
        -------
        Path
            The directory the store was saved to.
        """
        store = self._get_vector_store()
        return store.save(path)

    def load(self, path: Path | str | None = None) -> None:
        """Explicitly load the vector store from disk.

        Parameters
        ----------
        path : Path | str | None
            Override the default store path.
        """
        store = self._get_vector_store()
        store.load(path)
        # Rebuild paper-id cache.
        self._indexed_paper_ids = {
            meta.get("paper_id", "")
            for meta in store.metadata
            if meta.get("paper_id")
        }

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    @property
    def num_indexed_papers(self) -> int:
        """Return the count of distinct papers that have been indexed."""
        return len(self._indexed_paper_ids)

    @property
    def num_chunks(self) -> int:
        """Return the total number of chunks in the store."""
        if self._vector_store is None:
            return 0
        return self._vector_store.size

    @property
    def indexed_paper_ids(self) -> frozenset[str]:
        """Return the set of paper IDs currently indexed."""
        return frozenset(self._indexed_paper_ids)

    def __repr__(self) -> str:
        return (
            f"RAGPipeline(chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, top_k={self.top_k}, "
            f"papers={self.num_indexed_papers}, chunks={self.num_chunks})"
        )
