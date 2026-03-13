"""ReagentAI Paper Chat Interface Backend -- Phase 12.

Provides a session-aware conversational interface for interacting with
research papers.  Uses the :class:`~backend.agents.chat_agent.ChatAgent`
combined with the :class:`~backend.retrieval.rag_pipeline.RAGPipeline`
to answer user questions grounded in the paper text and generated code.

Core features:

* **Session management** -- Each paper has its own chat session with
  persistent conversation history.
* **RAG-augmented answers** -- Every user question triggers a semantic
  retrieval pass to inject the most relevant paper chunks and code
  snippets into the LLM prompt.
* **Source attribution** -- Answers include the retrieved chunks that
  informed them, so users can verify claims.
* **Follow-up suggestions** -- Each answer comes with suggested follow-up
  questions to guide exploration.

Supported query types include:

* ``"Explain the algorithm"``
* ``"What dataset is used?"``
* ``"How does the architecture work?"``
* ``"What loss function is used?"``
* ``"Compare this approach to [X]"``

Usage::

    from backend.orchestration.chat_handler import ChatHandler

    handler = ChatHandler()
    handler.register_paper(
        paper_id="arxiv:2301.00001",
        paper_text="We propose ...",
        research_summary={...},
        source_files={"src/model.py": "...", ...},
    )
    response = await handler.ask(
        paper_id="arxiv:2301.00001",
        question="What optimizer is used?",
    )
    print(response.answer)
    print(response.sources)
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.agents import ChatAgent
from backend.config.settings import settings
from backend.retrieval.rag_pipeline import RAGPipeline
from backend.utils.logging import get_logger

logger = get_logger("orchestration.chat_handler")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of conversation turns to keep in history
_MAX_HISTORY_TURNS = 20

# Maximum number of RAG chunks to retrieve per question
_DEFAULT_TOP_K = 5

# Maximum length of context passed to the agent
_MAX_CONTEXT_CHARS = 8000


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """A single message in a conversation.

    Attributes:
        role:      ``"user"`` or ``"assistant"``.
        content:   The message text.
        timestamp: Unix epoch timestamp when the message was created.
    """

    role: str
    content: str
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a dictionary compatible with ChatAgent history."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class ChatSession:
    """Conversation session for a specific paper.

    Attributes:
        session_id:       Unique session identifier.
        paper_id:         The paper this session is associated with.
        history:          Ordered list of messages in the conversation.
        created_at:       Unix epoch timestamp when the session was created.
        last_activity_at: Timestamp of the most recent message.
    """

    session_id: str
    paper_id: str
    history: List[ChatMessage] = field(default_factory=list)
    created_at: float = 0.0
    last_activity_at: float = 0.0

    def __post_init__(self) -> None:
        now = time.time()
        if self.created_at == 0.0:
            self.created_at = now
        if self.last_activity_at == 0.0:
            self.last_activity_at = now

    def add_message(self, role: str, content: str) -> ChatMessage:
        """Append a message to the conversation history.

        Automatically trims history to the last ``_MAX_HISTORY_TURNS``
        turns to prevent unbounded growth.

        Args:
            role:    ``"user"`` or ``"assistant"``.
            content: The message text.

        Returns:
            The newly created :class:`ChatMessage`.
        """
        msg = ChatMessage(role=role, content=content)
        self.history.append(msg)
        self.last_activity_at = msg.timestamp

        # Trim history to prevent unbounded growth
        if len(self.history) > _MAX_HISTORY_TURNS * 2:
            self.history = self.history[-_MAX_HISTORY_TURNS * 2:]

        return msg

    def get_history_dicts(self, max_turns: int | None = None) -> List[Dict[str, str]]:
        """Return conversation history as a list of dicts.

        Args:
            max_turns: Maximum number of recent turns to return.
                       Defaults to ``_MAX_HISTORY_TURNS``.

        Returns:
            List of ``{"role": ..., "content": ...}`` dictionaries.
        """
        turns = max_turns or _MAX_HISTORY_TURNS
        recent = self.history[-turns * 2:]  # Each turn has user + assistant
        return [m.to_dict() for m in recent]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the session metadata (without full history)."""
        return {
            "session_id": self.session_id,
            "paper_id": self.paper_id,
            "message_count": len(self.history),
            "created_at": self.created_at,
            "last_activity_at": self.last_activity_at,
        }


@dataclass
class ChatResponse:
    """Response from the chat handler.

    Attributes:
        answer:               The assistant's answer text.
        sources:              List of source references (paper chunks, code
                              files) that informed the answer.
        follow_up_suggestions: Suggested follow-up questions.
        session_id:           The session ID for continuing the conversation.
        elapsed:              Wall-clock time in seconds.
    """

    answer: str
    sources: List[str] = field(default_factory=list)
    follow_up_suggestions: List[str] = field(default_factory=list)
    session_id: str = ""
    elapsed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "follow_up_suggestions": self.follow_up_suggestions,
            "session_id": self.session_id,
            "elapsed": round(self.elapsed, 3),
        }


# ---------------------------------------------------------------------------
# Paper context container
# ---------------------------------------------------------------------------

@dataclass
class PaperContext:
    """Stored context for a registered paper.

    Attributes:
        paper_id:         Unique paper identifier.
        paper_text:       Full raw text of the paper.
        research_summary: Structured output from ResearchAgent.
        source_files:     Generated source code files.
    """

    paper_id: str
    paper_text: str = ""
    research_summary: Dict[str, Any] = field(default_factory=dict)
    source_files: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main chat handler
# ---------------------------------------------------------------------------

class ChatHandler:
    """Session-aware chat handler for paper Q&A.

    The handler manages multiple concurrent chat sessions (one per paper)
    and uses RAG-augmented prompts to answer user questions with source
    attribution.

    Parameters
    ----------
    rag_pipeline : RAGPipeline | None
        Pre-configured RAG pipeline.  If *None*, a new one is created.
    top_k : int
        Number of RAG chunks to retrieve per question.
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline | None = None,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        self._agent = ChatAgent()
        self._rag = rag_pipeline or RAGPipeline()
        self._top_k = top_k

        # Session storage: paper_id -> {session_id -> ChatSession}
        self._sessions: Dict[str, Dict[str, ChatSession]] = defaultdict(dict)

        # Paper context storage: paper_id -> PaperContext
        self._papers: Dict[str, PaperContext] = {}

        logger.info(
            "ChatHandler initialised (top_k={})",
            self._top_k,
        )

    # ------------------------------------------------------------------
    # Paper registration
    # ------------------------------------------------------------------

    def register_paper(
        self,
        paper_id: str,
        paper_text: str = "",
        research_summary: Dict[str, Any] | None = None,
        source_files: Dict[str, str] | None = None,
        force_reindex: bool = False,
    ) -> None:
        """Register a paper for chat interaction.

        Indexes the paper text into the RAG pipeline and stores the
        associated metadata for context building.

        Args:
            paper_id:         Unique paper identifier (e.g. arXiv ID).
            paper_text:       Full raw text of the paper.
            research_summary: Structured output from ResearchAgent.
            source_files:     Generated source code files from CodingAgent.
            force_reindex:    If *True*, re-index even if already indexed.
        """
        logger.info("Registering paper '{}' for chat", paper_id)

        # Store paper context
        self._papers[paper_id] = PaperContext(
            paper_id=paper_id,
            paper_text=paper_text,
            research_summary=research_summary or {},
            source_files=source_files or {},
        )

        # Index paper text in RAG pipeline
        if paper_text:
            try:
                num_chunks = self._rag.index_paper(
                    paper_text=paper_text,
                    paper_id=paper_id,
                    force=force_reindex,
                )
                logger.info(
                    "Paper '{}' indexed: {} chunks",
                    paper_id,
                    num_chunks,
                )
            except Exception as exc:
                logger.error(
                    "Failed to index paper '{}': {}",
                    paper_id,
                    exc,
                )

    def is_paper_registered(self, paper_id: str) -> bool:
        """Check whether a paper has been registered.

        Args:
            paper_id: Paper identifier.

        Returns:
            True if the paper is registered.
        """
        return paper_id in self._papers

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(self, paper_id: str) -> ChatSession:
        """Create a new chat session for a paper.

        Args:
            paper_id: Paper identifier.

        Returns:
            The newly created :class:`ChatSession`.

        Raises:
            ValueError: If the paper has not been registered.
        """
        if paper_id not in self._papers:
            raise ValueError(
                f"Paper '{paper_id}' is not registered. "
                f"Call register_paper() first."
            )

        session_id = str(uuid.uuid4())
        session = ChatSession(
            session_id=session_id,
            paper_id=paper_id,
        )
        self._sessions[paper_id][session_id] = session

        logger.info(
            "Created chat session '{}' for paper '{}'",
            session_id,
            paper_id,
        )
        return session

    def get_session(
        self,
        paper_id: str,
        session_id: str | None = None,
    ) -> ChatSession:
        """Retrieve an existing session or create a new one.

        If *session_id* is provided and exists, return it.  Otherwise
        create a new session.

        Args:
            paper_id:   Paper identifier.
            session_id: Optional existing session ID.

        Returns:
            A :class:`ChatSession` instance.
        """
        if session_id:
            paper_sessions = self._sessions.get(paper_id, {})
            session = paper_sessions.get(session_id)
            if session is not None:
                return session
            logger.warning(
                "Session '{}' not found for paper '{}'; creating new session",
                session_id,
                paper_id,
            )

        return self.create_session(paper_id)

    def list_sessions(self, paper_id: str) -> List[Dict[str, Any]]:
        """List all chat sessions for a paper.

        Args:
            paper_id: Paper identifier.

        Returns:
            List of session metadata dicts.
        """
        paper_sessions = self._sessions.get(paper_id, {})
        return [s.to_dict() for s in paper_sessions.values()]

    def delete_session(self, paper_id: str, session_id: str) -> bool:
        """Delete a chat session.

        Args:
            paper_id:   Paper identifier.
            session_id: Session identifier.

        Returns:
            True if the session was found and deleted.
        """
        paper_sessions = self._sessions.get(paper_id, {})
        if session_id in paper_sessions:
            del paper_sessions[session_id]
            logger.info(
                "Deleted session '{}' for paper '{}'",
                session_id,
                paper_id,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Main ask method
    # ------------------------------------------------------------------

    async def ask(
        self,
        paper_id: str,
        question: str,
        session_id: str | None = None,
    ) -> ChatResponse:
        """Ask a question about a registered paper.

        This is the primary public method.  It:

        1. Retrieves (or creates) a chat session.
        2. Performs a RAG retrieval pass to find relevant chunks.
        3. Builds the full context (RAG chunks + paper summary + code).
        4. Calls the ChatAgent with the question and history.
        5. Records the exchange in the session history.
        6. Returns a structured :class:`ChatResponse`.

        Args:
            paper_id:   Identifier of the registered paper.
            question:   The user's natural-language question.
            session_id: Optional session ID for continuing an existing
                        conversation.  If not provided, a new session
                        is created.

        Returns:
            A :class:`ChatResponse` containing the answer, sources,
            suggestions, and session ID.

        Raises:
            ValueError: If the paper is not registered.
        """
        if not question or not question.strip():
            return ChatResponse(
                answer="Please provide a question.",
                session_id=session_id or "",
            )

        if paper_id not in self._papers:
            raise ValueError(
                f"Paper '{paper_id}' is not registered. "
                f"Call register_paper() first."
            )

        t0 = time.monotonic()

        logger.info(
            "Processing question for paper '{}': {}",
            paper_id,
            question[:100],
        )

        # Step 1: Get or create session
        session = self.get_session(paper_id, session_id)

        # Step 2: Record the user question
        session.add_message("user", question)

        # Step 3: Retrieve relevant chunks via RAG
        rag_chunks = self._retrieve_chunks(paper_id, question)

        # Step 4: Build agent context
        paper_ctx = self._papers[paper_id]
        context = self._build_agent_context(
            question=question,
            rag_chunks=rag_chunks,
            paper_ctx=paper_ctx,
            session=session,
        )

        # Step 5: Call the ChatAgent
        try:
            agent_result = await self._agent.run(context)
        except Exception as exc:
            logger.error("ChatAgent failed: {}", exc)
            error_response = ChatResponse(
                answer=(
                    "I apologize, but I encountered an error while "
                    "processing your question. Please try again."
                ),
                session_id=session.session_id,
                elapsed=time.monotonic() - t0,
            )
            session.add_message("assistant", error_response.answer)
            return error_response

        # Step 6: Extract the answer and metadata
        answer = agent_result.get("answer", "I could not generate an answer.")
        sources = agent_result.get("sources", [])
        follow_ups = agent_result.get("follow_up_suggestions", [])

        # Step 7: Record the assistant response
        session.add_message("assistant", answer)

        elapsed = time.monotonic() - t0

        logger.info(
            "Question answered for paper '{}' in {:.2f}s "
            "(session='{}', sources={}, history={})",
            paper_id,
            elapsed,
            session.session_id,
            len(sources),
            len(session.history),
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            follow_up_suggestions=follow_ups,
            session_id=session.session_id,
            elapsed=elapsed,
        )

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------

    def _retrieve_chunks(
        self,
        paper_id: str,
        question: str,
    ) -> List[str]:
        """Retrieve relevant text chunks for a question using the RAG pipeline.

        Args:
            paper_id: Paper identifier (used to filter results).
            question: The user's question.

        Returns:
            A list of chunk text strings ranked by relevance.
        """
        try:
            results = self._rag.query_paper(
                question=question,
                top_k=self._top_k,
                paper_id=paper_id,
            )
            chunks = [r.get("text", "") for r in results if r.get("text")]

            logger.debug(
                "RAG retrieval for '{}': {} chunks retrieved",
                question[:60],
                len(chunks),
            )
            return chunks

        except Exception as exc:
            logger.warning("RAG retrieval failed: {}", exc)
            return []

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_agent_context(
        self,
        question: str,
        rag_chunks: List[str],
        paper_ctx: PaperContext,
        session: ChatSession,
    ) -> Dict[str, Any]:
        """Build the context dictionary expected by the ChatAgent.

        Assembles the question, RAG chunks, paper metadata, source files,
        and conversation history into a single dict.

        Args:
            question:   The user's question.
            rag_chunks: Retrieved paper chunks.
            paper_ctx:  Registered paper context.
            session:    Active chat session.

        Returns:
            A context dictionary suitable for ``ChatAgent.run()``.
        """
        context: Dict[str, Any] = {
            "question": question,
            "rag_chunks": rag_chunks,
            "research_summary": paper_ctx.research_summary,
            "files": paper_ctx.source_files,
            "history": session.get_history_dicts(),
        }

        return context

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def ask_without_session(
        self,
        paper_id: str,
        question: str,
        history: List[Dict[str, str]] | None = None,
    ) -> ChatResponse:
        """One-shot question without explicit session management.

        This creates a temporary session, asks the question, and returns
        the response.  The session is not persisted.

        Args:
            paper_id: Paper identifier.
            question: User question.
            history:  Optional conversation history (list of
                      ``{"role": ..., "content": ...}`` dicts).

        Returns:
            A :class:`ChatResponse`.
        """
        if paper_id not in self._papers:
            raise ValueError(
                f"Paper '{paper_id}' is not registered. "
                f"Call register_paper() first."
            )

        t0 = time.monotonic()

        # Build a temporary session with pre-loaded history
        session = ChatSession(
            session_id=str(uuid.uuid4()),
            paper_id=paper_id,
        )
        if history:
            for msg in history:
                session.add_message(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                )

        # Retrieve RAG chunks
        rag_chunks = self._retrieve_chunks(paper_id, question)

        # Build context
        session.add_message("user", question)
        paper_ctx = self._papers[paper_id]
        context = self._build_agent_context(
            question=question,
            rag_chunks=rag_chunks,
            paper_ctx=paper_ctx,
            session=session,
        )

        # Call agent
        try:
            agent_result = await self._agent.run(context)
        except Exception as exc:
            logger.error("ChatAgent failed (one-shot): {}", exc)
            return ChatResponse(
                answer="An error occurred while processing your question.",
                session_id=session.session_id,
                elapsed=time.monotonic() - t0,
            )

        elapsed = time.monotonic() - t0

        return ChatResponse(
            answer=agent_result.get("answer", ""),
            sources=agent_result.get("sources", []),
            follow_up_suggestions=agent_result.get("follow_up_suggestions", []),
            session_id=session.session_id,
            elapsed=elapsed,
        )

    def get_session_history(
        self,
        paper_id: str,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve the full conversation history for a session.

        Args:
            paper_id:   Paper identifier.
            session_id: Session identifier.

        Returns:
            List of message dicts with ``role``, ``content``, and
            ``timestamp`` keys.

        Raises:
            ValueError: If the session is not found.
        """
        paper_sessions = self._sessions.get(paper_id, {})
        session = paper_sessions.get(session_id)
        if session is None:
            raise ValueError(
                f"Session '{session_id}' not found for paper '{paper_id}'."
            )
        return [m.to_dict() for m in session.history]

    def clear_session_history(
        self,
        paper_id: str,
        session_id: str,
    ) -> bool:
        """Clear conversation history for a session while keeping the session.

        Args:
            paper_id:   Paper identifier.
            session_id: Session identifier.

        Returns:
            True if the session was found and cleared.
        """
        paper_sessions = self._sessions.get(paper_id, {})
        session = paper_sessions.get(session_id)
        if session is None:
            return False

        session.history.clear()
        session.last_activity_at = time.time()
        logger.info(
            "Cleared history for session '{}' (paper '{}')",
            session_id,
            paper_id,
        )
        return True

    @property
    def registered_papers(self) -> List[str]:
        """Return a list of all registered paper IDs."""
        return list(self._papers.keys())

    @property
    def active_session_count(self) -> int:
        """Return the total number of active sessions across all papers."""
        return sum(
            len(sessions)
            for sessions in self._sessions.values()
        )
