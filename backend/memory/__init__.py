"""ReagentAI Agent Memory System -- Phase 15.

Provides persistent, vector-indexed memory for every agent in the
ReagentAI pipeline.  Agents can :meth:`store` observations during a run
and later :meth:`recall` the most semantically relevant past context,
enabling cross-session learning and richer reasoning.

Memory types
------------
- **research_memory** -- facts, paper summaries, and literature context
  retained by the ResearchAgent.
- **architecture_memory** -- past design decisions, directory layouts,
  and technology-stack choices stored by the ArchitectureAgent.
- **debug_memory** -- error signatures, stack traces, and successful
  fixes accumulated by the DebugAgent across retry loops.

Quick start::

    from backend.memory import AgentMemoryStore, MemoryEntry

    store = AgentMemoryStore()

    # Persist a new memory
    store.store(
        agent_name="ResearchAgent",
        content="The paper proposes a transformer-based architecture ...",
        memory_type="research_memory",
        metadata={"paper_id": "arxiv:2401.12345"},
    )

    # Recall related memories
    results: list[MemoryEntry] = store.recall(
        query="transformer architecture design",
        memory_type="research_memory",
        top_k=5,
    )
"""

from backend.memory.agent_memory import AgentMemoryStore, MemoryEntry

__all__ = [
    "AgentMemoryStore",
    "MemoryEntry",
]
