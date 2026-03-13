"""ReagentAI Multi-Agent System.

This package contains all agents used in the ReagentAI paper-to-code
pipeline. Each agent inherits from :class:`BaseAgent` and implements
a single ``run(context) -> dict`` method.

Pipeline order::

    1. ResearchAgent     -- Extract structured info from the paper
    2. PlanningAgent     -- Convert research summary into implementation plan
    3. ArchitectureAgent -- Design the project directory structure
    4. CodingAgent       -- Generate all implementation files
    5. TestingAgent      -- Generate pytest test suites
    6. ValidationAgent   -- Run lightweight validation checks
    7. DebugAgent        -- Diagnose and fix errors (used in retry loop)
    8. DiagramAgent      -- Generate Mermaid.js architecture diagrams
    9. ChatAgent         -- Answer user questions (RAG-powered)

Usage::

    from backend.agents import ResearchAgent, PlanningAgent

    agent = ResearchAgent()
    result = await agent.run({"paper_text": "..."})
"""

from backend.agents.base_agent import BaseAgent
from backend.agents.research_agent import ResearchAgent
from backend.agents.planning_agent import PlanningAgent
from backend.agents.architecture_agent import ArchitectureAgent
from backend.agents.coding_agent import CodingAgent
from backend.agents.testing_agent import TestingAgent
from backend.agents.debug_agent import DebugAgent
from backend.agents.validation_agent import ValidationAgent
from backend.agents.diagram_agent import DiagramAgent
from backend.agents.chat_agent import ChatAgent

__all__ = [
    "BaseAgent",
    "ResearchAgent",
    "PlanningAgent",
    "ArchitectureAgent",
    "CodingAgent",
    "TestingAgent",
    "DebugAgent",
    "ValidationAgent",
    "DiagramAgent",
    "ChatAgent",
]
