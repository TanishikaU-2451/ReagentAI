"""ReagentAI Orchestration Package -- Phases 6, 8-12, 15-16.

This package contains the high-level orchestration modules that coordinate
the multi-agent pipeline for code generation, debugging, validation,
diagram generation, reproducibility scoring, and conversational Q&A.

Modules:
    pipeline           : Master pipeline orchestrating all stages.
    code_generator     : Phase 6  -- Orchestrates the CodingAgent to produce a
                         complete project from research, plan, and architecture
                         outputs.
    debug_loop         : Phase 8  -- Implements the generate-execute-debug retry
                         loop that iteratively fixes runtime errors.
    validator          : Phase 9  -- Implementation validation pipeline.
    diagram_generator  : Phase 10 -- Mermaid.js architecture diagram generation.
    reproducibility    : Phase 11 -- Reproducibility scoring and assessment.
    chat_handler       : Phase 12 -- Paper chat interface backend (RAG-powered).
    project_packager   : Phase 16 -- Downloadable project packaging.
"""

from backend.orchestration.code_generator import CodeGenerationPipeline
from backend.orchestration.debug_loop import DebugLoop, DebugLoopResult
from backend.orchestration.validator import ValidationPipeline
from backend.orchestration.diagram_generator import DiagramGenerator
from backend.orchestration.reproducibility import ReproducibilityScorer, ReproducibilityReport
from backend.orchestration.chat_handler import ChatHandler, ChatResponse
from backend.orchestration.project_packager import ProjectPackager

__all__ = [
    "CodeGenerationPipeline",
    "DebugLoop",
    "DebugLoopResult",
    "ValidationPipeline",
    "DiagramGenerator",
    "ReproducibilityScorer",
    "ReproducibilityReport",
    "ChatHandler",
    "ChatResponse",
    "ProjectPackager",
]
