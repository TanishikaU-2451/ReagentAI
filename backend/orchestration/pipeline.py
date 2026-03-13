"""ReagentAI Master Orchestration Pipeline -- Phase 13.

Orchestrates the complete paper-to-code workflow as a single linear
pipeline with progress reporting at every stage.  The pipeline connects
all subsystems -- parsing, retrieval, agents, code generation, execution,
debugging, validation, diagram generation, reproducibility scoring, and
project packaging -- into a cohesive, fault-tolerant workflow.

Pipeline stages::

    1.  PDF Parsing          -- Extract text, metadata, sections, and equations.
    2.  Section Extraction   -- Detect paper sections (abstract, methodology, ...).
    3.  Equation Extraction  -- Extract LaTeX equations from the paper text.
    4.  RAG Indexing          -- Chunk and embed the paper into the vector store.
    5.  Research Agent        -- Extract structured research summary via LLM.
    6.  Planning Agent        -- Convert research summary into implementation plan.
    7.  Architecture Agent    -- Design project directory layout and modules.
    8.  GitHub Retrieval      -- Search GitHub for reference repositories.
    9.  HuggingFace Retrieval -- Search HuggingFace for pre-trained models.
    10. Code Generation       -- Generate all project files via CodingAgent.
    11. Execution             -- Run the generated project in a sandbox.
    12. Self-Debug Loop       -- Iteratively fix runtime errors.
    13. Validation            -- Run validation checks (static + runtime).
    14. Diagram Generation    -- Generate Mermaid.js architecture diagrams.
    15. Reproducibility Score -- Compute reproducibility assessment.
    16. Project Packaging     -- Package everything into a downloadable zip.

Usage::

    from backend.orchestration.pipeline import ReagentPipeline

    pipeline = ReagentPipeline()
    result = await pipeline.run(
        paper_path="path/to/paper.pdf",
        project_id="abc123",
        progress_callback=my_callback,
    )

    print(result.status)             # "completed" or "failed"
    print(result.reproducibility)    # ReproducibilityReport
    print(result.package_path)       # Path to the generated zip file
"""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from backend.agents import (
    ArchitectureAgent,
    PlanningAgent,
    ResearchAgent,
    TestingAgent,
)
from backend.config.settings import settings
from backend.execution import CodeRunner
from backend.memory import AgentMemoryStore
from backend.orchestration.chat_handler import ChatHandler
from backend.orchestration.code_generator import CodeGenerationPipeline
from backend.orchestration.debug_loop import DebugLoop
from backend.orchestration.diagram_generator import DiagramGenerator
from backend.orchestration.reproducibility import ReproducibilityScorer
from backend.orchestration.validator import ValidationPipeline
from backend.parser import parse_paper
from backend.retrieval import (
    CodeExtractor,
    GitHubSearcher,
    HuggingFaceRetriever,
    RAGPipeline,
    RepoRanker,
)
from backend.utils.logging import get_logger

logger = get_logger("orchestration.pipeline")

# ---------------------------------------------------------------------------
# Progress callback type alias
# ---------------------------------------------------------------------------

ProgressCallback = Optional[
    Callable[[str, str, str], Coroutine[Any, Any, None] | None]
]

# ---------------------------------------------------------------------------
# Stage status constants
# ---------------------------------------------------------------------------


class StageStatus:
    """Status constants for pipeline stages."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """Result of a single pipeline stage.

    Attributes:
        stage_name:      Human-readable stage name.
        status:          One of StageStatus constants.
        detail:          Descriptive message about the outcome.
        data:            Arbitrary result data from the stage.
        elapsed_seconds: Wall-clock time in seconds for this stage.
        error:           Error message if the stage failed.
    """

    stage_name: str
    status: str = StageStatus.PENDING
    detail: str = ""
    data: Any = None
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (excluding large data payloads)."""
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "detail": self.detail,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "error": self.error,
        }


@dataclass
class PipelineResult:
    """Complete result of the ReagentAI pipeline.

    Attributes:
        project_id:        Unique project identifier.
        status:            ``"completed"``, ``"failed"``, or ``"partial"``.
        stages:            Ordered list of stage results.
        parsed_paper:      Structured parsing output.
        research_summary:  Output of the ResearchAgent.
        plan:              Output of the PlanningAgent.
        architecture:      Output of the ArchitectureAgent.
        github_repos:      Retrieved GitHub repositories.
        huggingface_models: Retrieved HuggingFace models.
        reference_code:    Extracted reference code from GitHub.
        generated_code:    Dict of generated source files.
        debug_result:      Debug loop outcome.
        validation_report: Validation pipeline report.
        diagrams:          Generated Mermaid.js diagrams.
        reproducibility:   Reproducibility score report.
        package_path:      Path to the packaged zip archive.
        total_elapsed:     Total wall-clock time in seconds.
        error:             Top-level error message (if applicable).
    """

    project_id: str = ""
    status: str = "pending"
    stages: List[StageResult] = field(default_factory=list)
    parsed_paper: Any = None
    research_summary: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    architecture: Dict[str, Any] = field(default_factory=dict)
    github_repos: List[Dict[str, Any]] = field(default_factory=list)
    huggingface_models: List[Dict[str, Any]] = field(default_factory=list)
    reference_code: Dict[str, str] = field(default_factory=dict)
    generated_code: Dict[str, str] = field(default_factory=dict)
    debug_result: Any = None
    validation_report: Dict[str, Any] = field(default_factory=dict)
    diagrams: Dict[str, Any] = field(default_factory=dict)
    reproducibility: Any = None
    package_path: Optional[str] = None
    total_elapsed: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns:
            A nested dictionary suitable for JSON serialisation.  Large
            payloads (generated code, reference code) are summarised
            rather than included verbatim.
        """
        return {
            "project_id": self.project_id,
            "status": self.status,
            "stages": [s.to_dict() for s in self.stages],
            "research_summary": self.research_summary,
            "plan": self.plan,
            "architecture": self.architecture,
            "github_repos_count": len(self.github_repos),
            "huggingface_models_count": len(self.huggingface_models),
            "reference_code_files": list(self.reference_code.keys()),
            "generated_code_files": list(self.generated_code.keys()),
            "debug_result": (
                self.debug_result.to_dict()
                if self.debug_result and hasattr(self.debug_result, "to_dict")
                else None
            ),
            "validation_report": self.validation_report,
            "diagrams": self.diagrams,
            "reproducibility": (
                self.reproducibility.to_dict()
                if self.reproducibility and hasattr(self.reproducibility, "to_dict")
                else None
            ),
            "package_path": self.package_path,
            "total_elapsed": round(self.total_elapsed, 3),
            "error": self.error,
        }

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary suitable for status polling.

        Returns:
            A small dictionary with key progress indicators.
        """
        completed = sum(
            1 for s in self.stages if s.status == StageStatus.COMPLETED
        )
        failed = sum(
            1 for s in self.stages if s.status == StageStatus.FAILED
        )
        return {
            "project_id": self.project_id,
            "status": self.status,
            "stages_total": len(self.stages),
            "stages_completed": completed,
            "stages_failed": failed,
            "total_elapsed": round(self.total_elapsed, 3),
            "package_path": self.package_path,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------


class ReagentPipeline:
    """Master orchestration pipeline for ReagentAI.

    Connects every subsystem in the paper-to-code workflow and exposes a
    single :meth:`run` method that drives the entire process.

    The pipeline is designed to be fault-tolerant: if a non-critical stage
    fails (e.g. GitHub retrieval, diagram generation), execution continues
    with the remaining stages rather than aborting the entire run.

    Parameters
    ----------
    rag_pipeline : RAGPipeline | None
        Pre-configured RAG pipeline.  Created automatically if *None*.
    memory_store : AgentMemoryStore | None
        Pre-configured agent memory store.  Created automatically if *None*.
    """

    # Stages that are critical -- pipeline aborts if they fail
    _CRITICAL_STAGES = frozenset({
        "pdf_parsing",
        "research_agent",
        "planning_agent",
        "architecture_agent",
        "code_generation",
    })

    def __init__(
        self,
        *,
        rag_pipeline: Optional[RAGPipeline] = None,
        memory_store: Optional[AgentMemoryStore] = None,
    ) -> None:
        """Initialise the pipeline and all its subsystem components."""
        logger.info("Initialising ReagentPipeline ...")

        # Core subsystems
        self._rag = rag_pipeline or RAGPipeline()
        self._memory = memory_store or AgentMemoryStore()

        # Agents
        self._research_agent = ResearchAgent()
        self._planning_agent = PlanningAgent()
        self._architecture_agent = ArchitectureAgent()
        self._testing_agent = TestingAgent()

        # Retrieval
        self._github_searcher = GitHubSearcher()
        self._repo_ranker = RepoRanker()
        self._code_extractor = CodeExtractor()
        self._hf_retriever = HuggingFaceRetriever()

        # Orchestration sub-pipelines
        self._code_gen = CodeGenerationPipeline()
        self._debug_loop = DebugLoop()
        self._validator = ValidationPipeline()
        self._diagram_gen = DiagramGenerator()
        self._reproducibility = ReproducibilityScorer()
        self._chat_handler = ChatHandler(rag_pipeline=self._rag)

        # Paths
        self._projects_dir = Path(settings.generated_projects_path).resolve()
        self._projects_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ReagentPipeline initialised (projects_dir={})", self._projects_dir
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        paper_path: str,
        project_id: Optional[str] = None,
        progress_callback: ProgressCallback = None,
    ) -> PipelineResult:
        """Run the full paper-to-code pipeline.

        Executes every stage in sequence, reporting progress through the
        optional *progress_callback*.  Non-critical stage failures are
        recorded but do not abort the pipeline.

        Args:
            paper_path:        Path to the uploaded PDF paper.
            project_id:        Optional project identifier.  A UUID is
                               generated if omitted.
            progress_callback: Optional async or sync callable invoked at
                               each stage transition with
                               ``(stage_name, status, detail)``.

        Returns:
            A :class:`PipelineResult` containing all outputs and metadata.
        """
        pipeline_start = time.monotonic()
        project_id = project_id or uuid.uuid4().hex[:12]

        logger.info(
            "=== ReagentPipeline starting (project_id={}, paper={}) ===",
            project_id,
            paper_path,
        )

        result = PipelineResult(project_id=project_id)

        # Helper: notify progress
        async def _notify(stage: str, status: str, detail: str = "") -> None:
            """Fire the progress callback, handling both sync and async."""
            stage_result = _find_or_create_stage(result, stage)
            stage_result.status = status
            stage_result.detail = detail
            if progress_callback is not None:
                try:
                    ret = progress_callback(stage, status, detail)
                    if asyncio.iscoroutine(ret):
                        await ret
                except Exception as exc:
                    logger.warning(
                        "Progress callback error: {}", exc
                    )

        # Helper: run a pipeline stage with error handling
        async def _run_stage(
            stage_name: str,
            coro: Coroutine[Any, Any, Any],
        ) -> Any:
            """Execute a stage coroutine with timing and error handling.

            Args:
                stage_name: Identifier for the stage.
                coro:       The async callable to execute.

            Returns:
                The stage result data, or *None* on failure.

            Raises:
                PipelineAbortError: If a critical stage fails.
            """
            stage_result = _find_or_create_stage(result, stage_name)
            await _notify(stage_name, StageStatus.RUNNING)
            t0 = time.monotonic()

            try:
                data = await coro
                elapsed = time.monotonic() - t0
                stage_result.status = StageStatus.COMPLETED
                stage_result.data = data
                stage_result.elapsed_seconds = elapsed
                stage_result.detail = f"Completed in {elapsed:.1f}s"
                await _notify(
                    stage_name,
                    StageStatus.COMPLETED,
                    stage_result.detail,
                )
                logger.info(
                    "Stage '{}' completed in {:.2f}s", stage_name, elapsed
                )
                return data

            except Exception as exc:
                elapsed = time.monotonic() - t0
                error_msg = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc()
                stage_result.status = StageStatus.FAILED
                stage_result.elapsed_seconds = elapsed
                stage_result.error = error_msg
                stage_result.detail = f"Failed: {error_msg}"
                await _notify(
                    stage_name, StageStatus.FAILED, error_msg
                )
                logger.error(
                    "Stage '{}' failed after {:.2f}s: {}\n{}",
                    stage_name,
                    elapsed,
                    error_msg,
                    tb,
                )

                if stage_name in self._CRITICAL_STAGES:
                    result.status = "failed"
                    result.error = (
                        f"Critical stage '{stage_name}' failed: {error_msg}"
                    )
                    result.total_elapsed = time.monotonic() - pipeline_start
                    raise _PipelineAbortError(stage_name, error_msg) from exc

                return None

        try:
            # ---- Stage 1: PDF Parsing ----
            parsed_paper = await _run_stage(
                "pdf_parsing",
                self._stage_parse_paper(paper_path),
            )
            result.parsed_paper = parsed_paper

            paper_text = ""
            paper_title = ""
            equations = []
            if parsed_paper is not None:
                paper_text = parsed_paper.document.full_text
                paper_title = parsed_paper.document.metadata.title or ""
                equations = [
                    eq.latex for eq in parsed_paper.equations.equations
                ]

            # ---- Stage 2: Section Extraction ----
            sections_data = await _run_stage(
                "section_extraction",
                self._stage_extract_sections(parsed_paper),
            )

            # ---- Stage 3: Equation Extraction ----
            equations_data = await _run_stage(
                "equation_extraction",
                self._stage_extract_equations(parsed_paper),
            )

            # ---- Stage 4: RAG Indexing ----
            await _run_stage(
                "rag_indexing",
                self._stage_rag_indexing(paper_text, project_id),
            )

            # ---- Stage 5: Research Agent ----
            research_summary = await _run_stage(
                "research_agent",
                self._stage_research_agent(paper_text, paper_title),
            )
            result.research_summary = research_summary or {}

            # Store research findings in memory
            if research_summary:
                self._memory.store(
                    agent_name="ResearchAgent",
                    content=str(research_summary.get("summary", "")),
                    memory_type="research_memory",
                    metadata={"project_id": project_id},
                )

            # ---- Stage 6: Planning Agent ----
            plan = await _run_stage(
                "planning_agent",
                self._stage_planning_agent(result.research_summary),
            )
            result.plan = plan or {}

            # ---- Stage 7: Architecture Agent ----
            architecture = await _run_stage(
                "architecture_agent",
                self._stage_architecture_agent(
                    result.research_summary, result.plan
                ),
            )
            result.architecture = architecture or {}

            # Store architecture decisions in memory
            if architecture:
                self._memory.store(
                    agent_name="ArchitectureAgent",
                    content=str(architecture),
                    memory_type="architecture_memory",
                    metadata={"project_id": project_id},
                )

            # ---- Stage 8: GitHub Retrieval ----
            github_data = await _run_stage(
                "github_retrieval",
                self._stage_github_retrieval(
                    result.research_summary, paper_text
                ),
            )
            if github_data:
                result.github_repos = github_data.get("repos", [])
                result.reference_code = github_data.get("reference_code", {})

            # ---- Stage 9: HuggingFace Retrieval ----
            hf_models = await _run_stage(
                "huggingface_retrieval",
                self._stage_huggingface_retrieval(
                    result.research_summary, paper_text
                ),
            )
            result.huggingface_models = hf_models or []

            # ---- Stage 10: Code Generation ----
            code_gen_result = await _run_stage(
                "code_generation",
                self._stage_code_generation(
                    project_id,
                    result.plan,
                    result.architecture,
                    result.research_summary,
                    result.reference_code,
                ),
            )
            if code_gen_result and code_gen_result.success:
                result.generated_code = code_gen_result.files

            # ---- Stage 10b: Test Generation ----
            test_result = await _run_stage(
                "test_generation",
                self._stage_test_generation(
                    result.generated_code,
                    result.research_summary,
                    result.architecture,
                ),
            )
            if test_result and test_result.get("test_files"):
                # Merge test files into generated code
                result.generated_code.update(test_result["test_files"])
                # Also write test files to disk
                if code_gen_result and code_gen_result.project_dir:
                    await self._code_gen.regenerate_files(
                        code_gen_result.project_dir,
                        test_result["test_files"],
                    )

            # ---- Stage 11: Execution & Self-Debug Loop ----
            debug_result = await _run_stage(
                "self_debug_loop",
                self._stage_debug_loop(
                    project_id,
                    code_gen_result,
                    result.research_summary,
                ),
            )
            result.debug_result = debug_result
            if debug_result and debug_result.final_code:
                result.generated_code = debug_result.final_code

            # ---- Stage 12: Validation ----
            validation_report = await _run_stage(
                "validation",
                self._stage_validation(
                    result.generated_code,
                    result.research_summary,
                    project_id,
                ),
            )
            result.validation_report = validation_report or {}

            # ---- Stage 13: Diagram Generation ----
            diagrams = await _run_stage(
                "diagram_generation",
                self._stage_diagram_generation(
                    result.research_summary,
                    result.generated_code,
                    result.architecture,
                ),
            )
            result.diagrams = diagrams or {}

            # ---- Stage 14: Reproducibility Score ----
            repro_report = await _run_stage(
                "reproducibility_score",
                self._stage_reproducibility(
                    result.research_summary, paper_text
                ),
            )
            result.reproducibility = repro_report

            # ---- Stage 15: Project Packaging ----
            package_path = await _run_stage(
                "project_packaging",
                self._stage_package_project(
                    project_id,
                    result,
                ),
            )
            result.package_path = package_path

            # ---- Register paper for chat ----
            try:
                self._chat_handler.register_paper(
                    paper_id=project_id,
                    paper_text=paper_text,
                    research_summary=result.research_summary,
                    source_files=result.generated_code,
                )
            except Exception as exc:
                logger.warning("Failed to register paper for chat: {}", exc)

            # ---- Determine final status ----
            failed_count = sum(
                1 for s in result.stages if s.status == StageStatus.FAILED
            )
            if failed_count == 0:
                result.status = "completed"
            else:
                result.status = "partial"

        except _PipelineAbortError:
            # Critical stage failure -- result.status already set
            pass

        except Exception as exc:
            result.status = "failed"
            result.error = f"Unexpected pipeline error: {exc}"
            logger.error("Pipeline crashed: {}\n{}", exc, traceback.format_exc())

        result.total_elapsed = time.monotonic() - pipeline_start

        logger.info(
            "=== ReagentPipeline finished: status={}, elapsed={:.1f}s, "
            "stages={}/{} completed ===",
            result.status,
            result.total_elapsed,
            sum(1 for s in result.stages if s.status == StageStatus.COMPLETED),
            len(result.stages),
        )

        await _notify("pipeline", result.status, f"Pipeline {result.status}")

        return result

    # ------------------------------------------------------------------
    # Chat interface (post-pipeline)
    # ------------------------------------------------------------------

    @property
    def chat_handler(self) -> ChatHandler:
        """Access the chat handler for post-pipeline Q&A.

        Returns:
            The :class:`ChatHandler` instance with registered papers.
        """
        return self._chat_handler

    # ------------------------------------------------------------------
    # Individual stage implementations
    # ------------------------------------------------------------------

    async def _stage_parse_paper(self, paper_path: str) -> Any:
        """Stage 1: Parse the PDF paper.

        Runs the full Phase 1 parsing pipeline (text extraction, section
        detection, equation extraction) in a thread executor to avoid
        blocking the event loop.

        Args:
            paper_path: Absolute path to the PDF file.

        Returns:
            A :class:`ParsedPaper` instance.
        """
        logger.info("Stage: PDF Parsing -- {}", paper_path)
        loop = asyncio.get_running_loop()
        parsed = await loop.run_in_executor(None, parse_paper, paper_path)
        logger.info(
            "Parsed paper: {} pages, {} sections, {} equations",
            parsed.document.metadata.page_count,
            len(parsed.sections.sections),
            parsed.equations.total_count,
        )
        return parsed

    async def _stage_extract_sections(self, parsed_paper: Any) -> Dict[str, Any]:
        """Stage 2: Extract and summarise sections from the parsed paper.

        Args:
            parsed_paper: Output of stage 1.

        Returns:
            A dictionary of section names to their content.
        """
        logger.info("Stage: Section Extraction")
        if parsed_paper is None:
            return {}

        sections = parsed_paper.sections
        section_data = {}
        for name, content in sections.sections.items():
            section_data[name] = {
                "text": content.text[:2000] if content.text else "",
                "word_count": content.word_count,
            }

        logger.info("Extracted {} sections", len(section_data))
        return section_data

    async def _stage_extract_equations(self, parsed_paper: Any) -> List[Dict[str, Any]]:
        """Stage 3: Extract equations from the parsed paper.

        Args:
            parsed_paper: Output of stage 1.

        Returns:
            A list of equation dictionaries.
        """
        logger.info("Stage: Equation Extraction")
        if parsed_paper is None:
            return []

        equations = []
        for eq in parsed_paper.equations.equations:
            equations.append({
                "latex": eq.latex,
                "type": eq.equation_type.value if hasattr(eq.equation_type, "value") else str(eq.equation_type),
                "context": eq.context[:200] if eq.context else "",
            })

        logger.info("Extracted {} equations", len(equations))
        return equations

    async def _stage_rag_indexing(
        self, paper_text: str, project_id: str
    ) -> Dict[str, Any]:
        """Stage 4: Index the paper text into the RAG vector store.

        Args:
            paper_text: Full text of the paper.
            project_id: Unique project identifier used as the paper ID.

        Returns:
            A dictionary with indexing metadata.
        """
        logger.info("Stage: RAG Indexing")
        if not paper_text:
            return {"chunks_indexed": 0}

        loop = asyncio.get_running_loop()
        num_chunks = await loop.run_in_executor(
            None,
            self._rag.index_paper,
            paper_text,
            project_id,
        )
        logger.info("Indexed {} chunks for project '{}'", num_chunks, project_id)
        return {"chunks_indexed": num_chunks, "paper_id": project_id}

    async def _stage_research_agent(
        self, paper_text: str, paper_title: str
    ) -> Dict[str, Any]:
        """Stage 5: Run the ResearchAgent to produce a structured summary.

        Args:
            paper_text: Full text of the paper.
            paper_title: Title of the paper.

        Returns:
            The research summary dictionary.
        """
        logger.info("Stage: Research Agent")
        context = {
            "paper_text": paper_text,
            "paper_title": paper_title,
        }
        result = await self._research_agent.run(context)
        logger.info(
            "Research summary produced: algorithm='{}', {} datasets, {} hyperparams",
            result.get("algorithm", "unknown")[:60],
            len(result.get("datasets", [])),
            len(result.get("hyperparameters", {})),
        )
        return result

    async def _stage_planning_agent(
        self, research_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 6: Run the PlanningAgent to produce an implementation plan.

        Args:
            research_summary: Output of the ResearchAgent.

        Returns:
            The implementation plan dictionary.
        """
        logger.info("Stage: Planning Agent")
        context = {"research_summary": research_summary}
        result = await self._planning_agent.run(context)
        logger.info(
            "Plan produced: {} steps",
            len(result.get("steps", result.get("implementation_steps", []))),
        )
        return result

    async def _stage_architecture_agent(
        self,
        research_summary: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 7: Run the ArchitectureAgent to design the project layout.

        Args:
            research_summary: Output of the ResearchAgent.
            plan:             Output of the PlanningAgent.

        Returns:
            The architecture specification dictionary.
        """
        logger.info("Stage: Architecture Agent")
        context = {
            "research_summary": research_summary,
            "plan": plan,
        }
        result = await self._architecture_agent.run(context)
        logger.info(
            "Architecture designed: {} files specified",
            len(result.get("files", result.get("file_structure", []))),
        )
        return result

    async def _stage_test_generation(
        self,
        source_files: Dict[str, str],
        research_summary: Dict[str, Any],
        architecture: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 10b: Generate pytest test files for the project.

        Args:
            source_files:     Generated source code files.
            research_summary: ResearchAgent output.
            architecture:     ArchitectureAgent output.

        Returns:
            A dictionary with ``test_files`` and ``generation_log``.
        """
        logger.info("Stage: Test Generation")
        if not source_files:
            logger.warning("Skipping test generation: no source files")
            return {"test_files": {}, "generation_log": []}

        context = {
            "files": source_files,
            "research_summary": research_summary,
            "architecture": architecture,
        }
        result = await self._testing_agent.run(context)
        logger.info(
            "Test generation: {} test files produced",
            len(result.get("test_files", {})),
        )
        return result

    async def _stage_github_retrieval(
        self,
        research_summary: Dict[str, Any],
        paper_text: str,
    ) -> Dict[str, Any]:
        """Stage 8: Search GitHub for reference repositories and extract code.

        Searches for repositories matching the paper's algorithm and
        keywords, ranks them by relevance, and extracts key source files
        from the top-ranked repository.

        Args:
            research_summary: Output of the ResearchAgent.
            paper_text:       Full text of the paper.

        Returns:
            A dictionary with ``repos`` (ranked list) and ``reference_code``
            (extracted source files from the top repository).
        """
        logger.info("Stage: GitHub Retrieval")

        # Search for repositories
        search_query = research_summary.get("algorithm", "")
        if not search_query:
            search_query = paper_text[:500] if paper_text else ""

        repos = await self._github_searcher.search_from_paper(
            search_query, max_results=10
        )
        if not repos:
            logger.warning("No GitHub repositories found")
            return {"repos": [], "reference_code": {}}

        # Rank repositories
        keywords = []
        if research_summary.get("algorithm"):
            keywords.append(research_summary["algorithm"])
        datasets = research_summary.get("datasets", [])
        for ds in datasets:
            if isinstance(ds, str):
                keywords.append(ds)
            elif isinstance(ds, dict) and ds.get("name"):
                keywords.append(ds["name"])

        ranked = self._repo_ranker.rank(repos, keywords=keywords)
        ranked_repos = [
            scored.repo for scored in ranked[:5]
        ] if ranked else repos[:5]

        logger.info("Found {} repos, top-5 ranked", len(repos))

        # Extract code from the top repository
        reference_code: Dict[str, str] = {}
        if ranked_repos:
            top_repo = ranked_repos[0]
            repo_url = top_repo.get("html_url", top_repo.get("url", ""))
            if repo_url:
                try:
                    extraction = await self._code_extractor.extract(repo_url)
                    reference_code = extraction.get("files", {})
                    logger.info(
                        "Extracted {} reference files from '{}'",
                        len(reference_code),
                        repo_url,
                    )
                except Exception as exc:
                    logger.warning(
                        "Code extraction from '{}' failed: {}", repo_url, exc
                    )

        return {"repos": ranked_repos, "reference_code": reference_code}

    async def _stage_huggingface_retrieval(
        self,
        research_summary: Dict[str, Any],
        paper_text: str,
    ) -> List[Dict[str, Any]]:
        """Stage 9: Search HuggingFace for pre-trained models.

        Args:
            research_summary: Output of the ResearchAgent.
            paper_text:       Full text of the paper.

        Returns:
            A list of model metadata dictionaries.
        """
        logger.info("Stage: HuggingFace Retrieval")
        search_query = research_summary.get("algorithm", "")
        if not search_query:
            search_query = paper_text[:500] if paper_text else ""

        models = await self._hf_retriever.search_from_paper(
            search_query, max_results=10
        )
        logger.info("Found {} HuggingFace models", len(models))
        return models

    async def _stage_code_generation(
        self,
        project_id: str,
        plan: Dict[str, Any],
        architecture: Dict[str, Any],
        research_summary: Dict[str, Any],
        reference_code: Dict[str, str],
    ) -> Any:
        """Stage 10: Generate all project files.

        Args:
            project_id:       Project identifier.
            plan:             PlanningAgent output.
            architecture:     ArchitectureAgent output.
            research_summary: ResearchAgent output.
            reference_code:   Reference code from GitHub.

        Returns:
            A :class:`CodeGenerationResult`.
        """
        logger.info("Stage: Code Generation")
        gen_result = await self._code_gen.generate(
            project_id=project_id,
            plan=plan,
            architecture=architecture,
            research_summary=research_summary,
            reference_code=reference_code,
        )
        logger.info(
            "Code generation: {} files, success={}",
            len(gen_result.files),
            gen_result.success,
        )
        return gen_result

    async def _stage_debug_loop(
        self,
        project_id: str,
        code_gen_result: Any,
        research_summary: Dict[str, Any],
    ) -> Any:
        """Stage 11: Execute the project and run the self-debug loop.

        If initial code generation produced files, runs them in the sandbox
        and iteratively debugs any errors.

        Args:
            project_id:       Project identifier.
            code_gen_result:  Output of the code generation stage.
            research_summary: ResearchAgent output (for debug context).

        Returns:
            A :class:`DebugLoopResult`.
        """
        logger.info("Stage: Self-Debug Loop")

        if code_gen_result is None or not code_gen_result.success:
            logger.warning("Skipping debug loop: code generation failed")
            return None

        debug_result = await self._debug_loop.run_with_existing_code(
            project_id=project_id,
            project_dir=code_gen_result.project_dir,
            files=code_gen_result.files,
            research_summary=research_summary,
        )

        # Store debug observations in memory
        if debug_result and debug_result.iteration_history:
            for iteration in debug_result.iteration_history:
                if iteration.diagnosis:
                    self._memory.store(
                        agent_name="DebugAgent",
                        content=iteration.diagnosis,
                        memory_type="debug_memory",
                        metadata={
                            "project_id": project_id,
                            "attempt": iteration.attempt,
                        },
                    )

        logger.info(
            "Debug loop: success={}, attempts={}",
            debug_result.final_success if debug_result else False,
            debug_result.attempts_made if debug_result else 0,
        )
        return debug_result

    async def _stage_validation(
        self,
        source_files: Dict[str, str],
        research_summary: Dict[str, Any],
        project_id: str,
    ) -> Dict[str, Any]:
        """Stage 12: Run the validation pipeline.

        Args:
            source_files:     Generated source code files.
            research_summary: ResearchAgent output.
            project_id:       Project identifier.

        Returns:
            A validation report dictionary.
        """
        logger.info("Stage: Validation")
        if not source_files:
            logger.warning("Skipping validation: no source files available")
            return {"overall_status": "skipped", "checks_passed": 0, "checks_failed": 0, "details": []}

        project_dir = self._projects_dir / project_id
        report = await self._validator.validate(
            source_files=source_files,
            research_summary=research_summary,
            project_dir=project_dir,
        )
        logger.info(
            "Validation: {} passed, {} failed, status='{}'",
            report.get("checks_passed", 0),
            report.get("checks_failed", 0),
            report.get("overall_status", "unknown"),
        )
        return report

    async def _stage_diagram_generation(
        self,
        research_summary: Dict[str, Any],
        source_files: Dict[str, str],
        architecture: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stage 13: Generate Mermaid.js architecture diagrams.

        Args:
            research_summary: ResearchAgent output.
            source_files:     Generated source code files.
            architecture:     ArchitectureAgent output.

        Returns:
            A dictionary of diagram type to diagram data.
        """
        logger.info("Stage: Diagram Generation")
        if not research_summary:
            logger.warning("Skipping diagram generation: no research summary")
            return {}

        diagrams = await self._diagram_gen.generate(
            research_summary=research_summary,
            source_files=source_files,
            architecture=architecture,
        )
        logger.info("Generated {} diagrams", len(diagrams))
        return diagrams

    async def _stage_reproducibility(
        self,
        research_summary: Dict[str, Any],
        paper_text: str,
    ) -> Any:
        """Stage 14: Compute the reproducibility score.

        Args:
            research_summary: ResearchAgent output.
            paper_text:       Full text of the paper.

        Returns:
            A :class:`ReproducibilityReport`.
        """
        logger.info("Stage: Reproducibility Score")
        report = await self._reproducibility.score(
            research_summary=research_summary,
            paper_text=paper_text,
        )
        logger.info(
            "Reproducibility: overall={:.2f}, grade={}",
            report.overall_score,
            report.grade,
        )
        return report

    async def _stage_package_project(
        self,
        project_id: str,
        pipeline_result: PipelineResult,
    ) -> Optional[str]:
        """Stage 15: Package the generated project as a zip archive.

        Args:
            project_id:      Project identifier.
            pipeline_result: The current pipeline result (for metadata).

        Returns:
            The absolute path to the generated zip file, or *None* on failure.
        """
        logger.info("Stage: Project Packaging")

        # Import the project packager here (it is defined in the same package)
        from backend.orchestration.project_packager import ProjectPackager

        packager = ProjectPackager()
        package_path = await packager.package(
            project_id=project_id,
            pipeline_result=pipeline_result,
        )

        if package_path:
            logger.info("Project packaged: {}", package_path)
        else:
            logger.warning("Project packaging returned no path")

        return package_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_or_create_stage(result: PipelineResult, name: str) -> StageResult:
    """Find an existing stage by name, or append a new one.

    Args:
        result: The pipeline result to search / modify.
        name:   The stage name.

    Returns:
        The :class:`StageResult` instance.
    """
    for stage in result.stages:
        if stage.stage_name == name:
            return stage
    new_stage = StageResult(stage_name=name)
    result.stages.append(new_stage)
    return new_stage


class _PipelineAbortError(Exception):
    """Internal exception raised when a critical stage fails."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"Pipeline aborted at '{stage}': {message}")
