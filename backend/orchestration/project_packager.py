"""ReagentAI Project Packager -- Phase 13.

Packages a generated ML project into a downloadable zip archive.  The
archive contains all generated source files, architecture diagrams,
the validation report, the reproducibility report, and a machine-readable
``manifest.json`` with project metadata.

Archive layout::

    {project_id}.zip
        {project_id}/
            src/               -- Generated source code files
            diagrams/          -- Mermaid diagram source files (.mmd)
            reports/
                validation.json
                reproducibility.json
            manifest.json      -- Project metadata and file listing
            README.md          -- Auto-generated README

Usage::

    from backend.orchestration.project_packager import ProjectPackager

    packager = ProjectPackager()
    zip_path = await packager.package(
        project_id="abc123",
        pipeline_result=result,
    )
    print(zip_path)  # /path/to/generated_projects/abc123.zip
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from backend.config.settings import settings
from backend.utils.logging import get_logger

if TYPE_CHECKING:
    from backend.orchestration.pipeline import PipelineResult

logger = get_logger("orchestration.project_packager")


class ProjectPackager:
    """Package generated projects into downloadable zip archives.

    The packager creates a well-structured zip file containing all outputs
    from the ReagentAI pipeline: generated source code, diagrams,
    validation and reproducibility reports, and a manifest.

    Parameters
    ----------
    output_dir : Path | str | None
        Directory where zip archives are written.  Defaults to
        ``settings.generated_projects_path``.
    """

    def __init__(self, output_dir: Path | str | None = None) -> None:
        """Initialise the project packager.

        Args:
            output_dir: Target directory for zip files.
        """
        self._output_dir = Path(
            output_dir or settings.generated_projects_path
        ).resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ProjectPackager initialised (output_dir={})", self._output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def package(
        self,
        project_id: str,
        pipeline_result: "PipelineResult",
    ) -> Optional[str]:
        """Create a zip archive of the generated project.

        Collects all outputs from the pipeline result and bundles them
        into a single zip file.  File I/O is offloaded to a thread
        executor to avoid blocking the event loop.

        Args:
            project_id:      Unique project identifier.
            pipeline_result: The complete pipeline result containing all
                             generated artefacts.

        Returns:
            The absolute path to the created zip file, or *None* if
            packaging failed.
        """
        logger.info("Packaging project '{}'", project_id)
        t0 = time.monotonic()

        try:
            loop = asyncio.get_running_loop()
            zip_path = await loop.run_in_executor(
                None,
                self._create_zip,
                project_id,
                pipeline_result,
            )
            elapsed = time.monotonic() - t0
            logger.info(
                "Project '{}' packaged in {:.2f}s: {}",
                project_id,
                elapsed,
                zip_path,
            )
            return str(zip_path)

        except Exception as exc:
            logger.error(
                "Failed to package project '{}': {}", project_id, exc
            )
            return None

    async def package_directory(
        self,
        project_id: str,
        project_dir: Path | str,
    ) -> Optional[str]:
        """Create a zip archive from an existing project directory.

        This is a simpler alternative to :meth:`package` that zips an
        entire directory tree without requiring a ``PipelineResult``.

        Args:
            project_id:  Unique project identifier.
            project_dir: Path to the project directory to archive.

        Returns:
            The absolute path to the created zip file, or *None* on failure.
        """
        project_dir = Path(project_dir)
        if not project_dir.is_dir():
            logger.error("Project directory does not exist: {}", project_dir)
            return None

        try:
            loop = asyncio.get_running_loop()
            zip_path = await loop.run_in_executor(
                None,
                self._zip_directory,
                project_id,
                project_dir,
            )
            return str(zip_path)

        except Exception as exc:
            logger.error(
                "Failed to zip directory '{}': {}", project_dir, exc
            )
            return None

    # ------------------------------------------------------------------
    # Internal: create zip from PipelineResult
    # ------------------------------------------------------------------

    def _create_zip(
        self,
        project_id: str,
        result: "PipelineResult",
    ) -> Path:
        """Build the zip archive from pipeline outputs (runs in executor).

        Args:
            project_id: The project identifier.
            result:     The complete pipeline result.

        Returns:
            The absolute path to the created zip file.
        """
        zip_path = self._output_dir / f"{project_id}.zip"
        prefix = project_id

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Generated source code files
            generated_code = result.generated_code or {}
            for relative_path, content in generated_code.items():
                arcname = f"{prefix}/{relative_path}"
                zf.writestr(arcname, content)
                logger.debug("Added source file: {}", arcname)

            # 2. Diagrams (as .mmd Mermaid files)
            diagrams = result.diagrams or {}
            for diagram_type, diagram_data in diagrams.items():
                mermaid_code = ""
                if isinstance(diagram_data, dict):
                    mermaid_code = diagram_data.get("mermaid", "")
                elif isinstance(diagram_data, str):
                    mermaid_code = diagram_data

                if mermaid_code:
                    arcname = f"{prefix}/diagrams/{diagram_type}.mmd"
                    zf.writestr(arcname, mermaid_code)
                    logger.debug("Added diagram: {}", arcname)

            # 3. Validation report
            validation_report = result.validation_report or {}
            if validation_report:
                report_json = json.dumps(validation_report, indent=2, default=str)
                arcname = f"{prefix}/reports/validation.json"
                zf.writestr(arcname, report_json)
                logger.debug("Added validation report")

            # 4. Reproducibility report
            if result.reproducibility is not None:
                repro_data = (
                    result.reproducibility.to_dict()
                    if hasattr(result.reproducibility, "to_dict")
                    else result.reproducibility
                )
                repro_json = json.dumps(repro_data, indent=2, default=str)
                arcname = f"{prefix}/reports/reproducibility.json"
                zf.writestr(arcname, repro_json)
                logger.debug("Added reproducibility report")

            # 5. Manifest
            manifest = self._build_manifest(project_id, result)
            manifest_json = json.dumps(manifest, indent=2, default=str)
            zf.writestr(f"{prefix}/manifest.json", manifest_json)
            logger.debug("Added manifest.json")

            # 6. Auto-generated README
            readme = self._generate_readme(project_id, result)
            zf.writestr(f"{prefix}/README.md", readme)
            logger.debug("Added README.md")

        logger.info(
            "Zip archive created: {} ({} bytes)",
            zip_path,
            zip_path.stat().st_size,
        )
        return zip_path

    # ------------------------------------------------------------------
    # Internal: zip an existing directory
    # ------------------------------------------------------------------

    def _zip_directory(self, project_id: str, project_dir: Path) -> Path:
        """Zip an entire directory tree (runs in executor).

        Args:
            project_id:  Project identifier (used for the zip filename).
            project_dir: The directory to archive.

        Returns:
            Path to the created zip file.
        """
        zip_path = self._output_dir / f"{project_id}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in project_dir.rglob("*"):
                if file_path.is_file():
                    arcname = f"{project_id}/{file_path.relative_to(project_dir)}"
                    zf.write(file_path, arcname)

        logger.info(
            "Directory zipped: {} -> {} ({} bytes)",
            project_dir,
            zip_path,
            zip_path.stat().st_size,
        )
        return zip_path

    # ------------------------------------------------------------------
    # Manifest generation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_manifest(
        project_id: str,
        result: "PipelineResult",
    ) -> Dict[str, Any]:
        """Build a manifest dictionary summarising the project.

        Args:
            project_id: The project identifier.
            result:     The pipeline result.

        Returns:
            A JSON-serialisable manifest dictionary.
        """
        generated_files = list((result.generated_code or {}).keys())
        diagram_types = list((result.diagrams or {}).keys())

        # Reproducibility summary
        repro_summary = {}
        if result.reproducibility and hasattr(result.reproducibility, "overall_score"):
            repro_summary = {
                "overall_score": result.reproducibility.overall_score,
                "grade": result.reproducibility.grade,
            }

        # Validation summary
        val_summary = {}
        if result.validation_report:
            val_summary = {
                "overall_status": result.validation_report.get("overall_status", "unknown"),
                "checks_passed": result.validation_report.get("checks_passed", 0),
                "checks_failed": result.validation_report.get("checks_failed", 0),
            }

        # Debug summary
        debug_summary = {}
        if result.debug_result and hasattr(result.debug_result, "final_success"):
            debug_summary = {
                "final_success": result.debug_result.final_success,
                "attempts_made": result.debug_result.attempts_made,
            }

        return {
            "project_id": project_id,
            "generator": "ReagentAI",
            "pipeline_status": result.status,
            "total_elapsed_seconds": round(result.total_elapsed, 3),
            "stages": [s.to_dict() for s in result.stages],
            "generated_files": generated_files,
            "generated_file_count": len(generated_files),
            "diagram_types": diagram_types,
            "research_summary": {
                "algorithm": result.research_summary.get("algorithm", ""),
                "datasets": result.research_summary.get("datasets", []),
            },
            "validation": val_summary,
            "reproducibility": repro_summary,
            "debug": debug_summary,
        }

    # ------------------------------------------------------------------
    # README generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_readme(
        project_id: str,
        result: "PipelineResult",
    ) -> str:
        """Generate a README.md for the packaged project.

        Args:
            project_id: The project identifier.
            result:     The pipeline result.

        Returns:
            A Markdown-formatted README string.
        """
        algorithm = result.research_summary.get("algorithm", "Unknown Algorithm")
        summary = result.research_summary.get("summary", "")

        # Reproducibility info
        repro_section = ""
        if result.reproducibility and hasattr(result.reproducibility, "overall_score"):
            repro_section = (
                f"\n## Reproducibility Score\n\n"
                f"- **Overall Score:** {result.reproducibility.overall_score:.2f}\n"
                f"- **Grade:** {result.reproducibility.grade}\n"
            )
            if hasattr(result.reproducibility, "recommendations"):
                for rec in result.reproducibility.recommendations[:3]:
                    repro_section += f"- {rec}\n"

        # Validation info
        val_section = ""
        if result.validation_report:
            status = result.validation_report.get("overall_status", "unknown")
            passed = result.validation_report.get("checks_passed", 0)
            failed = result.validation_report.get("checks_failed", 0)
            val_section = (
                f"\n## Validation\n\n"
                f"- **Status:** {status}\n"
                f"- **Checks Passed:** {passed}\n"
                f"- **Checks Failed:** {failed}\n"
            )

        # File listing
        files = list((result.generated_code or {}).keys())
        file_listing = ""
        if files:
            file_listing = "\n## Generated Files\n\n"
            for f in sorted(files):
                file_listing += f"- `{f}`\n"

        readme = (
            f"# {algorithm}\n\n"
            f"*Auto-generated by ReagentAI*\n\n"
            f"**Project ID:** `{project_id}`\n\n"
        )

        if summary:
            readme += f"## Summary\n\n{summary}\n"

        readme += file_listing
        readme += val_section
        readme += repro_section

        readme += (
            "\n## Getting Started\n\n"
            "```bash\n"
            "pip install -r requirements.txt\n"
            "python src/trainer.py\n"
            "```\n\n"
            "---\n\n"
            "*This project was automatically generated from a research paper "
            "by [ReagentAI](https://github.com/reagentai).*\n"
        )

        return readme
