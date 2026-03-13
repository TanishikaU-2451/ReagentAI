"""ReagentAI FastAPI Application Entry Point.

Exposes the REST API and WebSocket endpoints that the Next.js frontend
communicates with.  Endpoints:

    POST  /api/upload            -- Upload a PDF research paper
    GET   /api/projects/{id}     -- Get project info and status
    GET   /api/code/{id}         -- Get generated file tree
    GET   /api/code/{id}/file    -- Get content of a specific file
    GET   /api/diagrams/{id}     -- Get Mermaid.js diagrams
    GET   /api/validation/{id}   -- Get validation results
    GET   /api/score/{id}        -- Get reproducibility score
    POST  /api/chat/{id}         -- Send a chat message about the paper
    GET   /api/download/{id}     -- Download the generated project zip
    WS    /ws/pipeline/{id}      -- Real-time pipeline progress stream

Usage::

    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.config.settings import settings
from backend.orchestration.pipeline import PipelineResult, ReagentPipeline, StageStatus
from backend.utils.logging import get_logger, setup_logging

# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

setup_logging()
logger = get_logger("backend.main")

app = FastAPI(
    title="ReagentAI",
    description="Convert scientific research papers into executable ML implementations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

# Lazily initialised pipeline (heavy — loads models and agents)
_pipeline: Optional[ReagentPipeline] = None

# In-memory project registry  { project_id: PipelineResult | dict }
_projects: Dict[str, Any] = {}

# Active WebSocket connections per project  { project_id: [WebSocket, ...] }
_ws_connections: Dict[str, list] = {}

UPLOAD_DIR = Path(settings.generated_projects_path).resolve() / "_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _get_pipeline() -> ReagentPipeline:
    """Return the singleton pipeline instance, creating it on first call."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising ReagentPipeline (first request) ...")
        _pipeline = ReagentPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# WebSocket progress broadcaster
# ---------------------------------------------------------------------------

async def _broadcast(project_id: str, payload: dict) -> None:
    """Send a JSON message to all WebSocket subscribers of a project."""
    for ws in list(_ws_connections.get(project_id, [])):
        try:
            await ws.send_json(payload)
        except Exception:
            _ws_connections[project_id].remove(ws)


async def _progress_callback(
    project_id: str, stage: str, status: str, detail: str = ""
) -> None:
    """Pipeline progress callback — broadcasts stage updates via WebSocket."""
    await _broadcast(project_id, {
        "event": "stage_update",
        "stage": stage,
        "status": status,
        "detail": detail,
    })

    # Also broadcast a log entry
    level = "ERROR" if status == StageStatus.FAILED else "INFO"
    await _broadcast(project_id, {
        "event": "log",
        "level": level,
        "message": f"[{stage}] {status}: {detail}" if detail else f"[{stage}] {status}",
    })


# ---------------------------------------------------------------------------
# File-tree helpers
# ---------------------------------------------------------------------------

def _build_file_tree(base_dir: Path) -> list:
    """Recursively build a file-tree list for the frontend CodeViewer."""
    nodes = []
    if not base_dir.is_dir():
        return nodes
    for item in sorted(base_dir.iterdir()):
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        if item.is_dir():
            nodes.append({
                "name": item.name,
                "path": str(item.relative_to(base_dir)),
                "type": "directory",
                "children": _build_file_tree(item),
            })
        else:
            nodes.append({
                "name": item.name,
                "path": str(item.relative_to(base_dir)),
                "type": "file",
            })
    return nodes


def _detect_language(path: str) -> str:
    """Guess programming language from file extension."""
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".yaml": "yaml", ".yml": "yaml", ".json": "json",
        ".md": "markdown", ".txt": "text", ".sh": "shell",
        ".dockerfile": "dockerfile", ".css": "css", ".html": "html",
    }
    ext = Path(path).suffix.lower()
    return ext_map.get(ext, "text")


# ---------------------------------------------------------------------------
# REST API endpoints
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_paper(file: UploadFile = File(...)):
    """Upload a PDF research paper and start the pipeline."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    project_id = uuid.uuid4().hex[:12]

    # Save the uploaded file
    upload_path = UPLOAD_DIR / f"{project_id}.pdf"
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    logger.info("Uploaded paper '{}' → project_id={}", file.filename, project_id)

    # Register the project as pending
    _projects[project_id] = {
        "id": project_id,
        "title": file.filename,
        "status": "pending",
        "paper_filename": file.filename,
        "stages": [],
        "upload_path": str(upload_path),
    }

    # Launch the pipeline in the background
    asyncio.create_task(_run_pipeline(project_id, str(upload_path)))

    return JSONResponse(
        status_code=202,
        content={"project_id": project_id, "message": "Pipeline started."},
    )


async def _run_pipeline(project_id: str, paper_path: str) -> None:
    """Execute the full pipeline for a project (runs as a background task)."""
    try:
        pipeline = _get_pipeline()

        async def _cb(stage: str, status: str, detail: str = "") -> None:
            await _progress_callback(project_id, stage, status, detail)

        result: PipelineResult = await pipeline.run(
            paper_path=paper_path,
            project_id=project_id,
            progress_callback=_cb,
        )

        _projects[project_id] = result

        # Notify completion
        await _broadcast(project_id, {
            "event": "pipeline_complete" if result.status != "failed" else "pipeline_error",
            "status": result.status,
            "error": result.error,
        })

    except Exception as exc:
        logger.error("Pipeline failed for {}: {}", project_id, exc)
        _projects[project_id] = {
            "id": project_id,
            "status": "failed",
            "error": str(exc),
        }
        await _broadcast(project_id, {
            "event": "pipeline_error",
            "status": "failed",
            "error": str(exc),
        })


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Return project info and current status."""
    project = _projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")

    if isinstance(project, PipelineResult):
        stages_list = [
            {
                "name": s.stage_name,
                "status": s.status,
                "agentName": s.stage_name.replace("_", " ").title(),
                "activityText": s.detail,
                "error": s.error,
            }
            for s in project.stages
        ]
        return {
            "id": project.project_id,
            "title": project.parsed_paper.document.metadata.title
            if project.parsed_paper else project_id,
            "status": project.status,
            "paper_filename": "",
            "stages": stages_list,
            "score": project.reproducibility.overall_score
            if project.reproducibility and hasattr(project.reproducibility, "overall_score")
            else None,
        }

    # Still a plain dict (pending / failed early)
    return project


@app.get("/api/code/{project_id}")
async def get_code_files(project_id: str):
    """Return the generated project's file tree."""
    project_dir = Path(settings.generated_projects_path).resolve() / project_id
    if not project_dir.is_dir():
        raise HTTPException(status_code=404, detail="No generated code found.")

    return {"files": _build_file_tree(project_dir)}


@app.get("/api/code/{project_id}/file")
async def get_file_content(project_id: str, path: str):
    """Return the content of a specific generated file."""
    project_dir = Path(settings.generated_projects_path).resolve() / project_id
    file_path = (project_dir / path).resolve()

    # Prevent path traversal
    if not str(file_path).startswith(str(project_dir)):
        raise HTTPException(status_code=403, detail="Access denied.")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read file.")

    return {
        "path": path,
        "content": content,
        "language": _detect_language(path),
    }


@app.get("/api/diagrams/{project_id}")
async def get_diagrams(project_id: str):
    """Return Mermaid.js diagrams for the project."""
    project = _projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")

    diagrams_data = {}
    if isinstance(project, PipelineResult):
        diagrams_data = project.diagrams or {}

    diagram_list = []
    for dtype, data in diagrams_data.items():
        if isinstance(data, str):
            diagram_list.append({"type": dtype, "title": dtype.replace("_", " ").title(), "mermaidCode": data})
        elif isinstance(data, dict):
            diagram_list.append({
                "type": dtype,
                "title": data.get("title", dtype.replace("_", " ").title()),
                "mermaidCode": data.get("mermaid", data.get("code", "")),
            })

    return {"diagrams": diagram_list}


@app.get("/api/validation/{project_id}")
async def get_validation(project_id: str):
    """Return validation results for the project."""
    project = _projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")

    if isinstance(project, PipelineResult) and project.validation_report:
        report = project.validation_report
        checks = []
        for detail in report.get("details", report.get("checks", [])):
            if isinstance(detail, dict):
                checks.append({
                    "name": detail.get("name", detail.get("check", "")),
                    "passed": detail.get("passed", detail.get("status") == "passed"),
                    "message": detail.get("message", detail.get("detail", "")),
                })
        return {
            "summary": report.get("overall_status", "unknown"),
            "checks": checks,
        }

    return {"summary": "pending", "checks": []}


@app.get("/api/score/{project_id}")
async def get_score(project_id: str):
    """Return reproducibility score for the project."""
    project = _projects.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")

    if isinstance(project, PipelineResult) and project.reproducibility:
        repro = project.reproducibility
        categories = []
        if hasattr(repro, "category_scores"):
            for cat_name, cat_score in repro.category_scores.items():
                categories.append({
                    "name": cat_name.replace("_", " ").title(),
                    "score": round(cat_score * 100),
                    "max": 100,
                })
        return {
            "overall_score": round(repro.overall_score * 100) if hasattr(repro, "overall_score") else 0,
            "categories": categories,
            "feedback": repro.feedback if hasattr(repro, "feedback") else "",
        }

    return {"overall_score": 0, "categories": [], "feedback": ""}


@app.post("/api/chat/{project_id}")
async def chat_with_paper(project_id: str, body: dict):
    """Send a chat message about the paper and receive a RAG-augmented answer."""
    message = body.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")

    pipeline = _get_pipeline()
    try:
        response = await pipeline.chat_handler.ask(
            paper_id=project_id,
            question=message,
        )
        sources = []
        if hasattr(response, "sources") and response.sources:
            for src in response.sources:
                if isinstance(src, dict):
                    sources.append({
                        "section": src.get("section", ""),
                        "page": src.get("page", None),
                        "text": src.get("text", "")[:300],
                    })
                elif isinstance(src, str):
                    sources.append({
                        "section": "",
                        "page": None,
                        "text": src[:300],
                    })
                else:
                    sources.append({
                        "section": getattr(src, "section", ""),
                        "page": getattr(src, "page", None),
                        "text": getattr(src, "text", str(src))[:300],
                    })
        return {
            "answer": response.answer if hasattr(response, "answer") else str(response),
            "citations": sources,
        }
    except Exception as exc:
        logger.error("Chat error for {}: {}", project_id, exc)
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")


@app.get("/api/download/{project_id}")
async def download_project(project_id: str):
    """Download the packaged project zip file."""
    project = _projects.get(project_id)

    # Try PipelineResult.package_path first
    zip_path = None
    if isinstance(project, PipelineResult) and project.package_path:
        zip_path = Path(project.package_path)

    # Fallback — look for the zip in generated_projects/
    if zip_path is None or not zip_path.is_file():
        zip_path = Path(settings.generated_projects_path).resolve() / f"{project_id}.zip"

    if not zip_path.is_file():
        # Try creating a zip on the fly from the project directory
        project_dir = Path(settings.generated_projects_path).resolve() / project_id
        if project_dir.is_dir():
            zip_path = project_dir.parent / f"{project_id}.zip"
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", project_dir)
        else:
            raise HTTPException(status_code=404, detail="Project package not found.")

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"{project_id}.zip",
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/pipeline/{project_id}")
async def websocket_pipeline(websocket: WebSocket, project_id: str):
    """Real-time pipeline progress stream."""
    await websocket.accept()

    if project_id not in _ws_connections:
        _ws_connections[project_id] = []
    _ws_connections[project_id].append(websocket)

    logger.info("WebSocket connected: project_id={}", project_id)

    # Send current project state immediately
    project = _projects.get(project_id)
    if project is not None:
        if isinstance(project, PipelineResult):
            await websocket.send_json({
                "event": "project_info",
                "data": project.summary(),
            })
        else:
            await websocket.send_json({
                "event": "project_info",
                "data": project,
            })

    try:
        # Keep the connection alive, reading messages (client may send pings)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if project_id in _ws_connections:
            _ws_connections[project_id] = [
                ws for ws in _ws_connections[project_id] if ws != websocket
            ]
        logger.info("WebSocket disconnected: project_id={}", project_id)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "reagentai-backend"}
