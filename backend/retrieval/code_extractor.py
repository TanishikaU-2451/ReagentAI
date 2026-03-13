"""Code Extractor -- Phase 4.

Given a GitHub repository URL, fetches key source files (model
definitions, training scripts, configuration files) via the GitHub
Contents API and returns their content for downstream analysis and
code generation.
"""

from __future__ import annotations

import base64
import posixpath
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from backend.config.settings import settings
from backend.utils.logging import get_logger

logger = get_logger("retrieval.code_extractor")

# ---------------------------------------------------------------------------
# GitHub API constants
# ---------------------------------------------------------------------------
GITHUB_API_BASE = "https://api.github.com"
CONTENTS_ENDPOINT = "/repos/{owner}/{repo}/contents/{path}"
TREE_ENDPOINT = "/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
DEFAULT_BRANCH_ENDPOINT = "/repos/{owner}/{repo}"

# Maximum file size we are willing to fetch (100 KB)
MAX_FILE_SIZE_BYTES = 100_000
# Maximum number of files to extract from a single repository
MAX_FILES_PER_REPO = 30

# ---------------------------------------------------------------------------
# File-matching patterns
# ---------------------------------------------------------------------------

# File extensions we consider relevant
RELEVANT_EXTENSIONS: set[str] = {
    ".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
}

# Filename patterns that suggest ML model / training code
_MODEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"model", re.IGNORECASE),
    re.compile(r"train", re.IGNORECASE),
    re.compile(r"network", re.IGNORECASE),
    re.compile(r"arch(itecture)?", re.IGNORECASE),
    re.compile(r"config", re.IGNORECASE),
    re.compile(r"dataset", re.IGNORECASE),
    re.compile(r"data_?loader", re.IGNORECASE),
    re.compile(r"loss", re.IGNORECASE),
    re.compile(r"optim", re.IGNORECASE),
    re.compile(r"eval", re.IGNORECASE),
    re.compile(r"infer", re.IGNORECASE),
    re.compile(r"predict", re.IGNORECASE),
    re.compile(r"transform", re.IGNORECASE),
    re.compile(r"preprocess", re.IGNORECASE),
    re.compile(r"utils?", re.IGNORECASE),
    re.compile(r"layers?", re.IGNORECASE),
    re.compile(r"modules?", re.IGNORECASE),
    re.compile(r"backbone", re.IGNORECASE),
    re.compile(r"head", re.IGNORECASE),
    re.compile(r"encoder", re.IGNORECASE),
    re.compile(r"decoder", re.IGNORECASE),
    re.compile(r"agent", re.IGNORECASE),
    re.compile(r"environment", re.IGNORECASE),
    re.compile(r"reward", re.IGNORECASE),
    re.compile(r"policy", re.IGNORECASE),
    re.compile(r"requirements", re.IGNORECASE),
    re.compile(r"setup", re.IGNORECASE),
    re.compile(r"main", re.IGNORECASE),
    re.compile(r"run", re.IGNORECASE),
    re.compile(r"app", re.IGNORECASE),
]

# Top-level files that are always interesting
ALWAYS_FETCH: set[str] = {
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "environment.yml",
    "environment.yaml",
    "Makefile",
    "Dockerfile",
}


@dataclass
class ExtractedFile:
    """A single file extracted from a GitHub repository."""

    path: str
    content: str
    size: int
    language: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "size": self.size,
            "language": self.language,
        }


@dataclass
class ExtractionResult:
    """Aggregated extraction result for an entire repository."""

    owner: str
    repo: str
    files: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "repo": self.repo,
            "files": self.files,
            "skipped": self.skipped,
            "errors": self.errors,
            "file_count": len(self.files),
        }


class CodeExtractor:
    """Extract relevant source files from a GitHub repository.

    Usage::

        extractor = CodeExtractor()
        result = await extractor.extract_from_url(
            "https://github.com/owner/repo"
        )
        for filename, content in result.items():
            print(filename)
    """

    def __init__(
        self,
        max_files: int = MAX_FILES_PER_REPO,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
    ) -> None:
        self.max_files = max_files
        self.max_file_size = max_file_size
        self._headers = self._build_headers()
        logger.info(
            "CodeExtractor initialised (max_files={}, max_file_size={})",
            self.max_files,
            self.max_file_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract_from_url(
        self,
        repo_url: str,
        *,
        subdirectory: str = "",
    ) -> dict[str, str]:
        """Extract key files from a GitHub repository.

        Args:
            repo_url: Full GitHub URL, e.g.
                ``https://github.com/owner/repo``.
            subdirectory: Optional subdirectory to scope the extraction to.

        Returns:
            A dict mapping relative file paths to their text content.
        """
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            logger.error("Could not parse repository URL: {}", repo_url)
            return {}

        logger.info("Extracting code from {}/{}", owner, repo)

        result = ExtractionResult(owner=owner, repo=repo)

        # Step 1: list all files in the repo (via the Git Trees API)
        file_tree = await self._fetch_file_tree(owner, repo)
        if not file_tree:
            logger.warning(
                "Could not retrieve file tree for {}/{}; "
                "falling back to contents API root listing",
                owner,
                repo,
            )
            file_tree = await self._fetch_contents_listing(owner, repo, subdirectory)

        if not file_tree:
            logger.error("No files found in {}/{}", owner, repo)
            return {}

        # Step 2: filter for relevant files
        relevant = self._filter_relevant_files(file_tree, subdirectory)
        logger.info(
            "Found {} relevant files out of {} total in {}/{}",
            len(relevant),
            len(file_tree),
            owner,
            repo,
        )

        # Step 3: fetch content for each relevant file (capped)
        files_to_fetch = relevant[: self.max_files]
        async with httpx.AsyncClient(
            headers=self._headers, timeout=30.0
        ) as client:
            for file_info in files_to_fetch:
                path = file_info["path"]
                size = file_info.get("size", 0)

                if size > self.max_file_size:
                    result.skipped.append(
                        f"{path} (size={size} exceeds limit={self.max_file_size})"
                    )
                    logger.debug("Skipping large file: {} ({} bytes)", path, size)
                    continue

                content = await self._fetch_file_content(
                    client, owner, repo, path
                )
                if content is not None:
                    result.files[path] = content
                else:
                    result.errors.append(f"Failed to fetch: {path}")

        logger.info(
            "Extracted {} files from {}/{} ({} skipped, {} errors)",
            len(result.files),
            owner,
            repo,
            len(result.skipped),
            len(result.errors),
        )
        return result.files

    async def extract_from_url_detailed(
        self,
        repo_url: str,
        *,
        subdirectory: str = "",
    ) -> ExtractionResult:
        """Like :meth:`extract_from_url` but returns full
        :class:`ExtractionResult` with metadata.

        Args:
            repo_url: Full GitHub URL.
            subdirectory: Optional subdirectory scope.

        Returns:
            An :class:`ExtractionResult` with files, skipped list, and
            errors.
        """
        owner, repo = self._parse_repo_url(repo_url)
        if not owner or not repo:
            logger.error("Could not parse repository URL: {}", repo_url)
            return ExtractionResult(owner="", repo="")

        result = ExtractionResult(owner=owner, repo=repo)

        file_tree = await self._fetch_file_tree(owner, repo)
        if not file_tree:
            file_tree = await self._fetch_contents_listing(owner, repo, subdirectory)
        if not file_tree:
            result.errors.append("Could not list repository files")
            return result

        relevant = self._filter_relevant_files(file_tree, subdirectory)
        files_to_fetch = relevant[: self.max_files]

        async with httpx.AsyncClient(
            headers=self._headers, timeout=30.0
        ) as client:
            for file_info in files_to_fetch:
                path = file_info["path"]
                size = file_info.get("size", 0)

                if size > self.max_file_size:
                    result.skipped.append(f"{path} ({size} bytes)")
                    continue

                content = await self._fetch_file_content(client, owner, repo, path)
                if content is not None:
                    result.files[path] = content
                else:
                    result.errors.append(f"Failed to fetch: {path}")

        return result

    # ------------------------------------------------------------------
    # File tree retrieval
    # ------------------------------------------------------------------

    async def _fetch_file_tree(
        self,
        owner: str,
        repo: str,
    ) -> list[dict[str, Any]]:
        """Fetch the recursive file tree via the Git Trees API.

        Uses the default branch's HEAD SHA.

        Returns:
            A list of dicts with ``path``, ``type``, and ``size`` keys,
            or an empty list on failure.
        """
        try:
            async with httpx.AsyncClient(
                headers=self._headers, timeout=30.0
            ) as client:
                # Get default branch
                repo_resp = await client.get(
                    f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
                )
                repo_resp.raise_for_status()
                default_branch = repo_resp.json().get("default_branch", "main")

                # Get tree recursively
                tree_url = (
                    f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
                    f"/git/trees/{default_branch}?recursive=1"
                )
                tree_resp = await client.get(tree_url)
                tree_resp.raise_for_status()
                tree_data = tree_resp.json()

            entries = tree_data.get("tree", [])
            # Only keep blobs (files), not trees (directories)
            files = [
                {
                    "path": e["path"],
                    "type": e.get("type", "blob"),
                    "size": e.get("size", 0),
                }
                for e in entries
                if e.get("type") == "blob"
            ]
            logger.debug(
                "Git tree for {}/{} contains {} files", owner, repo, len(files)
            )
            return files

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Failed to fetch tree for {}/{}: HTTP {}",
                owner,
                repo,
                exc.response.status_code,
            )
            return []
        except httpx.RequestError as exc:
            logger.error("Network error fetching tree for {}/{}: {}", owner, repo, exc)
            return []
        except Exception as exc:
            logger.error(
                "Unexpected error fetching tree for {}/{}: {}", owner, repo, exc
            )
            return []

    async def _fetch_contents_listing(
        self,
        owner: str,
        repo: str,
        path: str = "",
    ) -> list[dict[str, Any]]:
        """Fallback: list files via the Contents API (non-recursive root).

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: Subdirectory path (empty string for root).

        Returns:
            A flat list of file dicts.
        """
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
        try:
            async with httpx.AsyncClient(
                headers=self._headers, timeout=30.0
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                items = resp.json()

            if not isinstance(items, list):
                return []

            files = [
                {
                    "path": item["path"],
                    "type": item.get("type", "file"),
                    "size": item.get("size", 0),
                }
                for item in items
                if item.get("type") == "file"
            ]
            logger.debug(
                "Contents listing for {}/{}/{} returned {} files",
                owner,
                repo,
                path,
                len(files),
            )
            return files

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Contents API error for {}/{}/{}: HTTP {}",
                owner,
                repo,
                path,
                exc.response.status_code,
            )
            return []
        except httpx.RequestError as exc:
            logger.error(
                "Network error listing contents for {}/{}: {}", owner, repo, exc
            )
            return []
        except Exception as exc:
            logger.error(
                "Unexpected error listing contents for {}/{}: {}", owner, repo, exc
            )
            return []

    # ------------------------------------------------------------------
    # Individual file fetching
    # ------------------------------------------------------------------

    async def _fetch_file_content(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        path: str,
    ) -> str | None:
        """Fetch the decoded text content of a single file.

        Uses the GitHub Contents API which returns base64-encoded content.

        Args:
            client: Reusable httpx async client.
            owner: Repository owner.
            repo: Repository name.
            path: File path within the repository.

        Returns:
            The decoded file content as a string, or *None* on failure.
        """
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            encoding = data.get("encoding", "")
            raw_content = data.get("content", "")

            if encoding == "base64":
                decoded = base64.b64decode(raw_content).decode("utf-8", errors="replace")
            else:
                decoded = raw_content

            logger.debug("Fetched file: {} ({} chars)", path, len(decoded))
            return decoded

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Failed to fetch {}: HTTP {}", path, exc.response.status_code
            )
            return None
        except httpx.RequestError as exc:
            logger.error("Network error fetching {}: {}", path, exc)
            return None
        except UnicodeDecodeError:
            logger.warning("Binary file skipped: {}", path)
            return None
        except Exception as exc:
            logger.error("Unexpected error fetching {}: {}", path, exc)
            return None

    # ------------------------------------------------------------------
    # File filtering
    # ------------------------------------------------------------------

    def _filter_relevant_files(
        self,
        file_tree: list[dict[str, Any]],
        subdirectory: str = "",
    ) -> list[dict[str, Any]]:
        """Filter the file tree to keep only ML-relevant source files.

        Priority order:
        1. Always-fetch files (requirements.txt, setup.py, etc.)
        2. Files matching model/training name patterns
        3. Other Python files in relevant directories

        Args:
            file_tree: Full list of files in the repository.
            subdirectory: Optional subdirectory scope.

        Returns:
            A sorted list of file dicts, most relevant first.
        """
        priority_high: list[dict[str, Any]] = []
        priority_medium: list[dict[str, Any]] = []
        priority_low: list[dict[str, Any]] = []

        for f in file_tree:
            path: str = f["path"]

            # Scope to subdirectory if specified
            if subdirectory and not path.startswith(subdirectory):
                continue

            basename = posixpath.basename(path)
            _, ext = posixpath.splitext(basename)

            # Skip non-relevant extensions
            if ext not in RELEVANT_EXTENSIONS:
                continue

            # Skip test / docs / build directories
            parts = path.split("/")
            if any(
                p in {"test", "tests", "docs", "doc", "examples", "scripts",
                       "__pycache__", ".github", ".git", "node_modules",
                       "venv", ".venv", "env", "build", "dist", "egg-info"}
                for p in parts
            ):
                continue

            # Always-fetch files get highest priority
            if basename in ALWAYS_FETCH:
                priority_high.append(f)
                continue

            # Check model/training patterns
            if any(pat.search(basename) for pat in _MODEL_PATTERNS):
                priority_medium.append(f)
                continue

            # All other relevant-extension files
            if ext == ".py":
                priority_low.append(f)

        combined = priority_high + priority_medium + priority_low
        logger.debug(
            "File filter: {} high, {} medium, {} low priority",
            len(priority_high),
            len(priority_medium),
            len(priority_low),
        )
        return combined

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_repo_url(url: str) -> tuple[str, str]:
        """Parse a GitHub URL into (owner, repo).

        Handles URLs like:
        - https://github.com/owner/repo
        - https://github.com/owner/repo.git
        - https://github.com/owner/repo/tree/main/...
        - github.com/owner/repo

        Args:
            url: The repository URL.

        Returns:
            A ``(owner, repo)`` tuple, or ``("", "")`` on failure.
        """
        # Normalise: ensure scheme
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        parsed = urlparse(url)
        parts = [p for p in parsed.path.strip("/").split("/") if p]

        if len(parts) < 2:
            return ("", "")

        owner = parts[0]
        repo = parts[1]

        # Strip .git suffix
        if repo.endswith(".git"):
            repo = repo[:-4]

        return (owner, repo)

    @staticmethod
    def _build_headers() -> dict[str, str]:
        """Build HTTP headers with optional GitHub authentication."""
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token}"
            logger.debug("GitHub token configured for CodeExtractor")
        else:
            logger.warning(
                "No GitHub token configured; code extraction rate limits "
                "will be restrictive"
            )
        return headers
