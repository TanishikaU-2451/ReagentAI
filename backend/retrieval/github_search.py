"""GitHub Repository Search -- Phase 4.

Extracts keywords from research paper text (algorithm names, model names,
dataset names) and searches the GitHub API for matching repositories.
Returns structured repository metadata for downstream ranking and code
extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from backend.config.settings import settings
from backend.utils.logging import get_logger

logger = get_logger("retrieval.github_search")

# GitHub API constants
GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
DEFAULT_PER_PAGE = 30
MAX_PER_PAGE = 100

# ---------------------------------------------------------------------------
# Well-known names used for keyword extraction
# ---------------------------------------------------------------------------
KNOWN_MODEL_NAMES: set[str] = {
    "bert", "gpt", "gpt-2", "gpt-3", "gpt-4", "vit", "resnet",
    "transformer", "diffusion", "stable diffusion", "llama", "clip",
    "dall-e", "whisper", "yolo", "unet", "gan", "vae", "lstm", "gru",
    "attention", "convnext", "swin", "dino", "sam", "segment anything",
    "detr", "maskrcnn", "faster rcnn", "efficientnet", "mobilenet",
    "densenet", "inception", "alexnet", "vgg", "xlnet", "roberta",
    "albert", "electra", "t5", "bart", "pegasus", "opt", "bloom",
    "falcon", "mistral", "mixtral", "gemma", "phi", "codellama",
    "llava", "deepseek", "qwen", "mamba", "rwkv",
}

KNOWN_DATASET_NAMES: set[str] = {
    "imagenet", "cifar", "cifar-10", "cifar-100", "mnist", "coco",
    "squad", "glue", "superglue", "wikitext", "openwebtext",
    "laion", "common crawl", "the pile", "c4", "redpajama",
    "alpaca", "sharegpt", "orca", "pascal voc", "ade20k",
    "cityscapes", "kitti", "nuscenes", "waymo",
}

# Regex to catch CamelCase or UPPER_CASE identifiers that look like method /
# algorithm names (e.g. "AdamW", "LoRA", "FlashAttention").
_CAMEL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
_UPPER_RE = re.compile(r"\b[A-Z][A-Z_]{2,}\b")

# Words that frequently appear as false-positive algorithm names
_STOPWORDS: set[str] = {
    "the", "and", "for", "with", "from", "this", "that", "our",
    "which", "also", "based", "using", "method", "approach", "results",
    "section", "figure", "table", "paper", "model", "models", "data",
    "training", "learning", "network", "networks", "layer", "layers",
    "input", "output", "however", "therefore", "moreover", "furthermore",
    "abstract", "introduction", "conclusion", "references", "appendix",
    "IEEE", "ACM", "ICML", "ICLR", "NIPS", "CVPR", "ECCV", "ICCV",
    "AAAI", "IJCAI",
}


@dataclass
class RepoResult:
    """Structured representation of a single GitHub repository result."""

    name: str
    url: str
    stars: int
    description: str
    language: str | None
    owner: str = ""
    topics: list[str] = field(default_factory=list)
    updated_at: str = ""
    forks: int = 0
    open_issues: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "name": self.name,
            "url": self.url,
            "stars": self.stars,
            "description": self.description,
            "language": self.language,
            "owner": self.owner,
            "topics": self.topics,
            "updated_at": self.updated_at,
            "forks": self.forks,
            "open_issues": self.open_issues,
        }


class GitHubSearcher:
    """Search GitHub repositories by keywords extracted from research papers.

    Usage::

        searcher = GitHubSearcher()
        repos = await searcher.search_from_paper(paper_text)
    """

    def __init__(self, per_page: int = DEFAULT_PER_PAGE) -> None:
        self.per_page = min(per_page, MAX_PER_PAGE)
        self._headers = self._build_headers()
        logger.info("GitHubSearcher initialised (per_page={})", self.per_page)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_from_paper(
        self,
        paper_text: str,
        *,
        extra_keywords: list[str] | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Extract keywords from *paper_text* and search GitHub.

        Args:
            paper_text: Full text (or abstract) of the research paper.
            extra_keywords: Optional additional keywords supplied by the
                caller (e.g. from the user prompt).
            max_results: Cap on the number of repositories returned.  Defaults
                to ``self.per_page``.

        Returns:
            A list of repository dicts, each containing *name*, *url*,
            *stars*, *description*, and *language* (plus supplementary
            metadata).
        """
        keywords = self.extract_keywords(paper_text)
        if extra_keywords:
            keywords.extend(extra_keywords)

        if not keywords:
            logger.warning("No keywords extracted from paper text; aborting search")
            return []

        logger.info("Extracted keywords: {}", keywords)
        query = self._build_query(keywords)
        repos = await self.search_repos(query, max_results=max_results)
        logger.info("GitHub search returned {} repositories", len(repos))
        return [r.to_dict() for r in repos]

    async def search_repos(
        self,
        query: str,
        *,
        sort: str = "stars",
        order: str = "desc",
        max_results: int | None = None,
    ) -> list[RepoResult]:
        """Execute a raw GitHub repository search.

        Args:
            query: The search query string (GitHub search syntax).
            sort: Sort field -- ``stars``, ``forks``, or ``updated``.
            order: Sort order -- ``asc`` or ``desc``.
            max_results: Maximum number of results to return.

        Returns:
            A list of :class:`RepoResult` instances.
        """
        effective_per_page = min(self.per_page, max_results) if max_results else self.per_page
        params: dict[str, Any] = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": effective_per_page,
        }

        logger.debug("GitHub search params: {}", params)

        try:
            async with httpx.AsyncClient(
                headers=self._headers, timeout=30.0
            ) as client:
                response = await client.get(GITHUB_SEARCH_URL, params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "GitHub API HTTP error {}: {}",
                exc.response.status_code,
                exc.response.text[:500],
            )
            return []
        except httpx.RequestError as exc:
            logger.error("GitHub API request failed: {}", exc)
            return []
        except Exception as exc:
            logger.error("Unexpected error during GitHub search: {}", exc)
            return []

        items = data.get("items", [])
        total_count = data.get("total_count", 0)
        logger.info(
            "GitHub API returned {} items (total_count={})",
            len(items),
            total_count,
        )

        results: list[RepoResult] = []
        for item in items:
            results.append(self._parse_repo_item(item))

        if max_results:
            results = results[:max_results]

        return results

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------

    def extract_keywords(self, text: str) -> list[str]:
        """Extract research-relevant keywords from paper text.

        Identifies model names, dataset names, and CamelCase / UPPER_CASE
        identifiers that are likely algorithm or framework names.

        Args:
            text: The paper text to analyse.

        Returns:
            A deduplicated list of keywords suitable for a GitHub query.
        """
        text_lower = text.lower()
        keywords: list[str] = []

        # 1. Match known model names
        for name in KNOWN_MODEL_NAMES:
            if name in text_lower:
                keywords.append(name)

        # 2. Match known dataset names
        for name in KNOWN_DATASET_NAMES:
            if name in text_lower:
                keywords.append(name)

        # 3. CamelCase identifiers (e.g. FlashAttention, LoRaAdapter)
        for match in _CAMEL_RE.findall(text):
            token = match.strip()
            if token.lower() not in _STOPWORDS and len(token) >= 4:
                keywords.append(token)

        # 4. UPPER_CASE identifiers (e.g. RLHF, PPO, DPO)
        for match in _UPPER_RE.findall(text):
            token = match.strip()
            if token not in _STOPWORDS and len(token) >= 3:
                keywords.append(token)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for kw in keywords:
            normalised = kw.lower().strip()
            if normalised not in seen:
                seen.add(normalised)
                unique.append(kw)

        logger.debug("Keyword extraction produced {} unique keywords", len(unique))
        return unique

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_headers() -> dict[str, str]:
        """Build HTTP headers, including auth if a token is configured."""
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token}"
            logger.debug("GitHub token configured; authenticated requests enabled")
        else:
            logger.warning(
                "No GitHub token configured; API rate limits will be restrictive"
            )
        return headers

    @staticmethod
    def _build_query(keywords: list[str], *, max_terms: int = 8) -> str:
        """Combine keywords into a GitHub search query string.

        GitHub search has a query-length limit, so we cap at *max_terms*
        and prefer shorter, more specific keywords.

        Args:
            keywords: Extracted keywords.
            max_terms: Maximum number of terms to include.

        Returns:
            A space-separated query string.
        """
        # Sort by length (shorter = more specific, usually model/algo names)
        sorted_kw = sorted(keywords, key=len)[:max_terms]
        query = " ".join(sorted_kw)
        logger.debug("Built GitHub query: '{}'", query)
        return query

    @staticmethod
    def _parse_repo_item(item: dict[str, Any]) -> RepoResult:
        """Parse a single repository item from the GitHub API response."""
        return RepoResult(
            name=item.get("full_name", item.get("name", "")),
            url=item.get("html_url", ""),
            stars=item.get("stargazers_count", 0),
            description=item.get("description") or "",
            language=item.get("language"),
            owner=item.get("owner", {}).get("login", ""),
            topics=item.get("topics", []),
            updated_at=item.get("updated_at", ""),
            forks=item.get("forks_count", 0),
            open_issues=item.get("open_issues_count", 0),
        )
