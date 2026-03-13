"""HuggingFace Model Hub Retrieval -- Phase 5.

Searches the HuggingFace Hub API for pre-trained models matching
keywords extracted from research papers.  Returns structured model
metadata including model ID, pipeline tag, download count, and
library framework.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from backend.config.settings import settings
from backend.utils.logging import get_logger

logger = get_logger("retrieval.huggingface_retrieval")

# ---------------------------------------------------------------------------
# HuggingFace API constants
# ---------------------------------------------------------------------------
HF_API_BASE = "https://huggingface.co/api"
HF_MODELS_ENDPOINT = f"{HF_API_BASE}/models"
DEFAULT_LIMIT = 30
MAX_LIMIT = 100

# ---------------------------------------------------------------------------
# Well-known model families for keyword matching
# ---------------------------------------------------------------------------
KNOWN_MODEL_FAMILIES: dict[str, list[str]] = {
    # Transformers -- NLP
    "bert": ["bert", "bert-base", "bert-large"],
    "gpt": ["gpt2", "gpt-neo", "gpt-j"],
    "gpt-2": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    "gpt-3": ["gpt-3"],
    "gpt-4": ["gpt-4"],
    "t5": ["t5", "t5-small", "t5-base", "t5-large", "flan-t5"],
    "bart": ["bart", "bart-base", "bart-large"],
    "roberta": ["roberta", "roberta-base", "roberta-large"],
    "albert": ["albert"],
    "electra": ["electra"],
    "xlnet": ["xlnet"],
    "pegasus": ["pegasus"],
    "llama": ["llama", "llama-2", "llama-3", "codellama"],
    "mistral": ["mistral", "mixtral"],
    "falcon": ["falcon"],
    "opt": ["opt", "opt-1.3b", "opt-6.7b"],
    "bloom": ["bloom", "bloomz"],
    "phi": ["phi", "phi-2", "phi-3"],
    "gemma": ["gemma"],
    "qwen": ["qwen", "qwen2"],
    "deepseek": ["deepseek"],
    "mamba": ["mamba"],
    "rwkv": ["rwkv"],

    # Transformers -- Vision
    "vit": ["vit", "google/vit"],
    "clip": ["clip", "openai/clip"],
    "dino": ["dino", "dinov2"],
    "swin": ["swin"],
    "convnext": ["convnext"],
    "sam": ["sam", "segment-anything"],
    "detr": ["detr"],
    "yolo": ["yolo", "yolov5", "yolov8"],

    # Diffusion
    "stable diffusion": ["stable-diffusion", "stabilityai"],
    "diffusion": ["diffusion"],
    "dall-e": ["dall-e"],

    # Speech
    "whisper": ["whisper", "openai/whisper"],
    "wav2vec": ["wav2vec2"],

    # Classic architectures
    "resnet": ["resnet", "microsoft/resnet"],
    "efficientnet": ["efficientnet"],
    "mobilenet": ["mobilenet"],
    "densenet": ["densenet"],
    "unet": ["unet"],

    # Multimodal
    "llava": ["llava"],
}

# Pipeline tags for filtering
VALID_PIPELINE_TAGS: set[str] = {
    "text-generation",
    "text2text-generation",
    "text-classification",
    "token-classification",
    "question-answering",
    "summarization",
    "translation",
    "fill-mask",
    "feature-extraction",
    "image-classification",
    "object-detection",
    "image-segmentation",
    "image-to-text",
    "text-to-image",
    "automatic-speech-recognition",
    "audio-classification",
    "zero-shot-classification",
    "sentence-similarity",
    "conversational",
    "table-question-answering",
    "reinforcement-learning",
    "depth-estimation",
    "visual-question-answering",
}


@dataclass
class HFModelInfo:
    """Structured HuggingFace model information."""

    model_id: str
    pipeline_tag: str
    downloads: int
    library_name: str
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    author: str = ""
    last_modified: str = ""
    private: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "model_id": self.model_id,
            "pipeline_tag": self.pipeline_tag,
            "downloads": self.downloads,
            "library_name": self.library_name,
            "likes": self.likes,
            "tags": self.tags,
            "author": self.author,
            "last_modified": self.last_modified,
            "private": self.private,
        }


class HuggingFaceRetriever:
    """Search the HuggingFace Model Hub for models relevant to a paper.

    Usage::

        retriever = HuggingFaceRetriever()
        models = await retriever.search_from_paper(paper_text)
    """

    def __init__(self, limit: int = DEFAULT_LIMIT) -> None:
        self.limit = min(limit, MAX_LIMIT)
        self._headers = self._build_headers()
        logger.info("HuggingFaceRetriever initialised (limit={})", self.limit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_from_paper(
        self,
        paper_text: str,
        *,
        extra_queries: list[str] | None = None,
        pipeline_tag: str | None = None,
        library: str | None = None,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Extract model references from paper text and search HuggingFace.

        Args:
            paper_text: Full text or abstract of the research paper.
            extra_queries: Additional search terms to include.
            pipeline_tag: Filter by pipeline tag (e.g. ``text-generation``).
            library: Filter by library (e.g. ``transformers``, ``diffusers``).
            max_results: Cap on returned models.

        Returns:
            A list of model info dicts, each containing ``model_id``,
            ``pipeline_tag``, ``downloads``, and ``library_name``.
        """
        queries = self.extract_model_queries(paper_text)
        if extra_queries:
            queries.extend(extra_queries)

        if not queries:
            logger.warning("No model queries extracted from paper text")
            return []

        logger.info("Extracted HF search queries: {}", queries)

        # Search for each query and aggregate results
        all_models: dict[str, HFModelInfo] = {}
        for query in queries:
            models = await self.search_models(
                query=query,
                pipeline_tag=pipeline_tag,
                library=library,
            )
            for model in models:
                # Deduplicate by model_id, keeping the version with more
                # downloads
                existing = all_models.get(model.model_id)
                if existing is None or model.downloads > existing.downloads:
                    all_models[model.model_id] = model

        # Sort by downloads descending
        sorted_models = sorted(
            all_models.values(),
            key=lambda m: m.downloads,
            reverse=True,
        )

        if max_results:
            sorted_models = sorted_models[:max_results]

        logger.info(
            "HuggingFace search returned {} unique models from {} queries",
            len(sorted_models),
            len(queries),
        )
        return [m.to_dict() for m in sorted_models]

    async def search_models(
        self,
        query: str,
        *,
        sort: str = "downloads",
        direction: str = "-1",
        pipeline_tag: str | None = None,
        library: str | None = None,
        limit: int | None = None,
    ) -> list[HFModelInfo]:
        """Execute a raw HuggingFace models API search.

        Args:
            query: The search query string.
            sort: Sort field (``downloads``, ``likes``, ``lastModified``).
            direction: Sort direction (``-1`` descending, ``1`` ascending).
            pipeline_tag: Optional pipeline tag filter.
            library: Optional library filter.
            limit: Override the default result limit.

        Returns:
            A list of :class:`HFModelInfo` instances.
        """
        params: dict[str, Any] = {
            "search": query,
            "sort": sort,
            "direction": direction,
            "limit": limit or self.limit,
        }
        if pipeline_tag:
            params["pipeline_tag"] = pipeline_tag
        if library:
            params["library"] = library

        logger.debug("HuggingFace search params: {}", params)

        try:
            async with httpx.AsyncClient(
                headers=self._headers, timeout=30.0
            ) as client:
                response = await client.get(HF_MODELS_ENDPOINT, params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "HuggingFace API HTTP error {}: {}",
                exc.response.status_code,
                exc.response.text[:500],
            )
            return []
        except httpx.RequestError as exc:
            logger.error("HuggingFace API request failed: {}", exc)
            return []
        except Exception as exc:
            logger.error("Unexpected error during HuggingFace search: {}", exc)
            return []

        if not isinstance(data, list):
            logger.warning(
                "Unexpected HuggingFace API response type: {}", type(data).__name__
            )
            return []

        results: list[HFModelInfo] = []
        for item in data:
            results.append(self._parse_model_item(item))

        logger.info(
            "HuggingFace query '{}' returned {} models", query, len(results)
        )
        return results

    async def get_model_details(self, model_id: str) -> dict[str, Any] | None:
        """Fetch detailed information for a specific model.

        Args:
            model_id: The HuggingFace model ID (e.g. ``bert-base-uncased``).

        Returns:
            A dict with model details, or *None* on failure.
        """
        url = f"{HF_API_BASE}/models/{model_id}"
        logger.debug("Fetching model details for: {}", model_id)

        try:
            async with httpx.AsyncClient(
                headers=self._headers, timeout=30.0
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Failed to fetch model details for '{}': HTTP {}",
                model_id,
                exc.response.status_code,
            )
            return None
        except httpx.RequestError as exc:
            logger.error(
                "Network error fetching model details for '{}': {}",
                model_id,
                exc,
            )
            return None
        except Exception as exc:
            logger.error(
                "Unexpected error fetching model details for '{}': {}",
                model_id,
                exc,
            )
            return None

        model_info = self._parse_model_item(data)
        logger.info("Retrieved details for model: {}", model_id)
        return model_info.to_dict()

    # ------------------------------------------------------------------
    # Query extraction
    # ------------------------------------------------------------------

    def extract_model_queries(self, text: str) -> list[str]:
        """Extract HuggingFace-relevant model names from paper text.

        Identifies references to known model families and architecture
        names that are likely to exist on the HuggingFace Hub.

        Args:
            text: The paper text to analyse.

        Returns:
            A deduplicated list of search query strings.
        """
        text_lower = text.lower()
        queries: list[str] = []

        # 1. Match known model families
        for family, search_terms in KNOWN_MODEL_FAMILIES.items():
            if family in text_lower:
                # Use the primary search term (first in the list)
                queries.append(search_terms[0])

        # 2. Look for HuggingFace-style model references (org/model-name)
        hf_pattern = re.compile(
            r"\b([a-zA-Z0-9_-]+/[a-zA-Z0-9._-]{3,})\b"
        )
        for match in hf_pattern.findall(text):
            # Basic validation: exclude things like version numbers
            # or file paths that happen to have slashes
            if not re.match(r"^\d", match) and "/" in match:
                parts = match.split("/")
                if len(parts) == 2 and len(parts[0]) >= 2 and len(parts[1]) >= 3:
                    queries.append(match)

        # 3. Look for model checkpoint references with sizes
        # e.g. "BERT-base", "GPT2-large", "T5-3B"
        size_pattern = re.compile(
            r"\b([A-Za-z][A-Za-z0-9]*[-_]"
            r"(?:tiny|mini|small|base|medium|large|xl|xxl|"
            r"\d+[bBmMkK]?))\b"
        )
        for match in size_pattern.findall(text):
            candidate = match.lower().strip()
            # Only include if it resembles a known model family
            for family in KNOWN_MODEL_FAMILIES:
                if candidate.startswith(family):
                    queries.append(candidate)
                    break

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for q in queries:
            normalised = q.lower().strip()
            if normalised not in seen:
                seen.add(normalised)
                unique.append(q)

        logger.debug(
            "Model query extraction produced {} unique queries", len(unique)
        )
        return unique

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_headers() -> dict[str, str]:
        """Build HTTP headers with optional HuggingFace authentication."""
        headers: dict[str, str] = {
            "Accept": "application/json",
        }
        if settings.huggingface_api_token:
            headers["Authorization"] = f"Bearer {settings.huggingface_api_token}"
            logger.debug(
                "HuggingFace API token configured; authenticated requests enabled"
            )
        else:
            logger.warning(
                "No HuggingFace API token configured; some models may not "
                "be accessible"
            )
        return headers

    @staticmethod
    def _parse_model_item(item: dict[str, Any]) -> HFModelInfo:
        """Parse a single model item from the HuggingFace API response.

        Args:
            item: Raw JSON dict from the HuggingFace API.

        Returns:
            A structured :class:`HFModelInfo` instance.
        """
        # The API returns "modelId" or "id" depending on the endpoint
        model_id = item.get("modelId") or item.get("id", "")

        # Pipeline tag
        pipeline_tag = item.get("pipeline_tag") or item.get("pipelineTag", "")

        # Downloads -- might be nested under "downloads" or "downloadsAllTime"
        downloads = item.get("downloads") or item.get("downloadsAllTime", 0)
        if not isinstance(downloads, int):
            try:
                downloads = int(downloads)
            except (ValueError, TypeError):
                downloads = 0

        # Library name
        library_name = item.get("library_name") or item.get("libraryName", "")

        # Tags
        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        # Author
        author = item.get("author", "")

        # Last modified
        last_modified = item.get("lastModified", "")

        # Private flag
        private = item.get("private", False)

        # Likes
        likes = item.get("likes", 0)
        if not isinstance(likes, int):
            try:
                likes = int(likes)
            except (ValueError, TypeError):
                likes = 0

        return HFModelInfo(
            model_id=model_id,
            pipeline_tag=pipeline_tag,
            downloads=downloads,
            library_name=library_name,
            likes=likes,
            tags=tags,
            author=author,
            last_modified=last_modified,
            private=private,
        )
