"""Repository Ranker -- Phase 4.

Ranks GitHub repositories retrieved by :mod:`github_search` according to a
multi-factor relevance score.  Scoring factors include star count,
description similarity to the paper keywords, preferred programming
language, and repository recency.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from backend.utils.logging import get_logger

logger = get_logger("retrieval.repo_ranker")

# ---------------------------------------------------------------------------
# Default scoring weights (sum to 1.0)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "stars": 0.25,
    "description_match": 0.35,
    "language": 0.20,
    "recency": 0.20,
}

# Languages ordered by preference for ML/DL paper implementations
LANGUAGE_PREFERENCE: dict[str, float] = {
    "python": 1.0,
    "jupyter notebook": 0.90,
    "julia": 0.60,
    "c++": 0.50,
    "c": 0.45,
    "rust": 0.45,
    "java": 0.35,
    "r": 0.40,
    "go": 0.30,
    "lua": 0.55,       # Torch/Lua legacy
    "matlab": 0.45,
    "scala": 0.25,
    "javascript": 0.20,
    "typescript": 0.20,
}

# Default score for unlisted languages
DEFAULT_LANGUAGE_SCORE = 0.10


@dataclass
class ScoredRepo:
    """A repository with its computed relevance score and breakdown."""

    repo: dict[str, Any]
    total_score: float
    breakdown: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            **self.repo,
            "relevance_score": round(self.total_score, 4),
            "score_breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
        }


class RepoRanker:
    """Rank repositories by multi-factor relevance to a research paper.

    Usage::

        ranker = RepoRanker(keywords=["bert", "attention", "NER"])
        ranked = ranker.rank(repo_list)
    """

    def __init__(
        self,
        keywords: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Initialise the ranker.

        Args:
            keywords: Keywords extracted from the paper used for
                description matching.  If *None*, description matching is
                skipped (weight redistributed to other factors).
            weights: Override the default scoring weights.  Keys must be a
                subset of ``stars``, ``description_match``, ``language``,
                ``recency``.
        """
        self.keywords = [kw.lower() for kw in (keywords or [])]
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self._normalise_weights()
        logger.info(
            "RepoRanker initialised with {} keywords, weights={}",
            len(self.keywords),
            self.weights,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        repos: list[dict[str, Any]],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score and sort repositories by relevance.

        Args:
            repos: List of repository dicts as returned by
                :meth:`GitHubSearcher.search_from_paper`.
            top_k: If given, return only the top *k* results.

        Returns:
            A list of repo dicts augmented with ``relevance_score`` and
            ``score_breakdown``, sorted descending by score.
        """
        if not repos:
            logger.warning("No repositories to rank")
            return []

        scored: list[ScoredRepo] = []
        star_values = [r.get("stars", 0) for r in repos]
        max_stars = max(star_values) if star_values else 1

        for repo in repos:
            breakdown = self._score_repo(repo, max_stars=max_stars)
            total = sum(
                self.weights.get(factor, 0.0) * value
                for factor, value in breakdown.items()
            )
            scored.append(ScoredRepo(repo=repo, total_score=total, breakdown=breakdown))

        scored.sort(key=lambda s: s.total_score, reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        logger.info(
            "Ranked {} repos; top score={:.4f}, bottom score={:.4f}",
            len(scored),
            scored[0].total_score if scored else 0.0,
            scored[-1].total_score if scored else 0.0,
        )

        return [s.to_dict() for s in scored]

    # ------------------------------------------------------------------
    # Individual scoring factors
    # ------------------------------------------------------------------

    def _score_repo(
        self,
        repo: dict[str, Any],
        *,
        max_stars: int,
    ) -> dict[str, float]:
        """Compute per-factor scores for a single repository.

        Each factor produces a value in [0, 1].
        """
        return {
            "stars": self._score_stars(repo.get("stars", 0), max_stars),
            "description_match": self._score_description(
                repo.get("description", ""),
                repo.get("name", ""),
                repo.get("topics", []),
            ),
            "language": self._score_language(repo.get("language")),
            "recency": self._score_recency(repo.get("updated_at", "")),
        }

    @staticmethod
    def _score_stars(stars: int, max_stars: int) -> float:
        """Normalise star count using log scaling.

        Log scaling prevents a single mega-popular repo from dominating.
        The score is relative to the most-starred repo in the result set.

        Args:
            stars: Repository star count.
            max_stars: Maximum star count across the result set.

        Returns:
            A float in [0, 1].
        """
        if max_stars <= 0:
            return 0.0
        # log1p avoids log(0)
        return math.log1p(stars) / math.log1p(max_stars)

    def _score_description(
        self,
        description: str,
        name: str,
        topics: list[str],
    ) -> float:
        """Score how well the repo description matches paper keywords.

        Checks description, repository name, and topic tags for keyword
        overlap.

        Args:
            description: Repository description.
            name: Repository full name (owner/repo).
            topics: Topic tags assigned to the repository.

        Returns:
            A float in [0, 1].
        """
        if not self.keywords:
            return 0.5  # neutral when no keywords available

        searchable = " ".join([
            description.lower(),
            name.lower(),
            " ".join(t.lower() for t in topics),
        ])

        # Tokenise the searchable text into words
        tokens = set(re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", searchable))

        matches = 0
        for kw in self.keywords:
            kw_lower = kw.lower()
            # Exact substring match in the full text
            if kw_lower in searchable:
                matches += 1
            # Also check tokenised match
            elif kw_lower in tokens:
                matches += 1

        return min(matches / len(self.keywords), 1.0)

    @staticmethod
    def _score_language(language: str | None) -> float:
        """Score the repository language by ML-ecosystem preference.

        Args:
            language: Primary language reported by GitHub (may be *None*).

        Returns:
            A float in [0, 1].
        """
        if not language:
            return DEFAULT_LANGUAGE_SCORE
        return LANGUAGE_PREFERENCE.get(language.lower(), DEFAULT_LANGUAGE_SCORE)

    @staticmethod
    def _score_recency(updated_at: str) -> float:
        """Score how recently the repository was updated.

        Uses an exponential decay: repos updated within the last 30 days
        score close to 1.0; repos not updated for 2+ years score near 0.

        Args:
            updated_at: ISO-8601 timestamp of the last update.

        Returns:
            A float in [0, 1].
        """
        if not updated_at:
            return 0.0

        try:
            # GitHub returns ISO 8601 timestamps like "2024-01-15T10:30:00Z"
            updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            logger.debug("Could not parse updated_at timestamp: {}", updated_at)
            return 0.0

        now = datetime.now(timezone.utc)
        days_old = max((now - updated_dt).days, 0)

        # Half-life of ~180 days (6 months)
        half_life = 180.0
        score = math.exp(-0.693 * days_old / half_life)
        return max(min(score, 1.0), 0.0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalise_weights(self) -> None:
        """Ensure weights sum to 1.0.

        If no keywords are provided, redistribute the description_match
        weight among other factors.
        """
        if not self.keywords and "description_match" in self.weights:
            redistributed = self.weights.pop("description_match")
            remaining = list(self.weights.keys())
            if remaining:
                per_factor = redistributed / len(remaining)
                for key in remaining:
                    self.weights[key] += per_factor

        total = sum(self.weights.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            for key in self.weights:
                self.weights[key] /= total
