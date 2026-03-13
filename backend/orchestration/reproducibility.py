"""ReagentAI Reproducibility Score -- Phase 11.

Evaluates the reproducibility of a research paper by analysing the
structured information extracted from the paper and checking external
resource availability (GitHub repositories, HuggingFace models).

Six reproducibility factors are scored independently on a 0.0--1.0 scale:

1. **dataset_availability** -- Is the dataset explicitly named and
   publicly available?
2. **hyperparameters_defined** -- Are hyper-parameters (learning rate,
   batch size, epochs, etc.) explicitly stated?
3. **training_details** -- Are training details (optimiser, scheduler,
   hardware) described?
4. **evaluation_metrics** -- Are evaluation metrics specified and standard?
5. **code_available** -- Can a reference implementation be found on GitHub?
6. **model_available** -- Is a pre-trained model available on HuggingFace?

The overall reproducibility score is a weighted average of the factor
scores.  The module also emits human-readable recommendations for
improving reproducibility.

Usage::

    from backend.orchestration.reproducibility import ReproducibilityScorer

    scorer = ReproducibilityScorer()
    report = await scorer.score(
        research_summary={...},
        paper_text="...",
    )
    print(report.overall_score)        # 0.72
    print(report.recommendations)      # ["Consider publishing trained model ..."]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.config.settings import settings
from backend.retrieval.github_search import GitHubSearcher
from backend.retrieval.huggingface_retrieval import HuggingFaceRetriever
from backend.utils.logging import get_logger

logger = get_logger("orchestration.reproducibility")

# ---------------------------------------------------------------------------
# Factor weights for overall score computation
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "dataset_availability": 0.20,
    "hyperparameters_defined": 0.20,
    "training_details": 0.15,
    "evaluation_metrics": 0.15,
    "code_available": 0.20,
    "model_available": 0.10,
}

# ---------------------------------------------------------------------------
# Well-known public datasets (subset for quick matching)
# ---------------------------------------------------------------------------

PUBLIC_DATASETS: set[str] = {
    # Vision
    "imagenet", "cifar-10", "cifar-100", "cifar", "mnist", "fashion-mnist",
    "svhn", "coco", "pascal voc", "voc2007", "voc2012", "ade20k",
    "cityscapes", "kitti", "lfw", "celeba", "places365", "stl-10",
    "oxford flowers", "stanford cars", "caltech-101", "caltech-256",
    "nuscenes", "waymo", "bdd100k",
    # NLP
    "squad", "squad 2.0", "glue", "superglue", "wikitext", "wikitext-2",
    "wikitext-103", "openwebtext", "the pile", "c4", "common crawl",
    "bookcorpus", "ag news", "imdb", "sst-2", "sst-5", "yelp", "amazon",
    "multinli", "snli", "race", "copa", "winogrande", "hellaswag",
    "arc", "mmlu", "truthfulqa", "lambada", "piqa",
    # Multimodal
    "laion", "laion-5b", "laion-400m", "cc3m", "cc12m", "redcaps",
    "flickr30k", "visual genome", "vqa", "vizwiz", "nocaps",
    # Audio / Speech
    "librispeech", "commonvoice", "voxceleb", "audioset",
    # RL
    "atari", "mujoco", "openai gym",
    # Other
    "redpajama", "alpaca", "sharegpt", "orca", "dolly",
}

# ---------------------------------------------------------------------------
# Standard evaluation metrics (for matching)
# ---------------------------------------------------------------------------

STANDARD_METRICS: set[str] = {
    "accuracy", "top-1 accuracy", "top-5 accuracy",
    "precision", "recall", "f1", "f1-score", "f1 score",
    "bleu", "bleu-4", "rouge", "rouge-l", "rouge-1", "rouge-2",
    "meteor", "bertscore",
    "perplexity", "ppl",
    "map", "mean average precision", "ap", "ap50", "ap75",
    "iou", "miou", "dice", "dice coefficient",
    "auc", "auroc", "roc-auc",
    "mse", "rmse", "mae", "mape",
    "psnr", "ssim", "fid", "is", "inception score",
    "cer", "wer", "word error rate", "character error rate",
    "exact match", "em",
    "ndcg", "mrr", "hit@k",
    "elo", "win rate",
    "loss",
}

# ---------------------------------------------------------------------------
# Hyper-parameter keywords
# ---------------------------------------------------------------------------

HYPERPARAMETER_KEYWORDS: set[str] = {
    "learning rate", "lr", "batch size", "epochs", "num_epochs",
    "weight decay", "dropout", "momentum", "warmup", "warmup steps",
    "gradient clipping", "clip grad", "max grad norm",
    "hidden size", "hidden dim", "embedding dim", "num layers",
    "num heads", "attention heads", "num attention heads",
    "sequence length", "max length", "context length",
    "beta1", "beta2", "epsilon", "eps",
    "label smoothing", "temperature",
}

# ---------------------------------------------------------------------------
# Training detail keywords
# ---------------------------------------------------------------------------

TRAINING_DETAIL_KEYWORDS: set[str] = {
    "adam", "adamw", "sgd", "rmsprop", "adafactor", "lion",
    "cosine", "linear", "cosine annealing", "step lr", "warmup",
    "mixed precision", "fp16", "bf16", "amp",
    "gradient accumulation", "gradient checkpointing",
    "distributed", "ddp", "fsdp", "deepspeed", "data parallel",
    "a100", "v100", "h100", "tpu", "gpu",
    "wandb", "tensorboard", "mlflow",
    "checkpoint", "early stopping",
}


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReproducibilityReport:
    """Complete reproducibility assessment for a research paper.

    Attributes:
        factor_scores:   Per-factor score in [0.0, 1.0].
        factor_details:  Human-readable explanation for each score.
        overall_score:   Weighted average of all factor scores.
        recommendations: Actionable suggestions for improving
                         reproducibility.
        weights:         The weight assigned to each factor.
    """

    factor_scores: Dict[str, float] = field(default_factory=dict)
    factor_details: Dict[str, str] = field(default_factory=dict)
    overall_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "factor_scores": self.factor_scores,
            "factor_details": self.factor_details,
            "overall_score": round(self.overall_score, 4),
            "recommendations": self.recommendations,
            "weights": self.weights,
            "grade": self.grade,
        }

    @property
    def grade(self) -> str:
        """Return a letter grade based on the overall score.

        Returns:
            ``"A"`` (>= 0.8), ``"B"`` (>= 0.6), ``"C"`` (>= 0.4),
            ``"D"`` (>= 0.2), or ``"F"`` (< 0.2).
        """
        if self.overall_score >= 0.8:
            return "A"
        elif self.overall_score >= 0.6:
            return "B"
        elif self.overall_score >= 0.4:
            return "C"
        elif self.overall_score >= 0.2:
            return "D"
        else:
            return "F"


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class ReproducibilityScorer:
    """Score the reproducibility of a research paper.

    The scorer analyses the structured research summary and raw paper
    text, and optionally queries GitHub and HuggingFace to check for
    the availability of code and pre-trained models.

    Parameters
    ----------
    weights : dict[str, float] | None
        Custom weight mapping for the six factors.  Weights are
        normalised to sum to 1.0.  If *None*, default weights are used.
    check_external : bool
        If *True* (default), query GitHub and HuggingFace APIs.
        Set to *False* for offline scoring.
    """

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        check_external: bool = True,
    ) -> None:
        self._weights = self._normalise_weights(weights or DEFAULT_WEIGHTS)
        self._check_external = check_external

        if check_external:
            self._github_searcher = GitHubSearcher()
            self._hf_retriever = HuggingFaceRetriever()
        else:
            self._github_searcher = None
            self._hf_retriever = None

        logger.info(
            "ReproducibilityScorer initialised "
            "(check_external={}, weights={})",
            check_external,
            self._weights,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def score(
        self,
        research_summary: Dict[str, Any],
        paper_text: str = "",
    ) -> ReproducibilityReport:
        """Score the reproducibility of a paper.

        Args:
            research_summary: Structured output from ResearchAgent, expected
                to contain keys like ``algorithm``, ``datasets``,
                ``hyperparameters``, ``training_strategy``,
                ``evaluation_metrics``, etc.
            paper_text: Raw full-text of the paper (optional but improves
                keyword-based scoring).

        Returns:
            A :class:`ReproducibilityReport` with per-factor and overall
            scores plus recommendations.
        """
        logger.info("Starting reproducibility assessment")

        report = ReproducibilityReport(weights=dict(self._weights))

        # ---- Factor 1: Dataset Availability ----
        ds_score, ds_detail = self._score_dataset_availability(
            research_summary, paper_text
        )
        report.factor_scores["dataset_availability"] = ds_score
        report.factor_details["dataset_availability"] = ds_detail

        # ---- Factor 2: Hyperparameters Defined ----
        hp_score, hp_detail = self._score_hyperparameters(
            research_summary, paper_text
        )
        report.factor_scores["hyperparameters_defined"] = hp_score
        report.factor_details["hyperparameters_defined"] = hp_detail

        # ---- Factor 3: Training Details ----
        td_score, td_detail = self._score_training_details(
            research_summary, paper_text
        )
        report.factor_scores["training_details"] = td_score
        report.factor_details["training_details"] = td_detail

        # ---- Factor 4: Evaluation Metrics ----
        em_score, em_detail = self._score_evaluation_metrics(
            research_summary, paper_text
        )
        report.factor_scores["evaluation_metrics"] = em_score
        report.factor_details["evaluation_metrics"] = em_detail

        # ---- Factor 5: Code Available (GitHub) ----
        code_score, code_detail = await self._score_code_availability(
            research_summary, paper_text
        )
        report.factor_scores["code_available"] = code_score
        report.factor_details["code_available"] = code_detail

        # ---- Factor 6: Model Available (HuggingFace) ----
        model_score, model_detail = await self._score_model_availability(
            research_summary, paper_text
        )
        report.factor_scores["model_available"] = model_score
        report.factor_details["model_available"] = model_detail

        # ---- Compute overall score ----
        report.overall_score = self._compute_overall_score(
            report.factor_scores
        )

        # ---- Generate recommendations ----
        report.recommendations = self._generate_recommendations(
            report.factor_scores
        )

        logger.info(
            "Reproducibility assessment complete: overall={:.2f} (grade={})",
            report.overall_score,
            report.grade,
        )

        return report

    # ------------------------------------------------------------------
    # Factor 1: Dataset Availability
    # ------------------------------------------------------------------

    def _score_dataset_availability(
        self,
        research: Dict[str, Any],
        paper_text: str,
    ) -> tuple[float, str]:
        """Score dataset availability factor.

        Checks whether datasets are named in the research summary and
        whether they match known public datasets.

        Returns:
            Tuple of (score, detail).
        """
        datasets = research.get("datasets", [])
        text_lower = paper_text.lower() if paper_text else ""

        if not datasets and not text_lower:
            return 0.0, "No dataset information found."

        # Extract dataset names from the research summary
        named_datasets: List[str] = []
        for ds in datasets:
            if isinstance(ds, dict):
                name = ds.get("name", "").strip()
            elif isinstance(ds, str):
                name = ds.strip()
            else:
                continue
            if name:
                named_datasets.append(name.lower())

        # Also scan the raw text for known public datasets
        public_matches: List[str] = []
        for public_name in PUBLIC_DATASETS:
            if public_name in text_lower:
                public_matches.append(public_name)
            for ds_name in named_datasets:
                if public_name in ds_name or ds_name in public_name:
                    if public_name not in public_matches:
                        public_matches.append(public_name)

        if not named_datasets and not public_matches:
            return 0.1, "No recognisable dataset names found in the paper."

        # Score based on how many datasets are named and public
        if public_matches:
            # At least one public dataset identified
            base_score = 0.7
            # Bonus for multiple datasets
            bonus = min(0.3, len(public_matches) * 0.1)
            score = min(1.0, base_score + bonus)
            detail = (
                f"Found {len(public_matches)} public dataset(s): "
                f"{', '.join(public_matches[:5])}"
            )
        elif named_datasets:
            # Datasets are named but not matched to known public ones
            score = 0.4
            detail = (
                f"Found {len(named_datasets)} named dataset(s) but could not "
                f"verify public availability: {', '.join(named_datasets[:5])}"
            )
        else:
            score = 0.2
            detail = "Dataset references found in text but not clearly named."

        return round(score, 2), detail

    # ------------------------------------------------------------------
    # Factor 2: Hyperparameters Defined
    # ------------------------------------------------------------------

    def _score_hyperparameters(
        self,
        research: Dict[str, Any],
        paper_text: str,
    ) -> tuple[float, str]:
        """Score whether hyper-parameters are explicitly stated.

        Returns:
            Tuple of (score, detail).
        """
        hyperparams = research.get("hyperparameters", {})
        text_lower = paper_text.lower() if paper_text else ""

        # Count explicitly stated hyperparameters from the summary
        stated_params: List[str] = []
        if isinstance(hyperparams, dict):
            for key, value in hyperparams.items():
                if value is not None and str(value).strip():
                    stated_params.append(key)
        elif isinstance(hyperparams, list):
            stated_params = [str(h) for h in hyperparams if str(h).strip()]

        # Also scan the raw text for hyperparameter keywords with values
        text_matches: List[str] = []
        for kw in HYPERPARAMETER_KEYWORDS:
            if kw in text_lower:
                # Check if a numeric value follows the keyword
                pattern = re.escape(kw) + r"[\s:=]+[\d.]+"
                if re.search(pattern, text_lower):
                    text_matches.append(kw)
                else:
                    text_matches.append(kw + " (mentioned)")

        total_found = len(set(stated_params) | set(text_matches))

        # Key hyperparameters that should always be present
        critical_params = {"learning rate", "batch size", "epochs"}
        critical_found = sum(
            1 for cp in critical_params
            if any(
                cp.replace(" ", "_") in p.lower() or cp in p.lower()
                for p in stated_params + text_matches
            )
        )

        if total_found == 0:
            return 0.0, "No hyper-parameters found."

        # Base score from total parameters found
        base_score = min(0.6, total_found * 0.1)

        # Bonus for critical parameters
        critical_bonus = critical_found * 0.15

        score = min(1.0, base_score + critical_bonus)

        detail_parts = []
        if stated_params:
            detail_parts.append(
                f"{len(stated_params)} hyper-parameter(s) in summary: "
                f"{', '.join(stated_params[:6])}"
            )
        if text_matches:
            detail_parts.append(
                f"{len(text_matches)} hyper-parameter keyword(s) in text"
            )
        detail_parts.append(
            f"{critical_found}/{len(critical_params)} critical params found"
        )

        return round(score, 2), "; ".join(detail_parts)

    # ------------------------------------------------------------------
    # Factor 3: Training Details
    # ------------------------------------------------------------------

    def _score_training_details(
        self,
        research: Dict[str, Any],
        paper_text: str,
    ) -> tuple[float, str]:
        """Score whether training details are adequately described.

        Returns:
            Tuple of (score, detail).
        """
        training_strategy = research.get("training_strategy", "")
        text_lower = paper_text.lower() if paper_text else ""

        found_details: List[str] = []

        # Check research summary
        if training_strategy and str(training_strategy).strip():
            found_details.append("training_strategy")

        # Scan text for training detail keywords
        for kw in TRAINING_DETAIL_KEYWORDS:
            if kw in text_lower:
                found_details.append(kw)

        # Categorise what was found
        categories = {
            "optimiser": ["adam", "adamw", "sgd", "rmsprop", "adafactor", "lion"],
            "scheduler": ["cosine", "linear", "cosine annealing", "step lr", "warmup"],
            "precision": ["mixed precision", "fp16", "bf16", "amp"],
            "distributed": ["distributed", "ddp", "fsdp", "deepspeed", "data parallel"],
            "hardware": ["a100", "v100", "h100", "tpu", "gpu"],
            "logging": ["wandb", "tensorboard", "mlflow"],
        }

        categories_covered = 0
        category_details: List[str] = []
        for cat_name, cat_keywords in categories.items():
            if any(kw in found_details for kw in cat_keywords):
                categories_covered += 1
                category_details.append(cat_name)

        if not found_details:
            return 0.0, "No training details found."

        # Score based on number of detail categories covered
        score = min(1.0, categories_covered * 0.2 + 0.1)

        # Bonus if training strategy is described in summary
        if "training_strategy" in found_details:
            score = min(1.0, score + 0.15)

        detail = (
            f"Training details found: {categories_covered}/{len(categories)} "
            f"categories covered ({', '.join(category_details)})"
        )

        return round(score, 2), detail

    # ------------------------------------------------------------------
    # Factor 4: Evaluation Metrics
    # ------------------------------------------------------------------

    def _score_evaluation_metrics(
        self,
        research: Dict[str, Any],
        paper_text: str,
    ) -> tuple[float, str]:
        """Score whether evaluation metrics are specified.

        Returns:
            Tuple of (score, detail).
        """
        metrics = research.get("evaluation_metrics", [])
        text_lower = paper_text.lower() if paper_text else ""

        # Extract metric names from the research summary
        named_metrics: List[str] = []
        if isinstance(metrics, list):
            for m in metrics:
                if isinstance(m, dict):
                    name = m.get("name", "").strip().lower()
                elif isinstance(m, str):
                    name = m.strip().lower()
                else:
                    continue
                if name:
                    named_metrics.append(name)
        elif isinstance(metrics, dict):
            named_metrics = [k.lower() for k in metrics.keys() if k.strip()]

        # Scan raw text for standard metric keywords
        standard_found: List[str] = []
        for metric in STANDARD_METRICS:
            if metric in text_lower:
                standard_found.append(metric)
            for nm in named_metrics:
                if metric in nm or nm in metric:
                    if metric not in standard_found:
                        standard_found.append(metric)

        total_unique = len(set(named_metrics) | set(standard_found))

        if total_unique == 0:
            return 0.0, "No evaluation metrics found."

        # Check for quantitative results (numbers near metric names)
        has_quantitative = False
        for metric in standard_found:
            pattern = re.escape(metric) + r".*?[\d]+\.[\d]+"
            if re.search(pattern, text_lower):
                has_quantitative = True
                break

        # Score
        base_score = min(0.6, total_unique * 0.15)
        quant_bonus = 0.3 if has_quantitative else 0.0
        standard_bonus = min(0.2, len(standard_found) * 0.05)
        score = min(1.0, base_score + quant_bonus + standard_bonus)

        detail_parts = []
        if standard_found:
            detail_parts.append(
                f"{len(standard_found)} standard metric(s): "
                f"{', '.join(standard_found[:5])}"
            )
        if has_quantitative:
            detail_parts.append("quantitative results reported")
        if named_metrics:
            detail_parts.append(
                f"{len(named_metrics)} metric(s) in summary"
            )

        return round(score, 2), "; ".join(detail_parts) if detail_parts else "Metrics found."

    # ------------------------------------------------------------------
    # Factor 5: Code Availability (GitHub)
    # ------------------------------------------------------------------

    async def _score_code_availability(
        self,
        research: Dict[str, Any],
        paper_text: str,
    ) -> tuple[float, str]:
        """Score whether a reference implementation is available on GitHub.

        Returns:
            Tuple of (score, detail).
        """
        # Check for explicit code URLs in the paper
        code_url = research.get("code_url", "")
        github_urls: List[str] = []

        if code_url and "github.com" in code_url.lower():
            github_urls.append(code_url)

        # Scan text for GitHub URLs
        if paper_text:
            url_pattern = re.compile(
                r"https?://github\.com/[\w\-]+/[\w\-.]+"
            )
            github_urls.extend(url_pattern.findall(paper_text))

        if github_urls:
            return 1.0, f"Official code repository found: {github_urls[0]}"

        # Fallback: search GitHub API
        if not self._check_external or self._github_searcher is None:
            return 0.0, "No GitHub URL found and external checks disabled."

        try:
            repos = await self._github_searcher.search_from_paper(
                paper_text or research.get("algorithm", ""),
                max_results=5,
            )
        except Exception as exc:
            logger.warning("GitHub search failed: {}", exc)
            repos = []

        if not repos:
            return 0.0, "No related repositories found on GitHub."

        # Score based on repository quality
        top_repo = repos[0]
        stars = top_repo.get("stars", 0)
        name = top_repo.get("name", "")

        if stars >= 100:
            score = 0.8
        elif stars >= 10:
            score = 0.5
        else:
            score = 0.3

        detail = (
            f"Found related repository: {name} "
            f"({stars} stars). Not confirmed as official."
        )

        return round(score, 2), detail

    # ------------------------------------------------------------------
    # Factor 6: Model Availability (HuggingFace)
    # ------------------------------------------------------------------

    async def _score_model_availability(
        self,
        research: Dict[str, Any],
        paper_text: str,
    ) -> tuple[float, str]:
        """Score whether a pre-trained model is available on HuggingFace.

        Returns:
            Tuple of (score, detail).
        """
        # Check for explicit HuggingFace model references
        model_url = research.get("model_url", "")
        hf_refs: List[str] = []

        if model_url and "huggingface.co" in model_url.lower():
            hf_refs.append(model_url)

        # Scan text for HuggingFace URLs or model IDs
        if paper_text:
            hf_url_pattern = re.compile(
                r"https?://huggingface\.co/[\w\-]+/[\w\-.]+"
            )
            hf_refs.extend(hf_url_pattern.findall(paper_text))

        if hf_refs:
            return 1.0, f"HuggingFace model found: {hf_refs[0]}"

        # Fallback: search HuggingFace API
        if not self._check_external or self._hf_retriever is None:
            return 0.0, "No HuggingFace URL found and external checks disabled."

        try:
            models = await self._hf_retriever.search_from_paper(
                paper_text or research.get("algorithm", ""),
                max_results=5,
            )
        except Exception as exc:
            logger.warning("HuggingFace search failed: {}", exc)
            models = []

        if not models:
            return 0.0, "No related models found on HuggingFace."

        # Score based on model quality indicators
        top_model = models[0]
        downloads = top_model.get("downloads", 0)
        model_id = top_model.get("model_id", "")

        if downloads >= 10000:
            score = 0.8
        elif downloads >= 1000:
            score = 0.5
        elif downloads >= 100:
            score = 0.3
        else:
            score = 0.2

        detail = (
            f"Found related model: {model_id} "
            f"({downloads:,} downloads). Not confirmed as official."
        )

        return round(score, 2), detail

    # ------------------------------------------------------------------
    # Overall score computation
    # ------------------------------------------------------------------

    def _compute_overall_score(
        self,
        factor_scores: Dict[str, float],
    ) -> float:
        """Compute the weighted average of all factor scores.

        Args:
            factor_scores: Per-factor scores.

        Returns:
            Overall reproducibility score in [0.0, 1.0].
        """
        total = 0.0
        for factor, score in factor_scores.items():
            weight = self._weights.get(factor, 0.0)
            total += score * weight

        return round(total, 4)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_recommendations(
        factor_scores: Dict[str, float],
    ) -> List[str]:
        """Generate human-readable recommendations based on factor scores.

        Args:
            factor_scores: Per-factor scores.

        Returns:
            List of recommendation strings, ordered by priority
            (lowest-scoring factors first).
        """
        recommendations: List[str] = []

        # Recommendation templates for each factor
        templates: Dict[str, Dict[str, str]] = {
            "dataset_availability": {
                "low": (
                    "The dataset used in this paper could not be identified "
                    "as publicly available. Consider using well-known "
                    "benchmark datasets or publishing your dataset."
                ),
                "medium": (
                    "Some datasets are named but their public availability "
                    "is uncertain. Provide download links or hosting details "
                    "for all datasets used."
                ),
            },
            "hyperparameters_defined": {
                "low": (
                    "Critical hyper-parameters (learning rate, batch size, "
                    "epochs) are not clearly stated. Include a complete "
                    "hyper-parameter table in the paper."
                ),
                "medium": (
                    "Some hyper-parameters are reported but key values may "
                    "be missing. Ensure all hyper-parameters needed to "
                    "reproduce results are explicitly stated."
                ),
            },
            "training_details": {
                "low": (
                    "Training details (optimiser, learning rate schedule, "
                    "hardware) are insufficiently described. Provide a "
                    "comprehensive training protocol."
                ),
                "medium": (
                    "Some training details are present but important "
                    "information (e.g. hardware, training time, random "
                    "seeds) may be missing."
                ),
            },
            "evaluation_metrics": {
                "low": (
                    "No standard evaluation metrics were identified. "
                    "Report results using widely-accepted metrics for "
                    "your task domain."
                ),
                "medium": (
                    "Evaluation metrics are partially reported. Consider "
                    "adding quantitative comparisons with baselines and "
                    "reporting confidence intervals."
                ),
            },
            "code_available": {
                "low": (
                    "No reference implementation was found on GitHub. "
                    "Publishing code dramatically improves reproducibility. "
                    "Consider releasing your code as an open-source repository."
                ),
                "medium": (
                    "A potentially related repository was found but could "
                    "not be confirmed as official. Provide a direct link "
                    "to the official implementation in the paper."
                ),
            },
            "model_available": {
                "low": (
                    "No pre-trained model was found on HuggingFace or "
                    "similar platforms. Publishing model weights enables "
                    "others to reproduce and build upon your work."
                ),
                "medium": (
                    "A potentially related model was found but could not "
                    "be confirmed as official. Publish your trained model "
                    "weights on HuggingFace Hub with a model card."
                ),
            },
        }

        # Sort factors by score (ascending) to prioritize worst areas
        sorted_factors = sorted(
            factor_scores.items(), key=lambda x: x[1]
        )

        for factor, score in sorted_factors:
            if factor not in templates:
                continue
            if score < 0.3:
                rec = templates[factor].get("low", "")
            elif score < 0.7:
                rec = templates[factor].get("medium", "")
            else:
                continue  # Good enough -- no recommendation needed

            if rec:
                recommendations.append(rec)

        if not recommendations:
            recommendations.append(
                "This paper demonstrates strong reproducibility across all "
                "evaluated factors. Keep up the good practices!"
            )

        return recommendations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Normalise weights so they sum to 1.0.

        Args:
            weights: Raw weight mapping.

        Returns:
            Normalised weight mapping.
        """
        total = sum(weights.values())
        if total == 0:
            # Fall back to uniform weights
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: v / total for k, v in weights.items()}
