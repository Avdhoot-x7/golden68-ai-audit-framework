"""Minimal scoring helpers for Golden 68 evaluation outputs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.constants import RESULTS_DIR
from src.reporting.aggregation import (
    aggregate_by_level,
    aggregate_by_pillar,
    build_reliability_placeholder,
    summarize_evaluations,
)
from src.validation import calculate_kappa_from_judge_and_human_records


class Golden68Scorer:
    """Persist evaluation results and expose explicit benchmark metrics only."""

    def __init__(self, results_dir: str | None = None):
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def build_result_bundle(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the standardized result schema for one evaluation run."""
        summary = summarize_evaluations(evaluations)
        return {
            "total": summary["total"],
            "average_score": summary["average_score"],
            "pass_rate": summary["pass_rate"],
            "reliability": build_reliability_placeholder(),
            "pillar_scores": aggregate_by_pillar(evaluations),
            "level_scores": aggregate_by_level(evaluations),
            "evaluations": evaluations,
        }

    def save_evaluation(
        self,
        evaluation_id: str,
        model_name: str,
        result_bundle: Dict[str, Any],
    ) -> str:
        """Save one evaluation bundle to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{evaluation_id}_{timestamp}.json"
        filepath = self.results_dir / filename

        payload = {
            "evaluation_id": evaluation_id,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            **result_bundle,
        }

        filepath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(filepath)

    def load_evaluation(self, filepath: str) -> Dict[str, Any]:
        """Load an evaluation result bundle from disk."""
        return json.loads(Path(filepath).read_text(encoding="utf-8"))

    def build_reliability_from_human_results(
        self,
        judge_evaluations: List[Dict[str, Any]],
        human_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute Cohen's Kappa when aligned human audit labels are available."""
        return calculate_kappa_from_judge_and_human_records(judge_evaluations, human_results)


class AgreementDeltaCalculator:
    """Calculates the agreement metrics between AI Judge scores and Human Audit scores."""
    
    def calculate(self, judge_scores: List[float], human_scores: List[float]) -> Dict[str, Any]:
        if not judge_scores or not human_scores or len(judge_scores) != len(human_scores):
            return {
                "agreement_delta": 0.0,
                "rating": "N/A",
                "mean_absolute_difference": 0.0,
                "exact_agreement_rate": 0.0
            }
            
        n = len(judge_scores)
        exact_matches = sum(1 for j, h in zip(judge_scores, human_scores) if abs(j - h) < 0.1)
        abs_diffs = [abs(j - h) for j, h in zip(judge_scores, human_scores)]
        mean_abs_diff = sum(abs_diffs) / n
        
        # Calculate agreement delta (1 - (mean_abs_diff / 10)) since max score is 10
        agreement_delta = 1.0 - (mean_abs_diff / 10.0)
        
        # Rating based on delta
        if agreement_delta >= 0.9:
            rating = "Excellent"
        elif agreement_delta >= 0.8:
            rating = "Good"
        elif agreement_delta >= 0.7:
            rating = "Fair"
        else:
            rating = "Poor"
            
        return {
            "agreement_delta": agreement_delta,
            "rating": rating,
            "mean_absolute_difference": mean_abs_diff,
            "exact_agreement_rate": exact_matches / n
        }
