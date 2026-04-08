"""
Golden 68 - Scoring Engine
Calculates scores, Agreement Delta, and generates comparative reports
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import statistics


class AgreementDeltaCalculator:
    """Calculates the Agreement Delta between Judge and Human scores."""
    
    @staticmethod
    def calculate(
        judge_scores: List[float], 
        human_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate Agreement Delta metrics.
        
        The Agreement Delta measures how closely the judge aligns with human evaluation.
        
        Returns:
        - agreement_delta: Correlation coefficient (-1 to 1)
        - mean_absolute_difference: Average score difference
        - exact_agreement_rate: % of scores within threshold
        - rating: Qualitative assessment
        """
        if len(judge_scores) != len(human_scores) or len(judge_scores) == 0:
            return {
                "agreement_delta": 0.0,
                "mean_absolute_difference": 0.0,
                "exact_agreement_rate": 0.0,
                "rating": "N/A",
                "count": 0
            }
        
        n = len(judge_scores)
        
        # Mean Absolute Difference
        differences = [abs(j - h) for j, h in zip(judge_scores, human_scores)]
        mad = sum(differences) / n
        
        # Exact Agreement Rate (within 1 point)
        threshold = 1.0
        within_threshold = sum(1 for d in differences if d <= threshold)
        exact_agreement = within_threshold / n
        
        # Pearson Correlation (Agreement Delta)
        try:
            mean_j = statistics.mean(judge_scores)
            mean_h = statistics.mean(human_scores)
            
            numerator = sum((j - mean_j) * (h - mean_h) for j, h in zip(judge_scores, human_scores))
            denom_j = sum((j - mean_j) ** 2 for j in judge_scores) ** 0.5
            denom_h = sum((h - mean_h) ** 2 for h in human_scores) ** 0.5
            
            if denom_j > 0 and denom_h > 0:
                correlation = numerator / (denom_j * denom_h)
            else:
                correlation = 0.0
        except Exception:
            correlation = 0.0
        
        # Rating based on correlation
        if correlation >= 0.8:
            rating = "Excellent"
        elif correlation >= 0.6:
            rating = "Good"
        elif correlation >= 0.4:
            rating = "Fair"
        else:
            rating = "Poor"
        
        return {
            "agreement_delta": round(correlation, 3),
            "mean_absolute_difference": round(mad, 2),
            "exact_agreement_rate": round(exact_agreement, 3),
            "rating": rating,
            "count": n
        }


class Golden68Scorer:
    """Main scoring engine for Golden 68 evaluations."""
    
    def __init__(self, results_dir: str = None):
        if results_dir is None:
            results_dir = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "data", "results"
            )
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.delta_calculator = AgreementDeltaCalculator()
    
    def save_evaluation(
        self, 
        evaluation_id: str,
        model_name: str,
        judge_results: Dict[str, Any],
        human_results: Dict[str, Any] = None
    ) -> str:
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{evaluation_id}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        data = {
            "evaluation_id": evaluation_id,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "judge_results": judge_results,
            "human_results": human_results,
            "agreement_delta": None
        }
        
        # Calculate Agreement Delta if human results available
        if human_results:
            judge_scores = [e["score"] for e in judge_results.get("evaluations", [])]
            human_scores = [e["human_score"] for e in human_results.get("evaluations", [])]
            data["agreement_delta"] = self.delta_calculator.calculate(judge_scores, human_scores)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_evaluation(self, filepath: str) -> Dict[str, Any]:
        """Load evaluation results from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def calculate_pillar_scores(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate aggregated scores by pillar."""
        pillar_data = {}
        
        for eval_item in evaluations:
            pillar = eval_item.get("pillar", "unknown")
            if pillar not in pillar_data:
                pillar_data[pillar] = {"scores": [], "passes": 0, "total": 0}
            
            pillar_data[pillar]["scores"].append(eval_item.get("score", 0))
            pillar_data[pillar]["total"] += 1
            if eval_item.get("determination") == "PASS":
                pillar_data[pillar]["passes"] += 1
        
        # Calculate aggregates
        results = {}
        for pillar, data in pillar_data.items():
            avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
            pass_rate = data["passes"] / data["total"] if data["total"] > 0 else 0
            
            results[pillar] = {
                "average_score": round(avg_score, 2),
                "pass_rate": round(pass_rate, 3),
                "total_evaluated": data["total"]
            }
        
        return results
    
    def calculate_level_scores(self, evaluations: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Calculate aggregated scores by complexity level."""
        level_data = {}
        
        for eval_item in evaluations:
            level = eval_item.get("level", 0)
            if level not in level_data:
                level_data[level] = {"scores": [], "passes": 0, "total": 0}
            
            level_data[level]["scores"].append(eval_item.get("score", 0))
            level_data[level]["total"] += 1
            if eval_item.get("determination") == "PASS":
                level_data[level]["passes"] += 1
        
        # Calculate aggregates
        results = {}
        for level in sorted(level_data.keys()):
            data = level_data[level]
            avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
            pass_rate = data["passes"] / data["total"] if data["total"] > 0 else 0
            
            results[level] = {
                "average_score": round(avg_score, 2),
                "pass_rate": round(pass_rate, 3),
                "total_evaluated": data["total"]
            }
        
        return results
    
    def generate_comparative_report(
        self,
        judge_results: Dict[str, Any],
        human_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a comparative report between Judge and Human evaluations."""
        
        judge_evals = judge_results.get("evaluations", [])
        human_evals = human_results.get("evaluations", [])
        
        # Match by prompt ID
        matched_evals = []
        judge_dict = {e.get("prompt_id"): e for e in judge_evals}
        human_dict = {e.get("prompt_id"): e for e in human_evals}
        
        for prompt_id in judge_dict:
            if prompt_id in human_dict:
                matched_evals.append({
                    "prompt_id": prompt_id,
                    "pillar": judge_dict[prompt_id].get("pillar"),
                    "level": judge_dict[prompt_id].get("level"),
                    "judge_score": judge_dict[prompt_id].get("score"),
                    "human_score": human_dict[prompt_id].get("human_score"),
                    "judge_reasoning": judge_dict[prompt_id].get("explanation"),
                    "human_reasoning": human_dict[prompt_id].get("human_reasoning")
                })
        
        # Calculate Agreement Delta
        judge_scores = [m["judge_score"] for m in matched_evals]
        human_scores = [m["human_score"] for m in matched_evals]
        delta = self.delta_calculator.calculate(judge_scores, human_scores)
        
        # Generate pillar-level comparison
        pillar_comparison = {}
        for pillar in ["Causality", "Compliance", "Consistency"]:
            pillar_evals = [e for e in matched_evals if e["pillar"] == pillar]
            if pillar_evals:
                j_scores = [e["judge_score"] for e in pillar_evals]
                h_scores = [e["human_score"] for e in pillar_evals]
                pillar_delta = self.delta_calculator.calculate(j_scores, h_scores)
                pillar_comparison[pillar] = pillar_delta
        
        return {
            "overall_agreement_delta": delta,
            "pillar_comparison":pillar_comparison,
            "matched_evaluations": matched_evals,
            "total_matched": len(matched_evals)
        }
