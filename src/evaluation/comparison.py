"""
Golden 68 - Multi-Model Comparison & Statistical Analysis
Advanced benchmarking features for research and enterprise
"""

import json
import os
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class ModelComparisonResult:
    """Result of comparing multiple models."""
    model_a: str
    model_b: str
    avg_score_a: float
    avg_score_b: float
    score_diff: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    effect_size: float  # Cohen's d


class MultiModelComparison:
    """Compare multiple models on the same prompts."""
    
    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = results_dir
        
    def load_all_results(self) -> Dict[str, List[Dict]]:
        """Load all results grouped by model."""
        if not os.path.exists(self.results_dir):
            return {}
        
        all_results = {}
        
        for filename in os.listdir(self.results_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(self.results_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            model_name = data.get('test_model', data.get('model', 'unknown'))
            results = data.get('results', [])
            
            if model_name not in all_results:
                all_results[model_name] = []
            all_results[model_name].extend(results)
        
        return all_results
    
    def compare_models(self, model_a: str, model_b: str, all_results: Dict[str, List[Dict]]) -> ModelComparisonResult:
        """Compare two models using statistical tests."""
        results_a = all_results.get(model_a, [])
        results_b = all_results.get(model_b, [])
        
        scores_a = [r.get('judge_score', 0) for r in results_a]
        scores_b = [r.get('judge_score', 0) for r in results_b]
        
        avg_a = np.mean(scores_a) if scores_a else 0
        avg_b = np.mean(scores_b) if scores_b else 0
        
        # Paired t-test (if same prompts)
        if len(scores_a) == len(scores_b) and scores_a and scores_b:
            t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        else:
            t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
        effect_size = (avg_a - avg_b) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval (95%)
        if scores_a and scores_b:
            diff = np.array(scores_a) - np.array(scores_b)
            ci = (np.mean(diff) - 1.96 * np.std(diff) / np.sqrt(len(diff)),
                  np.mean(diff) + 1.96 * np.std(diff) / np.sqrt(len(diff)))
        else:
            ci = (0, 0)
        
        return ModelComparisonResult(
            model_a=model_a,
            model_b=model_b,
            avg_score_a=avg_a,
            avg_score_b=avg_b,
            score_diff=avg_a - avg_b,
            p_value=p_value,
            is_significant=p_value < 0.05,
            confidence_interval=ci,
            effect_size=effect_size
        )
    
    def generate_comparison_report(self, all_results: Dict[str, List[Dict]]) -> str:
        """Generate a comprehensive comparison report."""
        models = list(all_results.keys())
        
        report = "# Multi-Model Comparison Report\n\n"
        report += f"**Generated:** {os.popen('date').read().strip()}\n"
        report += f"**Models Compared:** {len(models)}\n\n"
        
        # Summary table
        report += "## Model Performance Summary\n\n"
        report += "| Model | Evaluations | Avg Score | Pass Rate | Std Dev |\n"
        report += "|-------|-------------|-----------|-----------|----------|\n"
        
        model_stats = []
        for model in models:
            results = all_results[model]
            scores = [r.get('judge_score', 0) for r in results]
            passes = sum(1 for r in results if r.get('judge_determination') == 'PASS')
            
            avg = np.mean(scores) if scores else 0
            std = np.std(scores) if scores else 0
            pass_rate = passes / len(results) * 100 if results else 0
            
            model_stats.append((model, len(results), avg, pass_rate, std))
            report += f"| {model} | {len(results)} | {avg:.2f} | {pass_rate:.1f}% | {std:.2f} |\n"
        
        # Sort by score
        model_stats.sort(key=lambda x: x[2], reverse=True)
        
        # Pairwise comparisons
        if len(models) >= 2:
            report += "\n## Statistical Comparison\n\n"
            report += "| Comparison | Score Diff | p-value | Significant? | Effect Size |\n"
            report += "|------------|------------|---------|--------------|-------------|\n"
            
            # Compare top 2
            if len(model_stats) >= 2:
                top_2 = [model_stats[0], model_stats[1]]
                comp = self.compare_models(top_2[0][0], top_2[1][0], all_results)
                sig = "Yes" if comp.is_significant else "No"
                effect = self._interpret_effect_size(comp.effect_size)
                report += f"| {top_2[0][0]} vs {top_2[1][0]} | {comp.score_diff:+.2f} | {comp.p_value:.4f} | {sig} | {effect} |\n"
        
        # Ranking
        report += "\n## Rankings\n\n"
        for i, (model, count, avg, pr, std) in enumerate(model_stats, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            report += f"{medal} **{model}** - Score: {avg:.2f}/10 (Pass Rate: {pr:.1f}%)\n"
        
        return report


class StatisticalAnalyzer:
    """Statistical analysis for evaluation results."""
    
    @staticmethod
    def analyze_results(results: List[Dict]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        scores = [r.get('judge_score', 0) for r in results]
        passes = [1 if r.get('judge_determination') == 'PASS' else 0 for r in results]
        
        analysis = {
            'mean': np.mean(scores) if scores else 0,
            'median': np.median(scores) if scores else 0,
            'std': np.std(scores) if scores else 0,
            'min': np.min(scores) if scores else 0,
            'max': np.max(scores) if scores else 0,
            'q1': np.percentile(scores, 25) if scores else 0,
            'q3': np.percentile(scores, 75) if scores else 0,
            'pass_rate': sum(passes) / len(passes) * 100 if passes else 0,
            'n': len(scores),
            'confidence_interval_95': StatisticalAnalyzer._confidence_interval(scores, 0.95),
            'confidence_interval_99': StatisticalAnalyzer._confidence_interval(scores, 0.99),
        }
        
        return analysis
    
    @staticmethod
    def _confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval."""
        if not data:
            return (0, 0)
        mean = np.mean(data)
        sem = stats.sem(data)
        margin = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return (mean - margin, mean + margin)
    
    @staticmethod
    def calculate_reliability(scores: List[float]) -> float:
        """Calculate Cronbach's alpha for reliability (simplified)."""
        if len(scores) < 2:
            return 0
        return np.corrcoef(scores[:-1], scores[1:])[0, 1]


class ErrorAnalyzer:
    """Analyze errors and failures in evaluations."""
    
    @staticmethod
    def analyze_failures(results: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in failed evaluations."""
        failures = [r for r in results if r.get('judge_determination') == 'FAIL']
        
        if not failures:
            return {'total_failures': 0, 'patterns': [], 'top_reasons': []}
        
        # Group by pillar
        by_pillar = {}
        for f in failures:
            pillar = f.get('pillar', 'Unknown')
            if pillar not in by_pillar:
                by_pillar[pillar] = []
            by_pillar[pillar].append(f)
        
        # Group by level
        by_level = {}
        for f in failures:
            level = f.get('level', 0)
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(f)
        
        # Common failure patterns in reasoning
        patterns = []
        low_scores = [f for f in failures if f.get('judge_score', 0) <= 3]
        if low_scores:
            patterns.append({
                'type': 'Critical Failure',
                'count': len(low_scores),
                'description': 'Very low scores (1-3) indicating fundamental issues'
            })
        
        return {
            'total_failures': len(failures),
            'failure_rate': len(failures) / len(results) * 100 if results else 0,
            'by_pillar': {k: len(v) for k, v in by_pillar.items()},
            'by_level': {k: len(v) for k, v in by_level.items()},
            'patterns': patterns,
            'critical_failures': low_scores[:5] if low_scores else []
        }


class BenchmarkExporter:
    """Export results in various formats."""
    
    @staticmethod
    def to_csv(results: List[Dict], output_path: str) -> str:
        """Export to CSV format."""
        import csv
        
        if not results:
            return "No results to export"
        
        fieldnames = ['prompt_id', 'pillar', 'level', 'judge_score', 'judge_determination', 
                      'judge_reasoning', 'expected_behavior', 'model_response']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {k: r.get(k, '') for k in fieldnames}
                # Truncate long text fields
                row['judge_reasoning'] = row['judge_reasoning'][:500] if row['judge_reasoning'] else ''
                row['model_response'] = row['model_response'][:500] if row['model_response'] else ''
                writer.writerow(row)
        
        return output_path
    
    @staticmethod
    def to_huggingface_format(results: List[Dict], model_name: str) -> Dict[str, Any]:
        """Export in HuggingFace Open LLM Leaderboard format."""
        scores = [r.get('judge_score', 0) for r in results]
        passes = sum(1 for r in results if r.get('judge_determination') == 'PASS')
        
        return {
            "model_name": model_name,
            "evaluation_dataset": "golden68",
            "average_score": sum(scores) / len(scores) if scores else 0,
            "pass_rate": passes / len(results) * 100 if results else 0,
            "total_samples": len(results),
            "score_std": np.std(scores) if scores else 0,
            "by_pillar": BenchmarkExporter._by_pillar(results)
        }
    
    @staticmethod
    def _by_pillar(results: List[Dict]) -> Dict[str, float]:
        """Calculate scores by pillar."""
        by_pillar = {}
        for r in results:
            p = r.get('pillar', 'unknown')
            if p not in by_pillar:
                by_pillar[p] = []
            by_pillar[p].append(r.get('judge_score', 0))
        
        return {p: sum(s) / len(s) for p, s in by_pillar.items() if s}
