"""
Golden 68 - Report Generator
Generates dual reports: LLM-Judge Report and Human-Audit Report
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..evaluation.scorer import Golden68Scorer


class ReportGenerator:
    """Generates comprehensive evaluation reports."""
    
    def __init__(self, reports_dir: str = None):
        if reports_dir is None:
            reports_dir = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "data", "reports"
            )
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def generate_llm_judge_report(
        self,
        model_name: str,
        judge_results: Dict[str, Any],
        pillar_scores: Dict[str, Any],
        level_scores: Dict[int, Any]
    ) -> Dict[str, Any]:
        """Generate the LLM-Judge automated assessment report."""
        
        total = judge_results.get("total_evaluations", 0)
        overall_score = judge_results.get("overall_score", 0)
        pass_rate = judge_results.get("pass_rate", 0)
        
        report = {
            "report_type": "LLM_JUDGE_REPORT",
            "model_name": model_name,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_prompts_evaluated": total,
                "overall_score": round(overall_score, 2),
                "overall_pass_rate": round(pass_rate * 100, 1),
                "grade": self._calculate_grade(overall_score)
            },
            "pillar_breakdown": {},
            "level_breakdown": {},
            "detailed_evaluations": judge_results.get("evaluations", []),
            "recommendations": self._generate_recommendations(pillar_scores)
        }
        
        # Add pillar scores
        for pillar, scores in pillar_scores.items():
            report["pillar_breakdown"][pillar] = {
                "average_score": scores.get("average_score", 0),
                "pass_rate": f"{scores.get('pass_rate', 0) * 100:.1f}%",
                "prompts_tested": scores.get("total_evaluated", 0)
            }
        
        # Add level scores
        for level, scores in level_scores.items():
            report["level_breakdown"][f"Level_{level}"] = {
                "average_score": scores.get("average_score", 0),
                "pass_rate": f"{scores.get('pass_rate', 0) * 100:.1f}%",
                "prompts_tested": scores.get("total_evaluated", 0)
            }
        
        return report
    
    def generate_human_audit_report(
        self,
        audit_results: Dict[str, Any],
        audit_statistics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate the Human-Audit ground-truth report."""
        
        total = audit_statistics.get("total_audits", 0)
        
        report = {
            "report_type": "HUMAN_AUDIT_REPORT",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_audits_completed": total,
                "agree_rate": f"{audit_statistics.get('agree_rate', 0) * 100:.1f}%",
                "partial_agreement_rate": f"{audit_statistics.get('partial_rate', 0) * 100:.1f}%",
                "disagree_rate": f"{audit_statistics.get('disagree_rate', 0) * 100:.1f}%",
                "average_human_score": round(audit_statistics.get("average_human_score", 0), 2),
                "average_judge_score": round(audit_statistics.get("average_judge_score", 0), 2)
            },
            "verdict_breakdown": {
                "agree": audit_statistics.get("agree_count", 0),
                "partial": audit_statistics.get("partial_count", 0),
                "disagree": audit_statistics.get("disagree_count", 0)
            },
            "detailed_audits": audit_results.get("audits", [])
        }
        
        return report
    
    def generate_comparison_report(
        self,
        llm_judge_report: Dict[str, Any],
        human_audit_report: Dict[str, Any],
        agreement_delta: Dict[str, Any],
        pillar_comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate the comparison/validation report."""
        
        report = {
            "report_type": "COMPARISON_REPORT",
            "generated_at": datetime.now().isoformat(),
            "agreement_analysis": {
                "overall_delta": agreement_delta.get("agreement_delta", 0),
                "delta_rating": agreement_delta.get("rating", "N/A"),
                "mean_absolute_difference": agreement_delta.get("mean_absolute_difference", 0),
                "exact_agreement_rate": f"{agreement_delta.get('exact_agreement_rate', 0) * 100:.1f}%",
                "evaluations_compared": agreement_delta.get("count", 0)
            },
            "pillar_agreement": {},
            "key_findings": [],
            "validation_status": self._determine_validation_status(agreement_delta)
        }
        
        # Add pillar comparison
        for pillar, delta in pillar_comparison.items():
            report["pillar_agreement"][pillar] = {
                "delta": delta.get("agreement_delta", 0),
                "rating": delta.get("rating", "N/A")
            }
        
        # Add key findings
        if agreement_delta.get("agreement_delta", 0) >= 0.8:
            report["key_findings"].append(
                "Judge demonstrates excellent alignment with human evaluation"
            )
        elif agreement_delta.get("agreement_delta", 0) >= 0.6:
            report["key_findings"].append(
                "Judge shows good correlation with human evaluation"
            )
        else:
            report["key_findings"].append(
                "Significant divergence between judge and human evaluation requires investigation"
            )
        
        return report
    
    def save_report(
        self, 
        report: Dict[str, Any], 
        prefix: str = "report"
    ) -> str:
        """Save report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        filepath = os.path.join(self.reports_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def generate_markdown_report(
        self,
        llm_judge_report: Dict[str, Any],
        human_audit_report: Dict[str, Any] = None,
        comparison_report: Dict[str, Any] = None
    ) -> str:
        """Generate a markdown-formatted report for display."""
        
        md = f"""# Golden 68 - Evaluation Report

**Model:** {llm_judge_report.get('model_name', 'Unknown')}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Prompts Evaluated | {llm_judge_report['summary']['total_prompts_evaluated']} |
| Overall Score | {llm_judge_report['summary']['overall_score']}/10 |
| Pass Rate | {llm_judge_report['summary']['overall_pass_rate']}% |
| Grade | {llm_judge_report['summary']['grade']} |

"""
        
        # Pillar Breakdown
        md += "## Pillar Breakdown\n\n"
        md += "| Pillar | Avg Score | Pass Rate | Prompts |\n"
        md += "|--------|-----------|-----------|----------|\n"
        for pillar, data in llm_judge_report.get("pillar_breakdown", {}).items():
            md += f"| {pillar} | {data['average_score']}/10 | {data['pass_rate']} | {data['prompts_tested']} |\n"
        
        md += "\n"
        
        # Level Breakdown
        md += "## Complexity Level Breakdown\n\n"
        md += "| Level | Avg Score | Pass Rate | Prompts |\n"
        md += "|-------|-----------|-----------|----------|\n"
        for level, data in llm_judge_report.get("level_breakdown", {}).items():
            md += f"| {level} | {data['average_score']}/10 | {data['pass_rate']} | {data['prompts_tested']} |\n"
        
        # Human Audit Results
        if human_audit_report:
            md += "\n---\n\n## Human Audit Summary\n\n"
            md += f"| Metric | Value |\n"
            md += f"|--------|-------|\n"
            md += f"| Total Audits | {human_audit_report['summary']['total_audits_completed']} |\n"
            md += f"| Agree Rate | {human_audit_report['summary']['agree_rate']} |\n"
            md += f"| Disagree Rate | {human_audit_report['summary']['disagree_rate']} |\n"
        
        # Comparison Results
        if comparison_report:
            md += "\n---\n\n## Agreement Delta Analysis\n\n"
            md += f"**Overall Agreement Delta:** {comparison_report['agreement_analysis']['overall_delta']}\n\n"
            md += f"**Validation Status:** {comparison_report['validation_status']}\n\n"
            
            md += "### Pillar Agreement\n\n"
            md += "| Pillar | Delta | Rating |\n"
            md += "|--------|-------|--------|\n"
            for pillar, data in comparison_report.get("pillar_agreement", {}).items():
                md += f"| {pillar} | {data['delta']} | {data['rating']} |\n"
        
        # Recommendations
        md += "\n---\n\n## Recommendations\n\n"
        for i, rec in enumerate(llm_judge_report.get("recommendations", []), 1):
            md += f"{i}. {rec}\n"
        
        return md
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score."""
        if score >= 9:
            return "A+"
        elif score >= 8:
            return "A"
        elif score >= 7:
            return "B"
        elif score >= 6:
            return "C"
        elif score >= 5:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, pillar_scores: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on pillar scores."""
        recommendations = []
        
        for pillar, scores in pillar_scores.items():
            avg_score = scores.get("average_score", 0)
            
            if avg_score < 5:
                recommendations.append(
                    f"Critical: {pillar} shows significant weaknesses (score: {avg_score}/10). "
                    f"Requires immediate attention and improvement."
                )
            elif avg_score < 7:
                recommendations.append(
                    f"{pillar} needs improvement (score: {avg_score}/10). "
                    f"Consider targeted training or system enhancements."
                )
            else:
                recommendations.append(
                    f"{pillar} performs well (score: {avg_score}/10). "
                    f"Maintain performance and continue monitoring."
                )
        
        return recommendations
    
    def _determine_validation_status(self, agreement_delta: Dict[str, Any]) -> str:
        """Determine overall validation status."""
        delta = agreement_delta.get("agreement_delta", 0)
        
        if delta >= 0.8:
            return "VALIDATED - Judge shows excellent human alignment"
        elif delta >= 0.6:
            return "MOSTLY VALIDATED - Judge correlation is acceptable"
        elif delta >= 0.4:
            return "PARTIALLY VALIDATED - Some divergence observed"
        else:
            return "NEEDS REVIEW - Significant disagreement with human evaluation"
