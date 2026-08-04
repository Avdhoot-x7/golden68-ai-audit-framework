"""Generate minimal reports centered on score summaries and heatmap data."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.constants import REPORTS_DIR
from src.reporting.aggregation import build_heatmap_data


class ReportGenerator:
    """Generate lightweight reports without validation or composite metrics."""

    def __init__(self, reports_dir: str | None = None):
        self.reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_llm_judge_report(self, model_name: str, result_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Build the report payload for one evaluation run."""
        return {
            "report_type": "LLM_JUDGE_REPORT",
            "model_name": model_name,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_prompts_evaluated": result_bundle.get("total", 0),
                "average_score": result_bundle.get("average_score", 0.0),
                "pass_rate": result_bundle.get("pass_rate", 0.0),
                "reliability": result_bundle.get("reliability", {"metric": None, "value": None}),
            },
            "pillar_breakdown": result_bundle.get("pillar_scores", {}),
            "level_breakdown": result_bundle.get("level_scores", {}),
            "heatmap": build_heatmap_data(result_bundle.get("evaluations", [])),
            "detailed_evaluations": result_bundle.get("evaluations", []),
        }

    def save_report(self, report: Dict[str, Any], prefix: str = "report") -> str:
        """Save a report payload to disk and log to vector database."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.reports_dir / f"{prefix}_{timestamp}.json"
        filepath.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        
        # Log to Vector Store
        try:
            from src.database.vector_store import get_vector_store
            db = get_vector_store()
            evaluations = report.get("detailed_evaluations", [])
            
            # Map to format expected by log_evaluations
            eval_data_list = []
            for ev in evaluations:
                eval_data = {
                    "session_id": timestamp,
                    "prompt_id": ev.get("prompt", {}).get("id", "unknown"),
                    "prompt_text": ev.get("prompt", {}).get("prompt", ""),
                    "model_name": report.get("model_name", "unknown"),
                    "provider": report.get("provider", "unknown"),  
                    "score": ev.get("judge_result", {}).get("score", 0),
                    "rationale": ev.get("judge_result", {}).get("rationale", ""),
                    "response_text": ev.get("model_response", "")
                }
                eval_data_list.append(eval_data)
                
            if eval_data_list:
                db.log_evaluations(eval_data_list)
        except Exception as e:
            print(f"Vector DB Logging Error: {e}")

        return str(filepath)

    def generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate a markdown summary with the heatmap as the central artifact."""
        summary = report.get("summary", {})
        heatmap = report.get("heatmap", {})

        md = [
            "# Golden 68 Evaluation Report",
            "",
            f"**Model:** {report.get('model_name', 'Unknown')}",
            f"**Generated:** {report.get('generated_at', datetime.now().isoformat())}",
            "",
            "## Summary",
            "",
            f"- Total prompts evaluated: {summary.get('total_prompts_evaluated', 0)}",
            f"- Average score: {summary.get('average_score', 0.0):.2f}/10",
            f"- Pass rate: {summary.get('pass_rate', 0.0) * 100:.1f}%",
            f"- Reliability metric: {summary.get('reliability', {}).get('metric')}",
            f"- Reliability value: {summary.get('reliability', {}).get('value')}",
            f"- Human audits: {len(report.get('human_audits', []))}",
            "",
            "## Heatmap",
            "",
            f"- Pillars: {', '.join(heatmap.get('pillars', []))}",
            f"- Levels: {', '.join(str(level) for level in heatmap.get('levels', []))}",
        ]

        return "\n".join(md)

    def generate_pdf_report(self, report: Dict[str, Any], prefix: str = "report") -> str:
        """Generate a styled PDF report using Markdown and xhtml2pdf."""
        import markdown
        from xhtml2pdf import pisa
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.reports_dir / f"{prefix}_{timestamp}.pdf"
        
        md_content = self.generate_markdown_report(report)
        
        # Sanitize text for xhtml2pdf to prevent UnicodeEncodeError crashes
        md_content = md_content.encode("ascii", "ignore").decode("ascii")
        
        # Inject RAG historical comparison if possible
        try:
            from src.rag.pipeline import RAGPipeline
            rag = RAGPipeline()
            model_name = report.get('model_name', '')
            if model_name:
                # Query historical performance for this model
                hist_results = rag.query_evaluations(query_text=model_name, n_results=50, where={"model_name": model_name})
                if hist_results:
                    scores = [res.get("metadata", {}).get("score", 0) for res in hist_results if "metadata" in res]
                    if scores:
                        hist_avg = sum(scores) / len(scores)
                        md_content += f"\n\n## Historical RAG Context\n\n"
                        md_content += f"Historically, `{model_name}` has an average score of **{hist_avg:.2f}/10** across {len(scores)} recorded evaluations in the vector database.\n"
                        current_avg = report.get("summary", {}).get("average_score", 0.0)
                        if current_avg > hist_avg:
                            md_content += f"This run (**{current_avg:.2f}**) shows an **improvement** over the historical average.\n"
                        else:
                            md_content += f"This run (**{current_avg:.2f}**) is **below or equal** to the historical average.\n"
        except Exception as e:
            print(f"Warning: RAG historical context failed during PDF generation: {e}")
            
        # Add a section for critical failures (score < 10)
        detailed_evals = report.get("detailed_evaluations", [])
        failures = [ev for ev in detailed_evals if ev.get("judge_result", {}).get("score", 10.0) < 10.0]
        
        if failures:
            md_content += "\n## Critical Failures & Imperfections\n\n"
            for fail in failures[:5]:  # Limit to top 5 in PDF to save space
                score = fail.get("judge_result", {}).get("score", 0)
                md_content += f"### Score: {score}/10\n"
                md_content += f"**Prompt:** {fail.get('prompt', {}).get('prompt', '')}\n\n"
                md_content += f"**Model Response:** {fail.get('model_response', '')[:200]}...\n\n"
                md_content += f"**Judge Rationale:** {fail.get('judge_result', {}).get('rationale', '')}\n\n"
                md_content += "---\n"
                
        # Convert to HTML
        html_content = markdown.markdown(md_content)
        
        # Add basic CSS styling
        styled_html = f"""
        <html>
        <head>
        <style>
            @page {{ size: a4; margin: 2cm; }}
            body {{ font-family: Helvetica, Arial, sans-serif; font-size: 12pt; line-height: 1.5; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
            h2 {{ color: #2980b9; margin-top: 20px; }}
            h3 {{ color: #e74c3c; font-size: 14pt; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #bdc3c7; padding: 8px; text-align: left; }}
            th {{ background-color: #ecf0f1; }}
            code {{ background-color: #f8f9fa; padding: 2px 4px; border-radius: 4px; font-family: monospace; }}
            .highlight {{ background-color: #fef9e7; padding: 10px; border-left: 4px solid #f1c40f; margin-bottom: 10px; }}
        </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        with open(filepath, "w+b") as result_file:
            pisa_status = pisa.CreatePDF(styled_html, dest=result_file)
            
        if pisa_status.err:
            raise Exception("PDF generation failed.")
            
        return str(filepath)
