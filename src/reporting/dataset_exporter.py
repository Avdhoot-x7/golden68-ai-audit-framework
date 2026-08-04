import json
import os
from pathlib import Path
from typing import List, Dict, Any

class DatasetExporter:
    """Exports failed evaluations into actionable fine-tuning datasets (e.g. RLHF/DPO)."""

    def __init__(self, export_dir: str = "data/fine_tuning"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_failures(self, model_name: str, detailed_evaluations: List[Dict[str, Any]], threshold: float = 10.0) -> str:
        """
        Extracts any evaluation with a score less than the threshold and exports it
        as a JSONL file specifically formatted for DPO / RLHF fine-tuning.
        """
        export_path = self.export_dir / f"{model_name}_failures.jsonl"
        
        failures_found = 0
        with open(export_path, 'w', encoding='utf-8') as f:
            for ev in detailed_evaluations:
                score = ev.get("judge_result", {}).get("score", 10.0)
                if score < threshold:
                    failures_found += 1
                    
                    # DPO/RLHF format
                    # prompt: The original question
                    # rejected: The bad model response
                    # rationale: Why the judge gave it a low score (used for reasoning tuning)
                    row = {
                        "prompt": ev.get("prompt", {}).get("prompt", ""),
                        "rejected": ev.get("model_response", ""),
                        "rationale": ev.get("judge_result", {}).get("rationale", ""),
                        "score": score
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    
        if failures_found == 0:
            # Clean up the empty file if no failures were found
            if export_path.exists():
                export_path.unlink()
            return ""
            
        return str(export_path)
