"""
Golden 68 - Human Audit System
Expert verification interface for auditing LLM evaluations
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class HumanAuditRecord:
    """Record for human expert audit."""
    prompt_id: str
    pillar: str
    level: int
    prompt: str
    model_response: str
    judge_score: int
    judge_determination: str
    judge_reasoning: str
    human_score: Optional[int] = None
    human_reasoning: Optional[str] = None
    human_verdict: Optional[str] = None  # AGREE/DISAGREE/OVERRIDE
    auditor_id: Optional[str] = None
    audit_timestamp: Optional[str] = None
    notes: Optional[str] = None


class HumanAuditManager:
    """Manages human expert audits of LLM evaluations."""
    
    def __init__(self, audit_dir: str = None):
        if audit_dir is None:
            audit_dir = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "data", "audit"
            )
        self.audit_dir = audit_dir
        os.makedirs(self.audit_dir, exist_ok=True)
        self.current_audit = None
    
    def create_audit_record(
        self,
        prompt_id: str,
        pillar: str,
        level: int,
        prompt: str,
        model_response: str,
        judge_result: Dict[str, Any]
    ) -> HumanAuditRecord:
        """Create a new audit record from a judge evaluation."""
        return HumanAuditRecord(
            prompt_id=prompt_id,
            pillar=pillar,
            level=level,
            prompt=prompt,
            model_response=model_response,
            judge_score=judge_result.get("score", 5),
            judge_determination=judge_result.get("determination", "FAIL"),
            judge_reasoning=judge_result.get("explanation", ""),
            audit_timestamp=datetime.now().isoformat()
        )
    
    def submit_audit(
        self,
        record: HumanAuditRecord,
        human_score: int,
        human_reasoning: str,
        auditor_id: str = "anonymous"
    ) -> HumanAuditRecord:
        """Submit a completed human audit."""
        record.human_score = human_score
        record.human_reasoning = human_reasoning
        record.auditor_id = auditor_id
        record.audit_timestamp = datetime.now().isoformat()
        
        # Determine verdict
        score_diff = abs(human_score - record.judge_score)
        if score_diff <= 1:
            record.human_verdict = "AGREE"
        elif score_diff <= 3:
            record.human_verdict = "PARTIAL"
        else:
            record.human_verdict = "DISAGREE"
        
        return record
    
    def save_audit(self, record: HumanAuditRecord, session_id: str = None) -> str:
        """Save audit record to file."""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d")
        
        filename = f"audit_{session_id}.json"
        filepath = os.path.join(self.audit_dir, filename)
        
        # Load existing audits or create new
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                audits = json.load(f)
        else:
            audits = {"session_id": session_id, "audits": []}
        
        # Add new audit
        audits["audits"].append(asdict(record))
        
        # Save back
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(audits, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_audits(self, session_id: str = None) -> List[Dict[str, Any]]:
        """Load audits from a session."""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d")
        
        filepath = os.path.join(self.audit_dir, f"audit_{session_id}.json")
        
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("audits", [])
        
        return []
    
    def get_pending_audits(
        self, 
        judge_results: List[Dict[str, Any]], 
        existing_audit_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get list of evaluations pending human audit."""
        pending = []
        
        for result in judge_results:
            prompt_id = result.get("prompt_id")
            if prompt_id and prompt_id not in existing_audit_ids:
                pending.append(result)
        
        return pending
    
    def get_audit_statistics(self, audits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from audit data."""
        if not audits:
            return {
                "total_audits": 0,
                "agree_count": 0,
                "partial_count": 0,
                "disagree_count": 0,
                "average_human_score": 0,
                "average_judge_score": 0
            }
        
        agree_count = sum(1 for a in audits if a.get("human_verdict") == "AGREE")
        partial_count = sum(1 for a in audits if a.get("human_verdict") == "PARTIAL")
        disagree_count = sum(1 for a in audits if a.get("human_verdict") == "DISAGREE")
        
        human_scores = [a.get("human_score", 0) for a in audits if a.get("human_score")]
        judge_scores = [a.get("judge_score", 0) for a in audits if a.get("judge_score")]
        
        return {
            "total_audits": len(audits),
            "agree_count": agree_count,
            "partial_count": partial_count,
            "disagree_count": disagree_count,
            "agree_rate": agree_count / len(audits) if audits else 0,
            "partial_rate": partial_count / len(audits) if audits else 0,
            "disagree_rate": disagree_count / len(audits) if audits else 0,
            "average_human_score": sum(human_scores) / len(human_scores) if human_scores else 0,
            "average_judge_score": sum(judge_scores) / len(judge_scores) if judge_scores else 0
        }
    
    def export_audit_report(self, audits: List[Dict[str, Any]], output_path: str = None) -> str:
        """Export a formatted audit report."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.audit_dir, f"audit_report_{timestamp}.json")
        
        stats = self.get_audit_statistics(audits)
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "audits": audits
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path
