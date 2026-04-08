"""
Cost Tracker and Smart Resume for Golden 68 Framework
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

PROVIDER_LIMITS = {
    "gemini": {"rpm": 15, "tpm": 1000000, "daily_limit": 1500, "cost_per_1k": 0.00015},
    "openai": {"rpm": 500, "tpm": 150000, "daily_limit": 10000, "cost_per_1k": 0.002},
    "openrouter": {"rpm": 200, "tpm": 200000, "daily_limit": 5000, "cost_per_1k": 0.001},
    "anthropic": {"rpm": 50, "tpm": 100000, "daily_limit": 5000, "cost_per_1k": 0.003},
    "nvidia": {"rpm": 120, "tpm": 500000, "daily_limit": 10000, "cost_per_1k": 0.0005}
}

class APICostTracker:
    def __init__(self, data_dir="data/cost_tracking"):
        self.data_dir = data_dir
        self.usage_history = {}
        self.PROVIDER_LIMITS = PROVIDER_LIMITS
        os.makedirs(data_dir, exist_ok=True)
        self._load_history()
    
    def track_request(self, provider, api_key, tokens_used=0, success=True, error_message=""):
        if provider not in self.usage_history:
            self.usage_history[provider] = []
        record = {
            "timestamp": datetime.now().isoformat(),
            "api_key_prefix": api_key[:8] + "..." if len(api_key) > 8 else api_key,
            "tokens_used": tokens_used,
            "success": success,
            "error_message": error_message
        }
        self.usage_history[provider].append(record)
        limits = PROVIDER_LIMITS.get(provider, {})
        total_tokens = sum(r.get("tokens_used", 0) for r in self.usage_history[provider])
        total_requests = len(self.usage_history[provider])
        estimated_cost = (total_tokens / 1000) * limits.get("cost_per_1k", 0)
        self._save_history()
        return {
            "provider": provider,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "estimated_cost": estimated_cost,
            "rpm_limit": limits.get("rpm", "N/A"),
            "tpm_limit": limits.get("tpm", "N/A"),
            "daily_limit": limits.get("daily_limit", "N/A")
        }
    
    def get_usage_summary(self, provider=None):
        if provider:
            records = self.usage_history.get(provider, [])
            limits = PROVIDER_LIMITS.get(provider, {})
            total_tokens = sum(r.get("tokens_used", 0) for r in records)
            total_requests = len(records)
            successful_requests = sum(1 for r in records if r.get("success", True))
            failed_requests = total_requests - successful_requests
            estimated_cost = (total_tokens / 1000) * limits.get("cost_per_1k", 0)
            return {
                "provider": provider,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "total_tokens": total_tokens,
                "estimated_cost": estimated_cost,
                "limits": limits
            }
        else:
            summary = {}
            total_cost = 0
            total_tokens_all = 0
            total_requests_all = 0
            for prov in self.usage_history.keys():
                prov_summary = self.get_usage_summary(prov)
                summary[prov] = prov_summary
                total_cost += prov_summary["estimated_cost"]
                total_tokens_all += prov_summary["total_tokens"]
                total_requests_all += prov_summary["total_requests"]
            return {
                "providers": summary,
                "total_cost": total_cost,
                "total_tokens": total_tokens_all,
                "total_requests": total_requests_all
            }
    
    def _load_history(self):
        history_file = os.path.join(self.data_dir, "usage_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.usage_history = json.load(f)
            except: pass
    
    def _save_history(self):
        history_file = os.path.join(self.data_dir, "usage_history.json")
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.usage_history, f, indent=2, ensure_ascii=False)
        except: pass

class SmartResume:
    def __init__(self, checkpoint_dir="data/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def create_checkpoint(self, checkpoint_id, completed, total, results, config):
        checkpoint_data = {
            "id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "completed": completed,
            "total": total,
            "results": results,
            "config": config
        }
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        return checkpoint_file
    
    def update_checkpoint(self, checkpoint_id, completed, results):
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        if not os.path.exists(checkpoint_file):
            return False
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            checkpoint_data["completed"] = completed
            checkpoint_data["results"] = results
            checkpoint_data["timestamp"] = datetime.now().isoformat()
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            return True
        except:
            return False
    
    def get_checkpoint(self, checkpoint_id):
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        if not os.path.exists(checkpoint_file):
            return None
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    
    def get_pending_prompts(self, checkpoint_id):
        checkpoint = self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            return []
        completed = checkpoint.get("completed", 0)
        total = checkpoint.get("total", 0)
        return list(range(completed, total))
    
    def list_checkpoints(self):
        checkpoints = []
        if not os.path.exists(self.checkpoint_dir):
            return checkpoints
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.json'):
                checkpoint_id = filename.replace('.json', '')
                checkpoint = self.get_checkpoint(checkpoint_id)
                if checkpoint:
                    checkpoints.append({
                        "id": checkpoint_id,
                        "timestamp": checkpoint.get("timestamp", ""),
                        "completed": checkpoint.get("completed", 0),
                        "total": checkpoint.get("total", 0),
                        "config": checkpoint.get("config", {})
                    })
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id):
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
        if not os.path.exists(checkpoint_file):
            return False
        try:
            os.remove(checkpoint_file)
            return True
        except:
            return False
