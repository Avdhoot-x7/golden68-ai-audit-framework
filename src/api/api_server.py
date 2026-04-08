"""
Golden 68 - API Server for CI/CD Integration
REST API for automated model evaluation
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.adapters import ModelAdapterFactory
from src.evaluation.loader import DatasetLoader
from src.judges.llm_judge import LLMJudge


class EvaluationAPI:
    """API handler for evaluation endpoints."""
    
    def __init__(self, judge_key: str, test_key: str, judge_model: str, test_model: str, provider: str):
        self.judge_key = judge_key
        self.test_key = test_key
        self.judge_model = judge_model
        self.test_model = test_model
        self.provider = provider
        
        # Initialize components
        self.judge_adapter = ModelAdapterFactory.create("gemini", judge_key, judge_model)
        self.test_adapter = ModelAdapterFactory.create(provider, test_key, test_model)
        self.llm_judge = LLMJudge(self.judge_adapter)
        self.loader = DatasetLoader()
    
    def evaluate_prompt(self, prompt: str, expected_behavior: str = "") -> Dict[str, Any]:
        """Evaluate a single prompt."""
        # Get model response
        response = self.test_adapter.generate(prompt, temperature=0.7)
        
        # Judge evaluation
        evaluation = self.llm_judge.evaluate(prompt, response, expected_behavior)
        
        return {
            "prompt": prompt,
            "response": response,
            "evaluation": evaluation,
            "timestamp": datetime.now().isoformat()
        }
    
    def evaluate_dataset(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate full dataset or limited subset."""
        prompts = self.loader.get_all_prompts()
        if limit:
            prompts = prompts[:limit]
        
        results = []
        for p in prompts:
            result = self.evaluate_prompt(
                p.get("prompt", ""),
                p.get("expected_behavior", "")
            )
            results.append({
                "prompt_id": p.get("id"),
                "pillar": p.get("pillar"),
                "level": p.get("level"),
                **result["evaluation"]
            })
        
        # Calculate summary
        scores = [r.get("score", 0) for r in results]
        passes = sum(1 for r in results if r.get("determination") == "PASS")
        
        return {
            "total": len(results),
            "completed": len(results),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "pass_count": passes,
            "pass_rate": passes / len(results) * 100 if results else 0,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get leaderboard from all historical evaluations."""
        results_dir = "data/results"
        if not os.path.exists(results_dir):
            return []
        
        model_scores = {}
        
        for filename in os.listdir(results_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            model = data.get('test_model', 'unknown')
            results = data.get('results', [])
            
            if model not in model_scores:
                model_scores[model] = []
            model_scores[model].extend([r.get('judge_score', 0) for r in results])
        
        leaderboard = []
        for model, scores in model_scores.items():
            passes = sum(1 for s in scores if s >= 7)
            avg = sum(scores) / len(scores) if scores else 0
            leaderboard.append({
                "model": model,
                "average_score": round(avg, 2),
                "pass_rate": round(passes / len(scores) * 100, 1) if scores else 0,
                "total_evaluations": len(scores)
            })
        
        leaderboard.sort(key=lambda x: x["average_score"], reverse=True)
        
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1
        
        return leaderboard


class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the API."""
    
    api: Optional[EvaluationAPI] = None
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        if path == "/health":
            self._send_json({"status": "healthy", "timestamp": datetime.now().isoformat()})
        
        elif path == "/leaderboard":
            if self.api:
                leaderboard = self.api.get_leaderboard()
                self._send_json({"leaderboard": leaderboard})
            else:
                self._send_json({"error": "API not initialized"}, status=500)
        
        elif path == "/evaluate":
            if self.api:
                limit = int(query.get("limit", [66])[0])
                results = self.api.evaluate_dataset(limit=limit)
                self._send_json(results)
            else:
                self._send_json({"error": "API not initialized"}, status=500)
        
        elif path == "/prompts":
            if self.api:
                prompts = self.api.loader.get_all_prompts()
                self._send_json({"count": len(prompts), "prompts": prompts[:10]})
            else:
                self._send_json({"error": "API not initialized"}, status=500)
        
        else:
            self._send_json({
                "name": "Golden 68 API",
                "version": "1.0",
                "endpoints": ["/health", "/leaderboard", "/evaluate", "/evaluate?limit=10", "/prompts"]
            })
    
    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return
        
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == "/evaluate/single":
            if self.api:
                prompt = data.get("prompt", "")
                expected = data.get("expected_behavior", "")
                result = self.api.evaluate_prompt(prompt, expected)
                self._send_json(result)
            else:
                self._send_json({"error": "API not initialized"}, status=500)
        
        else:
            self._send_json({"error": "Endpoint not found"}, status=404)
    
    def _send_json(self, data: Dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.endheaders()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Custom logging."""
        print(f"[API] {args[0]}")


def run_server(host: str = "0.0.0.0", port: int = 8080, 
               judge_key: str = "", test_key: str = "",
               judge_model: str = "gemini-2.5-flash",
               test_model: str = "openai/gpt-oss-120b",
               provider: str = "nvidia"):
    """Run the API server."""
    
    # Initialize API
    api = EvaluationAPI(judge_key, test_key, judge_model, test_model, provider)
    APIHandler.api = api
    
    server = HTTPServer((host, port), APIHandler)
    print(f"🎯 Golden 68 API Server running at http://{host}:{port}")
    print(f"   - GET  /health         - Health check")
    print(f"   - GET  /leaderboard    - Model rankings")
    print(f"   - GET  /evaluate      - Run full evaluation")
    print(f"   - POST /evaluate/single - Evaluate single prompt")
    print(f"\nPress Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped")
        server.shutdown()


# CLI for running the API
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golden 68 API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--judge-key", required=True, help="Judge API key")
    parser.add_argument("--test-key", required=True, help="Test model API key")
    parser.add_argument("--judge-model", default="gemini-2.5-flash", help="Judge model")
    parser.add_argument("--test-model", default="openai/gpt-oss-120b", help="Test model")
    parser.add_argument("--provider", default="nvidia", help="Test model provider")
    
    args = parser.parse_args()
    run_server(
        host=args.host,
        port=args.port,
        judge_key=args.judge_key,
        test_key=args.test_key,
        judge_model=args.judge_model,
        test_model=args.test_model,
        provider=args.provider
    )
