import os
from typing import List, Dict, Any
from src.database.vector_store import get_vector_store

class RAGPipeline:
    def __init__(self):
        self.db = get_vector_store()

    def query_evaluations(self, query_text: str, n_results: int = 5, where: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Query the historical evaluations collection using semantic search.
        
        Args:
            query_text: Natural language query (e.g., "adversarial coding fails")
            n_results: Number of results to return
            where: Optional metadata filters (e.g., {"model_name": "mixtral", "score": {"$lt": 5}})
        """
        results = self.db.evals_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        return self._format_results(results)

    def query_prompts(self, query_text: str, n_results: int = 5, where: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Query the golden dataset prompts using semantic search.
        """
        results = self.db.prompts_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        return self._format_results(results)

    def _format_results(self, raw_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format ChromaDB raw results into a cleaner list of dictionaries."""
        formatted = []
        if not raw_results or not raw_results.get("documents") or not raw_results["documents"][0]:
            return formatted

        for i in range(len(raw_results["documents"][0])):
            item = {
                "id": raw_results["ids"][0][i] if raw_results.get("ids") else None,
                "document": raw_results["documents"][0][i],
                "metadata": raw_results["metadatas"][0][i] if raw_results.get("metadatas") else {},
                "distance": raw_results["distances"][0][i] if raw_results.get("distances") else None
            }
            formatted.append(item)
            
        return formatted
