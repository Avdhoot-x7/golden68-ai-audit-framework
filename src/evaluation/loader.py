import json
from typing import List, Dict, Any, Optional

from src.constants import DEFAULT_BENCHMARK_DATASET_PATH, PILLAR_SET


class DatasetLoader:
    """Load and manage the Golden 68 benchmark dataset only."""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or str(DEFAULT_BENCHMARK_DATASET_PATH)
        self.dataset = self._load_dataset()
        
        # Sync dataset to vector store for RAG
        try:
            from src.database.vector_store import get_vector_store
            db = get_vector_store()
            db.sync_dataset(self.dataset_path)
        except Exception as e:
            print(f"Vector DB Sync Warning: {e}")
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load the benchmark dataset from JSON and validate pillar consistency."""
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if isinstance(dataset, list):
            dataset = {"prompts": dataset}

        prompts = dataset.get("prompts", [])
        invalid_pillars = sorted({p.get("pillar", "unknown") for p in prompts if p.get("pillar") not in PILLAR_SET})
        if invalid_pillars:
            raise ValueError(f"Dataset contains unsupported pillars: {', '.join(invalid_pillars)}")

        return dataset
    
    def get_all_prompts(self) -> List[Dict[str, Any]]:
        """Get all prompts from the dataset."""
        return self.dataset.get("prompts", [])
    
    def get_prompts_by_pillar(self, pillar: str) -> List[Dict[str, Any]]:
        """Get prompts filtered by one of the configured benchmark pillars."""
        return [p for p in self.get_all_prompts() if p.get("pillar") == pillar]
    
    def get_prompts_by_level(self, level: int) -> List[Dict[str, Any]]:
        """Get prompts filtered by complexity level (1-5)."""
        return [p for p in self.get_all_prompts() if p.get("level") == level]
    
    def get_prompts_by_pillar_and_level(
        self, 
        pillar: str, 
        level: int
    ) -> List[Dict[str, Any]]:
        """Get prompts filtered by both pillar and level."""
        return [
            p for p in self.get_all_prompts() 
            if p.get("pillar") == pillar and p.get("level") == level
        ]
    
    def get_prompt_by_id(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific prompt by its ID."""
        for prompt in self.get_all_prompts():
            if prompt.get("id") == prompt_id:
                return prompt
        return None
    
    def get_filtered_prompts(
        self,
        pillars: List[str] = None,
        levels: List[int] = None,
        categories: List[str] = None,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """Get prompts with optional filtering."""
        prompts = self.get_all_prompts()
        
        if pillars:
            prompts = [p for p in prompts if p.get("pillar") in pillars]
        
        if levels:
            prompts = [p for p in prompts if p.get("level") in levels]
        
        if categories:
            prompts = [p for p in prompts if p.get("category") in categories]
        
        if limit:
            prompts = prompts[:limit]
        
        return prompts
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        prompts = self.get_all_prompts()
        
        stats = {
            "total_prompts": len(prompts),
            "by_pillar": {},
            "by_level": {},
            "by_category": {}
        }
        
        for prompt in prompts:
            # By pillar
            pillar = prompt.get("pillar", "unknown")
            stats["by_pillar"][pillar] = stats["by_pillar"].get(pillar, 0) + 1
            
            # By level
            level = prompt.get("level", 0)
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
            
            # By category
            category = prompt.get("category", "unknown")
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
        
        return stats
    
    def get_pillar_names(self) -> List[str]:
        """Get the benchmark pillar names present in the dataset."""
        return sorted(set(p.get("pillar") for p in self.get_all_prompts()))
    
    def get_level_range(self) -> range:
        """Get the range of complexity levels."""
        levels = set(p.get("level") for p in self.get_all_prompts())
        if levels:
            return range(min(levels), max(levels) + 1)
        return range(1, 6)
