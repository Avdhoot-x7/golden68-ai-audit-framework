"""
Golden 68 - Dataset Loader
Loads and filters the Golden 68 dataset
"""

import json
import os
from typing import List, Dict, Any, Optional


class DatasetLoader:
    """Loads and manages the Golden 68 dataset."""
    
    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            dataset_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "data", "dataset", "golden68.json"
            )
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset()
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load the dataset from JSON file."""
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_all_prompts(self) -> List[Dict[str, Any]]:
        """Get all prompts from the dataset."""
        return self.dataset.get("prompts", [])
    
    def get_prompts_by_pillar(self, pillar: str) -> List[Dict[str, Any]]:
        """Get prompts filtered by pillar (Causality, Compliance, Consistency)."""
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
        """Get all pillar names in the dataset."""
        return list(set(p.get("pillar") for p in self.get_all_prompts()))
    
    def get_level_range(self) -> range:
        """Get the range of complexity levels."""
        levels = set(p.get("level") for p in self.get_all_prompts())
        if levels:
            return range(min(levels), max(levels) + 1)
        return range(1, 6)
