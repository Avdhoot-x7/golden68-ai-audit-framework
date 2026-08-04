"""Aggregation helpers for evaluation reporting and heatmap generation."""

from __future__ import annotations

from typing import Any, Dict, List

from src.constants import PILLARS
from src.validation import build_reliability_placeholder


def summarize_evaluations(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the minimal, explicit summary used across the framework."""
    scores = [item.get("judge_score", 0) for item in evaluations]
    pass_count = sum(1 for item in evaluations if item.get("judge_determination") == "PASS")
    total = len(evaluations)

    return {
        "total": total,
        "average_score": round(sum(scores) / total, 2) if total else 0.0,
        "pass_rate": round(pass_count / total, 3) if total else 0.0,
        "reliability": build_reliability_placeholder(),
    }


def aggregate_by_pillar(evaluations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate scores by the configured pillars."""
    aggregates = {
        pillar: {"scores": [], "passes": 0, "total": 0, "reliability": build_reliability_placeholder()}
        for pillar in PILLARS
    }

    for item in evaluations:
        pillar = item.get("pillar")
        if pillar not in aggregates:
            continue

        score = item.get("judge_score", 0)
        aggregates[pillar]["scores"].append(score)
        aggregates[pillar]["total"] += 1
        if item.get("judge_determination") == "PASS":
            aggregates[pillar]["passes"] += 1

    for pillar, data in aggregates.items():
        total = data["total"]
        scores = data.pop("scores")
        data["average_score"] = round(sum(scores) / total, 2) if total else 0.0
        data["pass_rate"] = round(data["passes"] / total, 3) if total else 0.0

    return aggregates


def aggregate_by_level(evaluations: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Aggregate scores by prompt complexity level."""
    grouped: Dict[int, Dict[str, Any]] = {}

    for item in evaluations:
        level = item.get("level", 0)
        if level not in grouped:
            grouped[level] = {
                "scores": [],
                "passes": 0,
                "total": 0,
                "reliability": build_reliability_placeholder(),
            }

        grouped[level]["scores"].append(item.get("judge_score", 0))
        grouped[level]["total"] += 1
        if item.get("judge_determination") == "PASS":
            grouped[level]["passes"] += 1

    for level, data in grouped.items():
        total = data["total"]
        scores = data.pop("scores")
        data["average_score"] = round(sum(scores) / total, 2) if total else 0.0
        data["pass_rate"] = round(data["passes"] / total, 3) if total else 0.0

    return dict(sorted(grouped.items()))


def build_heatmap_data(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the heatmap matrix data used as the primary output artifact."""
    levels = sorted({item.get("level", 0) for item in evaluations})
    z_matrix: List[List[float]] = []
    text_matrix: List[List[str]] = []

    for pillar in PILLARS:
        row_scores = []
        row_text = []

        for level in levels:
            matches = [
                item.get("judge_score", 0)
                for item in evaluations
                if item.get("pillar") == pillar and item.get("level") == level
            ]
            avg_score = round(sum(matches) / len(matches), 2) if matches else 0.0
            row_scores.append(avg_score)
            row_text.append(f"{avg_score:.2f}" if matches else "")

        z_matrix.append(row_scores)
        text_matrix.append(row_text)

    return {
        "pillars": list(PILLARS),
        "levels": levels,
        "z": z_matrix,
        "text": text_matrix,
        "reliability": build_reliability_placeholder(),
    }
