"""Cohen's Kappa utilities for future judge reliability validation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

from src.constants import RELIABILITY_PLACEHOLDER


def build_reliability_placeholder() -> Dict[str, Any]:
    """Return the standard reliability structure with Cohen's Kappa reserved."""
    return dict(RELIABILITY_PLACEHOLDER)


def calculate_cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> Dict[str, Any]:
    """Calculate Cohen's Kappa for two aligned categorical label sequences."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return build_reliability_placeholder()

    total = len(labels_a)
    observed_agreement = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / total

    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    categories = set(counts_a) | set(counts_b)

    expected_agreement = sum((counts_a[category] / total) * (counts_b[category] / total) for category in categories)

    if expected_agreement == 1:
        kappa = 1.0
    else:
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return {
        "metric": RELIABILITY_PLACEHOLDER["metric"],
        "value": round(kappa, 3),
    }


def calculate_kappa_from_judge_and_human_records(
    judge_records: Iterable[Dict[str, Any]],
    human_records: Iterable[Dict[str, Any]],
    pass_threshold: int = 6,
) -> Dict[str, Any]:
    """Calculate Cohen's Kappa using PASS/FAIL labels derived from judge and human records."""
    judge_by_prompt = {record.get("prompt_id"): record for record in judge_records if record.get("prompt_id")}
    human_by_prompt = {record.get("prompt_id"): record for record in human_records if record.get("prompt_id")}

    shared_prompt_ids: List[str] = sorted(set(judge_by_prompt) & set(human_by_prompt))
    if not shared_prompt_ids:
        return build_reliability_placeholder()

    judge_labels = []
    human_labels = []

    for prompt_id in shared_prompt_ids:
        judge_record = judge_by_prompt[prompt_id]
        human_record = human_by_prompt[prompt_id]

        judge_label = judge_record.get("judge_determination")
        if not judge_label:
            judge_label = "PASS" if judge_record.get("judge_score", 0) >= pass_threshold else "FAIL"

        human_score = human_record.get("human_score")
        if human_score is None:
            continue

        human_label = "PASS" if human_score >= pass_threshold else "FAIL"
        judge_labels.append(judge_label)
        human_labels.append(human_label)

    if not judge_labels:
        return build_reliability_placeholder()

    return calculate_cohens_kappa(judge_labels, human_labels)
