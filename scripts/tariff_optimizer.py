"""Pareto-optimal LLM model routing and cascading tariff cost optimizer.

Calculates multi-objective non-dominated Pareto frontiers across Quality,
Latency, and Token Tariff Cost (measured in fixed-point nano-dollars).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

NANO_FACTOR = 1_000_000_000  # 1 USD = 10^9 nano-dollars


@dataclass(frozen=True)
class ModelCandidate:
    model_id: str
    quality_score: float  # [0.0, 1.0]
    p95_latency_ms: float
    input_rate_per_m_nanos: int
    output_rate_per_m_nanos: int
    cache_read_rate_per_m_nanos: int = 0
    min_complexity_supported: float = 0.0


@dataclass(frozen=True)
class OptimizationGoal:
    task_complexity: float  # [0.0, 1.0]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    max_latency_ms: Optional[float] = None
    max_budget_nanos: Optional[int] = None
    min_quality: float = 0.0


@dataclass(frozen=True)
class OptimizationReport:
    selected_model: str
    estimated_cost_nanos: int
    estimated_latency_ms: float
    quality_score: float
    is_pareto_optimal: bool
    savings_vs_frontier_peak_percent: float
    pareto_candidates: List[str]
    cascade_fallback_model: Optional[str] = None


def calculate_token_cost_nanos(
    candidate: ModelCandidate,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> int:
    """Exact integer calculation of cost in nano-dollars."""
    if input_tokens < 0 or output_tokens < 0 or cache_read_tokens < 0:
        raise ValueError("Token counts cannot be negative")

    in_cost = (input_tokens * candidate.input_rate_per_m_nanos) // 1_000_000
    out_cost = (output_tokens * candidate.output_rate_per_m_nanos) // 1_000_000
    cache_cost = (cache_read_tokens * candidate.cache_read_rate_per_m_nanos) // 1_000_000
    return in_cost + out_cost + cache_cost


def compute_pareto_frontier(
    candidates: List[ModelCandidate],
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> List[ModelCandidate]:
    """Extract non-dominated Pareto frontier candidates.

    Candidate A dominates Candidate B iff:
      A.quality >= B.quality,
      A.latency <= B.latency,
      A.cost <= B.cost,
      with at least one strict inequality.
    """
    if not candidates:
        return []

    costs = [
        calculate_token_cost_nanos(c, input_tokens, output_tokens, cache_read_tokens)
        for c in candidates
    ]

    frontier: List[ModelCandidate] = []
    n = len(candidates)

    for i in range(n):
        c_i = candidates[i]
        cost_i = costs[i]
        dominated = False

        for j in range(n):
            if i == j:
                continue
            c_j = candidates[j]
            cost_j = costs[j]

            if (
                c_j.quality_score >= c_i.quality_score
                and c_j.p95_latency_ms <= c_i.p95_latency_ms
                and cost_j <= cost_i
            ):
                if (
                    c_j.quality_score > c_i.quality_score
                    or c_j.p95_latency_ms < c_i.p95_latency_ms
                    or cost_j < cost_i
                ):
                    dominated = True
                    break

        if not dominated:
            frontier.append(c_i)

    return sorted(frontier, key=lambda c: c.quality_score)


class TariffOptimizer:
    """Multi-model router and tariff cost optimizer."""

    def __init__(self, candidates: List[ModelCandidate]) -> None:
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        self.candidates = list(candidates)

    def route(self, goal: OptimizationGoal) -> OptimizationReport:
        if not (0.0 <= goal.task_complexity <= 1.0):
            raise ValueError(f"Task complexity must be within [0.0, 1.0], got {goal.task_complexity}")
        if not (0.0 <= goal.min_quality <= 1.0):
            raise ValueError(f"min_quality must be within [0.0, 1.0], got {goal.min_quality}")

        frontier = compute_pareto_frontier(
            self.candidates,
            goal.input_tokens,
            goal.output_tokens,
            goal.cache_read_tokens,
        )
        pareto_ids = [c.model_id for c in frontier]

        eligible = []
        for c in self.candidates:
            if c.quality_score < goal.min_quality:
                continue
            if c.quality_score < goal.task_complexity:
                continue
            if goal.max_latency_ms is not None and c.p95_latency_ms > goal.max_latency_ms:
                continue
            cost = calculate_token_cost_nanos(
                c, goal.input_tokens, goal.output_tokens, goal.cache_read_tokens
            )
            if goal.max_budget_nanos is not None and cost > goal.max_budget_nanos:
                continue
            eligible.append((c, cost))

        if not eligible:
            best_c = max(self.candidates, key=lambda c: c.quality_score)
            cost = calculate_token_cost_nanos(
                best_c, goal.input_tokens, goal.output_tokens, goal.cache_read_tokens
            )
            return OptimizationReport(
                selected_model=best_c.model_id,
                estimated_cost_nanos=cost,
                estimated_latency_ms=best_c.p95_latency_ms,
                quality_score=best_c.quality_score,
                is_pareto_optimal=best_c.model_id in pareto_ids,
                savings_vs_frontier_peak_percent=0.0,
                pareto_candidates=pareto_ids,
                cascade_fallback_model=None,
            )

        pareto_eligible = [item for item in eligible if item[0].model_id in pareto_ids]
        pool = pareto_eligible if pareto_eligible else eligible
        selected, selected_cost = min(pool, key=lambda item: item[1])

        frontier_costs = [
            calculate_token_cost_nanos(c, goal.input_tokens, goal.output_tokens, goal.cache_read_tokens)
            for c in (frontier or self.candidates)
        ]
        peak_cost = max(frontier_costs) if frontier_costs else selected_cost
        savings = 0.0
        if peak_cost > 0:
            savings = max(0.0, float((peak_cost - selected_cost) / peak_cost) * 100.0)

        higher_quality = [c for c, _ in eligible if c.quality_score > selected.quality_score]
        cascade_fallback = min(higher_quality, key=lambda c: c.p95_latency_ms).model_id if higher_quality else None

        return OptimizationReport(
            selected_model=selected.model_id,
            estimated_cost_nanos=selected_cost,
            estimated_latency_ms=selected.p95_latency_ms,
            quality_score=selected.quality_score,
            is_pareto_optimal=selected.model_id in pareto_ids,
            savings_vs_frontier_peak_percent=round(savings, 2),
            pareto_candidates=pareto_ids,
            cascade_fallback_model=cascade_fallback,
        )
