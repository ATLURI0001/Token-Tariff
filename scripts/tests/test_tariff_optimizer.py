import pytest

from scripts.tariff_optimizer import (
    ModelCandidate,
    OptimizationGoal,
    TariffOptimizer,
    calculate_token_cost_nanos,
    compute_pareto_frontier,
)


@pytest.fixture
def standard_models():
    return [
        ModelCandidate(
            model_id="claude-3-5-haiku",
            quality_score=0.75,
            p95_latency_ms=250.0,
            input_rate_per_m_nanos=800_000_000,   # $0.80 / M
            output_rate_per_m_nanos=4_000_000_000, # $4.00 / M
            cache_read_rate_per_m_nanos=80_000_000, # $0.08 / M
        ),
        ModelCandidate(
            model_id="claude-3-5-sonnet",
            quality_score=0.92,
            p95_latency_ms=650.0,
            input_rate_per_m_nanos=3_000_000_000,  # $3.00 / M
            output_rate_per_m_nanos=15_000_000_000, # $15.00 / M
            cache_read_rate_per_m_nanos=300_000_000, # $0.30 / M
        ),
        ModelCandidate(
            model_id="claude-3-opus",
            quality_score=0.96,
            p95_latency_ms=1400.0,
            input_rate_per_m_nanos=15_000_000_000, # $15.00 / M
            output_rate_per_m_nanos=75_000_000_000, # $75.00 / M
            cache_read_rate_per_m_nanos=1_500_000_000, # $1.50 / M
        ),
        ModelCandidate(
            model_id="inefficient-legacy-model",
            quality_score=0.70,  # lower quality than Haiku
            p95_latency_ms=800.0,  # higher latency than Haiku
            input_rate_per_m_nanos=5_000_000_000,  # higher cost than Haiku
            output_rate_per_m_nanos=20_000_000_000,
        ),
    ]


def test_calculate_token_cost_nanos(standard_models):
    haiku = standard_models[0]
    # 10,000 in, 2,000 out, 5,000 cache read
    cost = calculate_token_cost_nanos(haiku, 10_000, 2_000, 5_000)
    # in: 10_000 * 800_000_000 // 1_000_000 = 8_000_000 nanos ($0.008)
    # out: 2_000 * 4_000_000_000 // 1_000_000 = 8_000_000 nanos ($0.008)
    # cache: 5_000 * 80_000_000 // 1_000_000 = 400_000 nanos ($0.0004)
    # total: 16_400_000 nanos ($0.0164)
    assert cost == 16_400_000

    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_token_cost_nanos(haiku, -1, 100)


def test_compute_pareto_frontier_filters_dominated(standard_models):
    frontier = compute_pareto_frontier(standard_models, 10_000, 2_000)
    frontier_ids = [c.model_id for c in frontier]

    # inefficient-legacy-model is strictly dominated by Haiku across quality, latency, and cost
    assert "inefficient-legacy-model" not in frontier_ids
    assert "claude-3-5-haiku" in frontier_ids
    assert "claude-3-5-sonnet" in frontier_ids
    assert "claude-3-opus" in frontier_ids


def test_tariff_optimizer_routes_low_complexity_to_cheapest_pareto(standard_models):
    optimizer = TariffOptimizer(standard_models)

    goal = OptimizationGoal(
        task_complexity=0.50,
        input_tokens=5_000,
        output_tokens=1_000,
    )
    report = optimizer.route(goal)
    assert report.selected_model == "claude-3-5-haiku"
    assert report.is_pareto_optimal is True
    assert report.savings_vs_frontier_peak_percent > 80.0
    assert report.cascade_fallback_model == "claude-3-5-sonnet"


def test_tariff_optimizer_routes_high_complexity_to_sonnet_or_opus(standard_models):
    optimizer = TariffOptimizer(standard_models)

    goal = OptimizationGoal(
        task_complexity=0.90,
        input_tokens=10_000,
        output_tokens=3_000,
        max_latency_ms=1000.0,  # Rules out Opus (1400ms)
    )
    report = optimizer.route(goal)
    assert report.selected_model == "claude-3-5-sonnet"
    assert report.is_pareto_optimal is True
    assert report.estimated_latency_ms <= 1000.0


def test_tariff_optimizer_budget_and_latency_bounds(standard_models):
    optimizer = TariffOptimizer(standard_models)

    # Tight budget forcing Haiku even if complexity request is borderline
    goal = OptimizationGoal(
        task_complexity=0.70,
        input_tokens=10_000,
        output_tokens=2_000,
        max_budget_nanos=20_000_000,  # $0.02 max budget
    )
    report = optimizer.route(goal)
    assert report.selected_model == "claude-3-5-haiku"
    assert report.estimated_cost_nanos <= 20_000_000


def test_tariff_optimizer_invalid_inputs(standard_models):
    with pytest.raises(ValueError, match="cannot be empty"):
        TariffOptimizer([])

    optimizer = TariffOptimizer(standard_models)
    with pytest.raises(ValueError, match="Task complexity must be within"):
        optimizer.route(OptimizationGoal(task_complexity=1.5, input_tokens=100, output_tokens=100))
