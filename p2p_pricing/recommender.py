"""P2P Pricing Mechanism Recommendation Engine.

Main entry point for recommending the best pricing mechanism
based on user inputs and priority weights.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
from .config import SimulationConfig, RecommendationWeights, TariffConfig
from .profiles import generate_all_profiles
from .settlements import run_all_settlements
from .metrics import compute_all_metrics, MetricsReport, MechanismMetrics


@dataclass
class Recommendation:
    """Result of the recommendation engine."""
    recommended_mechanism: str
    reasoning: str
    scores: Dict[str, float]          # Weighted scores for each mechanism
    metrics: MetricsReport            # Full metrics report
    weights_used: RecommendationWeights


def normalize_metric(values: Dict[str, float], higher_is_better: bool = True) -> Dict[str, float]:
    """
    Normalize metric values to [0, 1] range.

    Args:
        values: Dict of mechanism name -> metric value
        higher_is_better: If True, higher values get scores closer to 1
    """
    if not values:
        return {}

    vals = list(values.values())
    min_val = min(vals)
    max_val = max(vals)

    if max_val == min_val:
        return {k: 1.0 for k in values}

    normalized = {}
    for k, v in values.items():
        norm = (v - min_val) / (max_val - min_val)
        if not higher_is_better:
            norm = 1.0 - norm
        normalized[k] = norm

    return normalized


def compute_weighted_scores(
    metrics: MetricsReport,
    weights: RecommendationWeights
) -> Dict[str, float]:
    """
    Compute weighted scores for each mechanism.

    Score = w1*savings_norm + w2*fairness_norm + w3*stability_norm
    where stability = inverse of volatility (lower volatility = higher stability)
    """
    mechanisms = metrics.mechanisms

    # Skip conventional in recommendations (it's the baseline)
    p2p_mechanisms = {k: v for k, v in mechanisms.items() if k != 'Conventional'}

    # Extract raw values
    savings = {k: v.cost_savings_pct for k, v in p2p_mechanisms.items()}
    fairness = {k: v.fairness_index for k, v in p2p_mechanisms.items()}
    volatility = {k: v.daily_cost_volatility for k, v in p2p_mechanisms.items()}

    # Normalize
    savings_norm = normalize_metric(savings, higher_is_better=True)
    fairness_norm = normalize_metric(fairness, higher_is_better=True)
    stability_norm = normalize_metric(volatility, higher_is_better=False)

    # Compute weighted scores
    scores = {}
    for name in p2p_mechanisms:
        scores[name] = (
            weights.cost_savings * savings_norm[name] +
            weights.fairness * fairness_norm[name] +
            weights.stability * stability_norm[name]
        )

    return scores


def generate_reasoning(
    recommended: str,
    metrics: MetricsReport,
    scores: Dict[str, float]
) -> str:
    """Generate human-readable reasoning for the recommendation."""
    m = metrics.mechanisms[recommended]

    lines = [f"**{recommended}** is recommended because:"]

    # Cost savings
    if m.cost_savings_pct > 0:
        lines.append(f"- Achieves {m.cost_savings_pct:.1f}% cost savings vs conventional pricing")
    else:
        lines.append(f"- Cost impact: {m.cost_savings_pct:.1f}% vs conventional")

    # Fairness
    if m.fairness_index >= 0.9:
        lines.append("- Excellent fairness: benefits distributed nearly equally")
    elif m.fairness_index >= 0.7:
        lines.append("- Good fairness: reasonable balance between groups")
    else:
        diff = abs(m.consumer_savings_pct - m.prosumer_savings_pct)
        if m.consumer_savings_pct > m.prosumer_savings_pct:
            lines.append(f"- Favors consumers ({diff:.1f}% more savings)")
        else:
            lines.append(f"- Favors prosumers ({diff:.1f}% more savings)")

    # Stability
    lines.append(f"- Daily cost volatility: Rs {m.daily_cost_volatility:.2f}")

    # Comparison with runner-up
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) > 1:
        runner_up = sorted_scores[1][0]
        runner_up_m = metrics.mechanisms[runner_up]
        lines.append(f"\nAlternative: **{runner_up}** (savings: {runner_up_m.cost_savings_pct:.1f}%)")

    return "\n".join(lines)


def recommend(
    num_consumers: int,
    num_small_prosumers: int,
    num_medium_prosumers: int,
    num_large_prosumers: int,
    weights: RecommendationWeights = None,
    config: SimulationConfig = None,
    seed: int = None
) -> Recommendation:
    """
    Main entry point for P2P pricing mechanism recommendation.

    Args:
        num_consumers: Number of consumer households (no PV)
        num_small_prosumers: Number of small prosumers (40-60% PV)
        num_medium_prosumers: Number of medium prosumers (80-110% PV)
        num_large_prosumers: Number of large prosumers (130-180% PV)
        weights: Priority weights for scoring (uses defaults if None)
        config: Simulation configuration (uses defaults if None)
        seed: Random seed (overrides config if provided)

    Returns:
        Recommendation with best mechanism, reasoning, and detailed metrics
    """
    if weights is None:
        weights = RecommendationWeights()

    if config is None:
        config = SimulationConfig()

    # Generate profiles
    profiles = generate_all_profiles(
        num_consumers=num_consumers,
        num_small_prosumers=num_small_prosumers,
        num_medium_prosumers=num_medium_prosumers,
        num_large_prosumers=num_large_prosumers,
        config=config,
        seed=seed
    )

    # Run all settlement mechanisms
    settlement_results = run_all_settlements(profiles)

    # Compute metrics
    metrics = compute_all_metrics(
        settlement_results,
        profiles['household_types'],
        config.periods_per_day
    )

    # Compute weighted scores
    scores = compute_weighted_scores(metrics, weights)

    # Find best mechanism
    best = max(scores, key=scores.get)

    # Generate reasoning
    reasoning = generate_reasoning(best, metrics, scores)

    return Recommendation(
        recommended_mechanism=best,
        reasoning=reasoning,
        scores=scores,
        metrics=metrics,
        weights_used=weights
    )


# Simplified API for backward compatibility
def recommend_simple(
    num_consumers: int,
    num_prosumers: int,
    prosumer_distribution: str = "balanced"
) -> Recommendation:
    """
    Simplified recommendation API with automatic prosumer distribution.

    Args:
        num_consumers: Number of consumer households
        num_prosumers: Total number of prosumers
        prosumer_distribution: One of "balanced", "mostly_small", "mostly_large"

    Returns:
        Recommendation result
    """
    if prosumer_distribution == "mostly_small":
        # 60% small, 30% medium, 10% large
        num_small = int(num_prosumers * 0.6)
        num_medium = int(num_prosumers * 0.3)
        num_large = num_prosumers - num_small - num_medium
    elif prosumer_distribution == "mostly_large":
        # 20% small, 30% medium, 50% large
        num_small = int(num_prosumers * 0.2)
        num_medium = int(num_prosumers * 0.3)
        num_large = num_prosumers - num_small - num_medium
    else:  # balanced
        # Equal distribution
        num_small = num_prosumers // 3
        num_medium = num_prosumers // 3
        num_large = num_prosumers - num_small - num_medium

    return recommend(
        num_consumers=num_consumers,
        num_small_prosumers=num_small,
        num_medium_prosumers=num_medium,
        num_large_prosumers=num_large
    )