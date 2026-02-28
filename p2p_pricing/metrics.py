"""Metrics computation for P2P pricing mechanism evaluation.

Computes three key metrics:
1. Cost Savings - % reduction vs conventional pricing
2. Fairness Index - Balance of savings between consumer/prosumer groups
3. Cost Volatility - Standard deviation of daily costs (stability)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from .settlements import SettlementResult


@dataclass
class MechanismMetrics:
    """Computed metrics for a single pricing mechanism."""
    mechanism_name: str
    total_cost: float                 # Total community cost
    cost_savings_pct: float           # % savings vs conventional
    consumer_savings_pct: float       # % savings for consumers
    prosumer_savings_pct: float       # % savings for prosumers
    fairness_index: float             # 0-1, higher = more fair
    daily_cost_volatility: float      # Std dev of daily costs
    avg_daily_cost: float             # Average daily cost


@dataclass
class MetricsReport:
    """Complete metrics report for all mechanisms."""
    mechanisms: Dict[str, MechanismMetrics]
    household_types: List[str]
    conventional_total: float
    periods_per_day: int
    num_days: int


def compute_group_costs(
    bills: np.ndarray,
    household_types: List[str]
) -> Dict[str, float]:
    """
    Compute total costs by household group.

    Returns dict with 'consumer', 'small_prosumer', 'medium_prosumer',
    'large_prosumer', and 'all_prosumers' totals.
    """
    costs = {
        'consumer': 0.0,
        'small_prosumer': 0.0,
        'medium_prosumer': 0.0,
        'large_prosumer': 0.0,
        'all_prosumers': 0.0
    }

    for i, hh_type in enumerate(household_types):
        costs[hh_type] += bills[i]
        if hh_type != 'consumer':
            costs['all_prosumers'] += bills[i]

    return costs


def compute_fairness_index(
    consumer_savings_pct: float,
    prosumer_savings_pct: float
) -> float:
    """
    Compute fairness index based on savings balance.

    Fairness = 1 - |consumer_sav% - prosumer_sav%| / max_possible_diff

    Returns value in [0, 1] where 1 = perfectly fair (equal savings %)
    """
    # Max possible difference is 200% (one group saves 100%, other loses 100%)
    max_diff = 200.0
    actual_diff = abs(consumer_savings_pct - prosumer_savings_pct)

    fairness = 1.0 - (actual_diff / max_diff)
    return max(0.0, min(1.0, fairness))


def compute_daily_volatility(
    cost_ts: np.ndarray,
    periods_per_day: int
) -> float:
    """
    Compute standard deviation of daily total costs.

    Lower volatility = more predictable bills.
    """
    N, H = cost_ts.shape
    num_days = H // periods_per_day

    daily_costs = np.zeros(num_days)
    for d in range(num_days):
        start = d * periods_per_day
        end = (d + 1) * periods_per_day
        daily_costs[d] = np.sum(cost_ts[:, start:end])

    return float(np.std(daily_costs))


def compute_mechanism_metrics(
    result: SettlementResult,
    conventional_result: SettlementResult,
    household_types: List[str],
    periods_per_day: int
) -> MechanismMetrics:
    """
    Compute all metrics for a single pricing mechanism.

    Args:
        result: SettlementResult for this mechanism
        conventional_result: SettlementResult for conventional (baseline)
        household_types: List of household types
        periods_per_day: Number of periods per day

    Returns:
        MechanismMetrics with all computed values
    """
    # Total costs
    total_cost = float(np.sum(result.bills))
    conv_total = float(np.sum(conventional_result.bills))

    # Cost savings percentage
    if conv_total != 0:
        cost_savings_pct = ((conv_total - total_cost) / abs(conv_total)) * 100
    else:
        cost_savings_pct = 0.0

    # Group costs for this mechanism
    group_costs = compute_group_costs(result.bills, household_types)
    conv_group_costs = compute_group_costs(conventional_result.bills, household_types)

    # Consumer savings
    if conv_group_costs['consumer'] != 0:
        consumer_savings_pct = (
            (conv_group_costs['consumer'] - group_costs['consumer']) /
            abs(conv_group_costs['consumer'])
        ) * 100
    else:
        consumer_savings_pct = 0.0

    # Prosumer savings (all prosumer types combined)
    if conv_group_costs['all_prosumers'] != 0:
        prosumer_savings_pct = (
            (conv_group_costs['all_prosumers'] - group_costs['all_prosumers']) /
            abs(conv_group_costs['all_prosumers'])
        ) * 100
    else:
        prosumer_savings_pct = 0.0

    # Fairness index
    fairness = compute_fairness_index(consumer_savings_pct, prosumer_savings_pct)

    # Daily volatility
    volatility = compute_daily_volatility(result.cost_ts, periods_per_day)

    # Average daily cost
    N, H = result.cost_ts.shape
    num_days = H // periods_per_day
    avg_daily = total_cost / num_days if num_days > 0 else total_cost

    return MechanismMetrics(
        mechanism_name=result.mechanism_name,
        total_cost=total_cost,
        cost_savings_pct=cost_savings_pct,
        consumer_savings_pct=consumer_savings_pct,
        prosumer_savings_pct=prosumer_savings_pct,
        fairness_index=fairness,
        daily_cost_volatility=volatility,
        avg_daily_cost=avg_daily
    )


def compute_all_metrics(
    settlement_results: Dict[str, SettlementResult],
    household_types: List[str],
    periods_per_day: int = 96
) -> MetricsReport:
    """
    Compute metrics for all settlement mechanisms.

    Args:
        settlement_results: Dict mapping mechanism name to SettlementResult
        household_types: List of household types
        periods_per_day: Number of periods per day (default: 96 for 15-min slots)

    Returns:
        MetricsReport with metrics for all mechanisms
    """
    conventional = settlement_results['Conventional']
    conv_total = float(np.sum(conventional.bills))

    # Compute metrics for each mechanism
    metrics = {}
    for name, result in settlement_results.items():
        metrics[name] = compute_mechanism_metrics(
            result, conventional, household_types, periods_per_day
        )

    # Calculate number of days
    H = conventional.cost_ts.shape[1]
    num_days = H // periods_per_day

    return MetricsReport(
        mechanisms=metrics,
        household_types=household_types,
        conventional_total=conv_total,
        periods_per_day=periods_per_day,
        num_days=num_days
    )