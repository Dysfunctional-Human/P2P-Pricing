"""P2P Pricing Mechanism Recommendation System.

A system to recommend the best P2P pricing mechanism based on
community composition and user priorities.

Usage:
    from p2p_pricing import recommend

    result = recommend(
        num_consumers=5,
        num_small_prosumers=3,
        num_medium_prosumers=4,
        num_large_prosumers=3
    )
    print(result.recommended_mechanism)
    print(result.reasoning)
"""

from .config import SimulationConfig, RecommendationWeights, TariffConfig
from .recommender import recommend, recommend_simple, Recommendation
from .profiles import generate_all_profiles
from .settlements import run_all_settlements, SettlementResult
from .metrics import compute_all_metrics, MetricsReport, MechanismMetrics

__version__ = "0.1.0"

__all__ = [
    # Main API
    "recommend",
    "recommend_simple",
    "Recommendation",
    # Configuration
    "SimulationConfig",
    "RecommendationWeights",
    "TariffConfig",
    # Lower-level API
    "generate_all_profiles",
    "run_all_settlements",
    "SettlementResult",
    "compute_all_metrics",
    "MetricsReport",
    "MechanismMetrics",
]