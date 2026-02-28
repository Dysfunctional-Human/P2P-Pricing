"""Configuration dataclass for P2P pricing simulation."""

from dataclasses import dataclass
from typing import Tuple, Literal, Optional


# Tariff modes
TariffMode = Literal["constant", "random", "default"]


@dataclass
class TariffConfig:
    """Configuration for grid tariffs with flexible modes."""

    # Mode: "constant", "random", or "default"
    consumer_buy_mode: TariffMode = "default"
    prosumer_buy_mode: TariffMode = "default"

    # Constant values (used when mode is "constant" or "default")
    consumer_buy_price: float = 6.0
    prosumer_buy_price: float = 6.0
    grid_sell_price: float = 3.4  # Prosumers only

    # Random ranges (used when mode is "random")
    consumer_buy_range: Tuple[float, float] = (5.0, 7.0)
    prosumer_buy_range: Tuple[float, float] = (5.0, 7.0)


@dataclass
class SimulationConfig:
    """Configuration parameters for the P2P pricing simulation."""

    # Time settings
    days: int = 30
    periods_per_day: int = 96  # 15-minute resolution
    dt: float = 0.25  # hours per time slot (15 minutes)

    # Random seed for reproducibility
    random_seed: int = 42

    # Grid tariffs (Rs/kWh) - for backward compatibility
    # These are used as defaults when TariffConfig is not provided
    grid_buy_price: float = 6.0      # Price to buy from grid (all households)
    grid_sell_price: float = 3.4     # Price to sell to grid (prosumers only)

    # Flexible tariff configuration (optional)
    tariff_config: Optional[TariffConfig] = None

    # Load profile settings (kWh/day)
    consumer_load_range: Tuple[float, float] = (4.0, 7.0)
    prosumer_load_range: Tuple[float, float] = (5.0, 9.0)

    # PV sizing factors (ratio of daily PV generation to daily load)
    # Small prosumers: mainly self-consume, rarely export
    small_prosumer_pv_range: Tuple[float, float] = (0.4, 0.6)
    # Medium prosumers: balanced, can export moderate amounts
    medium_prosumer_pv_range: Tuple[float, float] = (0.8, 1.1)
    # Large prosumers: significant exporters
    large_prosumer_pv_range: Tuple[float, float] = (1.3, 1.8)

    # PV yield (kWh per kWp per day) - typical for India
    pv_yield_range: Tuple[float, float] = (4.5, 5.5)

    # Random events
    num_inverter_events: int = 2
    num_maintenance_events: int = 3
    maintenance_slots_per_event: int = 8  # 2 hours

    @property
    def total_periods(self) -> int:
        """Total number of time periods in simulation."""
        return self.days * self.periods_per_day


@dataclass
class RecommendationWeights:
    """Weights for recommendation scoring."""

    cost_savings: float = 0.4
    fairness: float = 0.3
    stability: float = 0.3

    def __post_init__(self):
        """Normalize weights to sum to 1."""
        total = self.cost_savings + self.fairness + self.stability
        if total > 0:
            self.cost_savings /= total
            self.fairness /= total
            self.stability /= total