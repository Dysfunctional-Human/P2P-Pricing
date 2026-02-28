"""Load and PV profile generation functions.

Refactored from toySystem.ipynb with support for 4 household types:
- Consumers: No PV
- Small prosumers: PV sized at 40-60% of load (mainly self-consume)
- Medium prosumers: PV sized at 80-110% of load (can export some)
- Large prosumers: PV sized at 130-180% of load (significant exporters)
"""

import numpy as np
from typing import Tuple
from .config import SimulationConfig


def generate_daily_load_shape(periods_per_day: int = 96, dt: float = 0.25) -> np.ndarray:
    """
    Create a generic 1-day load shape with morning & evening peaks.
    Returns an array of kW values (relative shape, not scaled to kWh).
    """
    t = np.arange(periods_per_day)
    hours = t * dt

    # Base: low at night, higher in evening
    base = 0.2  # constant background load like fridge, fan, etc.
    morning_peak = np.exp(-0.5 * ((hours - 8) / 1.5) ** 2)
    evening_peak = 1.3 * np.exp(-0.5 * ((hours - 20) / 2.0) ** 2)

    shape = base + 0.7 * morning_peak + 1.0 * evening_peak
    shape = np.maximum(shape, 0.05)  # avoid zero
    return shape


def generate_daily_pv_shape(periods_per_day: int = 96, dt: float = 0.25) -> np.ndarray:
    """
    Clear-sky PV power shape (kW per kWp) peaking at noon.
    """
    t = np.arange(periods_per_day)
    hours = t * dt

    # Simple sine-shaped irradiance between 6:00 and 18:00
    angle = np.pi * (hours - 6) / 12.0
    irradiance = np.sin(angle)
    irradiance[irradiance < 0] = 0.0

    return irradiance


def generate_household_loads(
    num_consumers: int,
    num_small_prosumers: int,
    num_medium_prosumers: int,
    num_large_prosumers: int,
    days: int = 30,
    periods_per_day: int = 96,
    dt: float = 0.25,
    consumer_load_range: Tuple[float, float] = (4.0, 7.0),
    prosumer_load_range: Tuple[float, float] = (5.0, 9.0)
) -> Tuple[np.ndarray, list]:
    """
    Generate L[N, H] load matrix (kW) for all households.

    Returns:
        Tuple of (L matrix, household_types list)
    """
    N = num_consumers + num_small_prosumers + num_medium_prosumers + num_large_prosumers
    H = days * periods_per_day
    shape_day = generate_daily_load_shape(periods_per_day, dt)

    # precompute day-normalization
    shape_energy_per_day = np.sum(shape_day * dt)

    L = np.zeros((N, H))
    household_types = []

    # Track indices for each type
    idx = 0

    # Consumers
    for _ in range(num_consumers):
        household_types.append('consumer')
        target_kwh_day = np.random.uniform(*consumer_load_range)
        scale = target_kwh_day / shape_energy_per_day
        base_profile = shape_day * scale

        for d in range(days):
            day_noise_factor = np.random.normal(1.0, 0.05)
            noise = np.random.normal(0.0, 0.05, size=periods_per_day)
            profile_day = base_profile * day_noise_factor * (1.0 + noise)
            profile_day = np.clip(profile_day, 0.05, None)
            start = d * periods_per_day
            end = (d + 1) * periods_per_day
            L[idx, start:end] = profile_day
        idx += 1

    # Small prosumers
    for _ in range(num_small_prosumers):
        household_types.append('small_prosumer')
        target_kwh_day = np.random.uniform(*prosumer_load_range)
        scale = target_kwh_day / shape_energy_per_day
        base_profile = shape_day * scale

        for d in range(days):
            day_noise_factor = np.random.normal(1.0, 0.05)
            noise = np.random.normal(0.0, 0.05, size=periods_per_day)
            profile_day = base_profile * day_noise_factor * (1.0 + noise)
            profile_day = np.clip(profile_day, 0.05, None)
            start = d * periods_per_day
            end = (d + 1) * periods_per_day
            L[idx, start:end] = profile_day
        idx += 1

    # Medium prosumers
    for _ in range(num_medium_prosumers):
        household_types.append('medium_prosumer')
        target_kwh_day = np.random.uniform(*prosumer_load_range)
        scale = target_kwh_day / shape_energy_per_day
        base_profile = shape_day * scale

        for d in range(days):
            day_noise_factor = np.random.normal(1.0, 0.05)
            noise = np.random.normal(0.0, 0.05, size=periods_per_day)
            profile_day = base_profile * day_noise_factor * (1.0 + noise)
            profile_day = np.clip(profile_day, 0.05, None)
            start = d * periods_per_day
            end = (d + 1) * periods_per_day
            L[idx, start:end] = profile_day
        idx += 1

    # Large prosumers
    for _ in range(num_large_prosumers):
        household_types.append('large_prosumer')
        target_kwh_day = np.random.uniform(*prosumer_load_range)
        scale = target_kwh_day / shape_energy_per_day
        base_profile = shape_day * scale

        for d in range(days):
            day_noise_factor = np.random.normal(1.0, 0.05)
            noise = np.random.normal(0.0, 0.05, size=periods_per_day)
            profile_day = base_profile * day_noise_factor * (1.0 + noise)
            profile_day = np.clip(profile_day, 0.05, None)
            start = d * periods_per_day
            end = (d + 1) * periods_per_day
            L[idx, start:end] = profile_day
        idx += 1

    return L, household_types


def generate_pv_profiles(
    L: np.ndarray,
    household_types: list,
    days: int = 30,
    periods_per_day: int = 96,
    dt: float = 0.25,
    small_pv_range: Tuple[float, float] = (0.4, 0.6),
    medium_pv_range: Tuple[float, float] = (0.8, 1.1),
    large_pv_range: Tuple[float, float] = (1.3, 1.8),
    pv_yield_range: Tuple[float, float] = (4.5, 5.5)
) -> np.ndarray:
    """
    Generate PV[N,H] matrix (kW) where prosumer PV size is chosen
    based on their type.
    """
    N, H = L.shape
    assert H == days * periods_per_day

    pv_shape = generate_daily_pv_shape(periods_per_day, dt)
    shape_energy_per_day_per_kwp = np.sum(pv_shape * dt)

    PV = np.zeros((N, H))

    for idx in range(N):
        hh_type = household_types[idx]

        if hh_type == 'consumer':
            # Consumers have no PV
            continue

        # Compute this household's average daily load
        total_load_kwh = np.sum(L[idx, :] * dt)
        avg_daily_load_kwh = total_load_kwh / days

        # Choose sizing factor based on type
        if hh_type == 'small_prosumer':
            sizing_factor = np.random.uniform(*small_pv_range)
        elif hh_type == 'medium_prosumer':
            sizing_factor = np.random.uniform(*medium_pv_range)
        elif hh_type == 'large_prosumer':
            sizing_factor = np.random.uniform(*large_pv_range)
        else:
            sizing_factor = 0.5  # fallback

        target_pv_kwh_day = sizing_factor * avg_daily_load_kwh

        # Realistic PV yield per kWp
        yield_per_kwp = np.random.uniform(*pv_yield_range)
        capacity_kwp = target_pv_kwh_day / yield_per_kwp

        # Scale clear-sky shape to this capacity
        base_profile_kw = pv_shape * capacity_kwp * (yield_per_kwp / shape_energy_per_day_per_kwp)

        for d in range(days):
            # Weather variation / clouds
            cloud_factor = np.random.normal(1.0, 0.15)
            noise = np.random.normal(0.0, 0.05, size=periods_per_day)

            profile_day = base_profile_kw * cloud_factor * (1.0 + noise)
            profile_day = np.clip(profile_day, 0.0, None)

            start = d * periods_per_day
            end = (d + 1) * periods_per_day
            PV[idx, start:end] = profile_day

    return PV


def apply_random_events(
    PV: np.ndarray,
    household_types: list,
    days: int = 30,
    periods_per_day: int = 96,
    num_inverter_events: int = 2,
    num_maintenance_events: int = 3,
    slots_per_event: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random inverter failures and maintenance windows.

    Returns:
        Tuple of (adjusted PV matrix, p2p_allowed boolean array)
    """
    PV = PV.copy()
    H = days * periods_per_day

    # Find prosumer indices
    prosumer_indices = [i for i, t in enumerate(household_types) if t != 'consumer']

    # Inverter failures
    for _ in range(num_inverter_events):
        if len(prosumer_indices) == 0:
            break
        prosumer_idx = np.random.choice(prosumer_indices)
        day = np.random.randint(0, days)
        start = day * periods_per_day
        end = (day + 1) * periods_per_day
        PV[prosumer_idx, start:end] = 0.0

    # Maintenance windows (P2P not allowed)
    p2p_allowed = np.ones(H, dtype=bool)
    for _ in range(num_maintenance_events):
        start_slot = np.random.randint(0, H - slots_per_event)
        end_slot = start_slot + slots_per_event
        p2p_allowed[start_slot:end_slot] = False

    return PV, p2p_allowed


def generate_tariffs(
    household_types: list,
    grid_buy_price: float = 6.0,
    grid_sell_price: float = 3.4,
    tariff_config: 'TariffConfig' = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Generate tariffs for each household.

    Supports three modes per household type (consumer/prosumer):
    - "constant": Use specified constant price
    - "random": Random price within specified range (different per household)
    - "default": Use grid_buy_price for all

    Only prosumers can sell to grid.

    Returns:
        Tuple of (lambda_buy_individual, lambda_sell_individual,
                  lambda_buy_ref, lambda_sell_ref)
    """
    from .config import TariffConfig

    N = len(household_types)
    lambda_buy_individual = np.zeros(N)
    lambda_sell_individual = np.zeros(N)

    # Use default config if not provided
    if tariff_config is None:
        tariff_config = TariffConfig(
            consumer_buy_mode="default",
            prosumer_buy_mode="default",
            consumer_buy_price=grid_buy_price,
            prosumer_buy_price=grid_buy_price,
            grid_sell_price=grid_sell_price
        )

    for i, hh_type in enumerate(household_types):
        is_consumer = (hh_type == 'consumer')

        if is_consumer:
            mode = tariff_config.consumer_buy_mode
            if mode == "constant":
                lambda_buy_individual[i] = tariff_config.consumer_buy_price
            elif mode == "random":
                low, high = tariff_config.consumer_buy_range
                lambda_buy_individual[i] = np.random.uniform(low, high)
            else:  # "default"
                lambda_buy_individual[i] = grid_buy_price
            # Consumers cannot sell to grid
            lambda_sell_individual[i] = 0.0
        else:
            # Prosumer (small, medium, or large)
            mode = tariff_config.prosumer_buy_mode
            if mode == "constant":
                lambda_buy_individual[i] = tariff_config.prosumer_buy_price
            elif mode == "random":
                low, high = tariff_config.prosumer_buy_range
                lambda_buy_individual[i] = np.random.uniform(low, high)
            else:  # "default"
                lambda_buy_individual[i] = grid_buy_price
            # Prosumers can sell to grid
            lambda_sell_individual[i] = tariff_config.grid_sell_price

    # Reference prices for mechanisms that need a single value
    lambda_buy_ref = grid_buy_price
    lambda_sell_ref = tariff_config.grid_sell_price

    return lambda_buy_individual, lambda_sell_individual, lambda_buy_ref, lambda_sell_ref


def generate_all_profiles(
    num_consumers: int,
    num_small_prosumers: int,
    num_medium_prosumers: int,
    num_large_prosumers: int,
    config: SimulationConfig = None,
    seed: int = None
) -> dict:
    """
    Generate all profiles needed for simulation.

    Args:
        num_consumers: Number of consumer households (no PV)
        num_small_prosumers: Number of small prosumers (40-60% PV)
        num_medium_prosumers: Number of medium prosumers (80-110% PV)
        num_large_prosumers: Number of large prosumers (130-180% PV)
        config: Simulation configuration (uses defaults if None)
        seed: Random seed (overrides config if provided)

    Returns:
        Dictionary with all simulation data
    """
    if config is None:
        config = SimulationConfig()

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(config.random_seed)

    days = config.days
    periods_per_day = config.periods_per_day
    dt = config.dt

    # Generate load profiles
    L, household_types = generate_household_loads(
        num_consumers, num_small_prosumers, num_medium_prosumers, num_large_prosumers,
        days, periods_per_day, dt,
        config.consumer_load_range, config.prosumer_load_range
    )

    # Generate PV profiles based on household types
    PV = generate_pv_profiles(
        L, household_types, days, periods_per_day, dt,
        config.small_prosumer_pv_range,
        config.medium_prosumer_pv_range,
        config.large_prosumer_pv_range,
        config.pv_yield_range
    )

    # Apply random events
    PV, p2p_allowed = apply_random_events(
        PV, household_types, days, periods_per_day,
        config.num_inverter_events, config.num_maintenance_events,
        config.maintenance_slots_per_event
    )

    # Generate tariffs
    lambda_buy_ind, lambda_sell_ind, lambda_buy_ref, lambda_sell_ref = \
        generate_tariffs(
            household_types,
            config.grid_buy_price,
            config.grid_sell_price,
            config.tariff_config
        )

    return {
        'L': L,
        'PV': PV,
        'p2p_allowed': p2p_allowed,
        'lambda_buy_ind': lambda_buy_ind,
        'lambda_sell_ind': lambda_sell_ind,
        'lambda_buy_ref': lambda_buy_ref,
        'lambda_sell_ref': lambda_sell_ref,
        'household_types': household_types,
        'config': config,
        'num_consumers': num_consumers,
        'num_small_prosumers': num_small_prosumers,
        'num_medium_prosumers': num_medium_prosumers,
        'num_large_prosumers': num_large_prosumers
    }