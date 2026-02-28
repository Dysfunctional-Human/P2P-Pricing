"""Settlement mechanisms for P2P energy trading.

Implements 4 pricing mechanisms:
1. Conventional - Traditional grid import/export
2. MMR (Mid-Market Rate) - Dynamic P2P pricing
3. Bill-Sharing - P2P trades with proportional sharing
4. SDR (Supply-Demand Ratio) - Incentive-based pricing

"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SettlementResult:
    """Result from a settlement mechanism."""
    mechanism_name: str
    bills: np.ndarray           # [N] total bill per household
    cost_ts: np.ndarray         # [N, H] cost time series
    extra_data: Optional[dict] = None  # Mechanism-specific data


def compute_cp2p_dynamic(
    L_total: np.ndarray,
    PV_total: np.ndarray,
    lambda_buy_ref: float,
    lambda_sell_ref: float,
    eps: float = 1e-9
) -> np.ndarray:
    """
    Compute cp2p[h] between sell and buy price depending on net balance.

    net = (L - PV)/(L + PV) in [-1,1]
      net = -1 -> huge surplus -> cp2p ~ lambda_sell
      net =  0 -> balanced     -> cp2p ~ midpoint
      net = +1 -> huge shortage-> cp2p ~ lambda_buy
    """
    H = len(L_total)
    cp2p = np.zeros(H)
    mid = 0.5 * (lambda_buy_ref + lambda_sell_ref)
    spread = 0.5 * (lambda_buy_ref - lambda_sell_ref)

    for h in range(H):
        denom = L_total[h] + PV_total[h]
        if denom < eps:
            cp2p[h] = mid
            continue
        net = (L_total[h] - PV_total[h]) / denom
        cp2p[h] = mid + spread * net

    return cp2p


def settle_conventional(
    L: np.ndarray,
    PV: np.ndarray,
    dt: float,
    lambda_buy_individual: np.ndarray,
    lambda_sell_individual: np.ndarray
) -> SettlementResult:
    """
    Conventional grid pricing settlement.

    Each household:
      - Self-consumes min(L, PV) for free
      - Net import from grid at its own lambda_buy_i
      - Net export to grid at its own lambda_sell_i

    Returns:
        SettlementResult with bills and cost time series
    """
    N, H = L.shape
    M = np.minimum(L, PV)
    residual_import = np.maximum(L - M, 0.0)
    residual_export = np.maximum(PV - M, 0.0)

    cost_ts = np.zeros((N, H))
    for i in range(N):
        cost_ts[i, :] = (
            residual_import[i, :] * dt * lambda_buy_individual[i] -
            residual_export[i, :] * dt * lambda_sell_individual[i]
        )

    bills = np.sum(cost_ts, axis=1)
    return SettlementResult(
        mechanism_name="Conventional",
        bills=bills,
        cost_ts=cost_ts
    )


def settle_mmr(
    L: np.ndarray,
    PV: np.ndarray,
    dt: float,
    lambda_buy_ref: float,
    lambda_sell_ref: float,
    p2p_allowed: np.ndarray,
    epsilon: float = 1e-6
) -> SettlementResult:
    """
    Mid-market rate with dynamic cp2p[h].

    Uses reference grid tariffs for simplicity.

    Returns:
        SettlementResult with bills, cost time series, and price data
    """
    N, H = L.shape
    L_total = np.sum(L, axis=0)
    PV_total = np.sum(PV, axis=0)

    # Dynamic cp2p per slot
    cp2p = compute_cp2p_dynamic(L_total, PV_total, lambda_buy_ref, lambda_sell_ref)

    Pim = np.zeros_like(L)
    Pex = np.zeros_like(L)
    for h in range(H):
        for i in range(N):
            M_ih = min(L[i, h], PV[i, h])
            Pim[i, h] = L[i, h] - M_ih
            Pex[i, h] = PV[i, h] - M_ih

    c_im = np.zeros(H)
    c_ex = np.zeros(H)
    for h in range(H):
        if not p2p_allowed[h]:
            c_im[h] = lambda_buy_ref
            c_ex[h] = lambda_sell_ref
            continue

        if abs(PV_total[h] - L_total[h]) <= epsilon:
            c_im[h] = cp2p[h]
            c_ex[h] = cp2p[h]
        elif PV_total[h] > L_total[h]:
            c_im[h] = cp2p[h]
            if PV_total[h] > 0:
                c_ex[h] = (L_total[h] * cp2p[h] +
                          (PV_total[h] - L_total[h]) * lambda_sell_ref) / PV_total[h]
            else:
                c_ex[h] = lambda_sell_ref
        else:
            c_ex[h] = cp2p[h]
            if L_total[h] > 0:
                c_im[h] = (PV_total[h] * cp2p[h] +
                          (L_total[h] - PV_total[h]) * lambda_buy_ref) / L_total[h]
            else:
                c_im[h] = lambda_buy_ref

    cost_ts = np.zeros((N, H))
    for i in range(N):
        for h in range(H):
            e_im = Pim[i, h] * dt
            e_ex = Pex[i, h] * dt
            cost_ts[i, h] += e_im * c_im[h]
            cost_ts[i, h] -= e_ex * c_ex[h]

    bills = np.sum(cost_ts, axis=1)
    return SettlementResult(
        mechanism_name="MMR",
        bills=bills,
        cost_ts=cost_ts,
        extra_data={'c_im': c_im, 'c_ex': c_ex, 'cp2p': cp2p}
    )


def settle_bill_sharing(
    L: np.ndarray,
    PV: np.ndarray,
    dt: float,
    lambda_buy_individual: np.ndarray,
    lambda_sell_individual: np.ndarray,
    lambda_buy_ref: float,
    lambda_sell_ref: float,
    p2p_allowed: np.ndarray,
    epsilon: float = 1e-9
) -> SettlementResult:
    """
    Bill-sharing P2P settlement (correct ex-post implementation).

    Based on "Comparison of Pricing Mechanisms in Peer-to-Peer Energy Communities":
    - Energy shared within community is FREE
    - Only grid interactions generate costs/revenues
    - P2P prices are CONSTANT over the billing period, calculated ex-post:
        p2p_buy = (retail_price × total_grid_shortage) / total_community_demand
        p2p_sell = (export_price × total_grid_surplus) / total_community_supply

    When p2p_allowed[h] is False, that slot uses grid-only pricing.

    Returns:
        SettlementResult with bills, cost time series, and p2p prices
    """
    N, H = L.shape

    # Step 1: Compute individual net positions per time slot
    # Positive = needs to import, Negative = can export
    net_power = L - PV  # [N, H]

    # Step 2: Aggregate community totals over the billing period
    # Only count slots where P2P is allowed
    total_demand = 0.0    # Sum of all positive net positions (buyers)
    total_supply = 0.0    # Sum of all negative net positions (sellers, absolute value)
    total_grid_import = 0.0  # Grid shortage (community net import)
    total_grid_export = 0.0  # Grid surplus (community net export)

    for h in range(H):
        if not p2p_allowed[h]:
            continue

        # Community aggregates for this slot
        buyers_mask = net_power[:, h] > 0
        sellers_mask = net_power[:, h] < 0

        slot_demand = net_power[buyers_mask, h].sum() if np.any(buyers_mask) else 0.0
        slot_supply = (-net_power[sellers_mask, h]).sum() if np.any(sellers_mask) else 0.0

        total_demand += slot_demand * dt
        total_supply += slot_supply * dt

        # Net community position
        community_net = slot_demand - slot_supply
        if community_net > 0:
            total_grid_import += community_net * dt
        else:
            total_grid_export += (-community_net) * dt

    # Step 3: Calculate ex-post constant P2P prices
    # p2p_buy = (retail × grid_shortage) / total_demand
    # p2p_sell = (export × grid_surplus) / total_supply
    if total_demand > epsilon:
        p2p_buy = (lambda_buy_ref * total_grid_import) / total_demand
    else:
        p2p_buy = 0.0

    if total_supply > epsilon:
        p2p_sell = (lambda_sell_ref * total_grid_export) / total_supply
    else:
        p2p_sell = 0.0

    # Step 4: Calculate costs for each household
    cost_ts = np.zeros((N, H))

    for h in range(H):
        if not p2p_allowed[h]:
            # No P2P: everyone uses grid at individual tariffs
            for i in range(N):
                if net_power[i, h] > 0:
                    cost_ts[i, h] = net_power[i, h] * dt * lambda_buy_individual[i]
                elif net_power[i, h] < 0:
                    cost_ts[i, h] = net_power[i, h] * dt * lambda_sell_individual[i]
            continue

        # P2P allowed: use constant ex-post prices
        for i in range(N):
            if net_power[i, h] > 0:
                # Buyer pays p2p_buy rate
                cost_ts[i, h] = net_power[i, h] * dt * p2p_buy
            elif net_power[i, h] < 0:
                # Seller receives p2p_sell rate (negative cost = revenue)
                cost_ts[i, h] = net_power[i, h] * dt * p2p_sell

    bills = np.sum(cost_ts, axis=1)
    return SettlementResult(
        mechanism_name="Bill-Sharing",
        bills=bills,
        cost_ts=cost_ts,
        extra_data={
            'p2p_buy': p2p_buy,
            'p2p_sell': p2p_sell,
            'total_demand': total_demand,
            'total_supply': total_supply,
            'total_grid_import': total_grid_import,
            'total_grid_export': total_grid_export
        }
    )


def settle_sdr(
    L: np.ndarray,
    PV: np.ndarray,
    dt: float,
    lambda_buy_ref: float,
    lambda_sell_ref: float,
    p2p_allowed: np.ndarray,
    epsilon: float = 1e-9
) -> SettlementResult:
    """
    SDR-based P2P pricing settlement.

    - Buyers: NP = L - PV > 0
    - Sellers: NP < 0
    - SDR = TSP / TBP
    - Prices follow SDR rules:
        * Shortage (0 <= SDR < 1):
            Pr_sell = (lambda_buy * lambda_sell) / ((lambda_buy - lambda_sell)*SDR + lambda_sell)
            Pr_buy  = Pr_sell * SDR + lambda_buy * (1 - SDR)
        * Surplus (SDR >= 1 or TBP = 0):
            Pr_sell = Pr_buy = lambda_sell

    Returns:
        SettlementResult with bills, cost time series, and pricing data
    """
    N, H = L.shape
    NP = L - PV  # +ve: buyer, -ve: seller

    # Grid imports/exports
    Pim = np.zeros_like(L)
    Pex = np.zeros_like(L)

    # P2P trades
    Pp2p_buy = np.zeros_like(L)
    Pp2p_sell = np.zeros_like(L)

    TSP = np.zeros(H)
    TBP = np.zeros(H)
    SDR = np.zeros(H)
    Pr_buy = np.zeros(H)
    Pr_sell = np.zeros(H)

    for h in range(H):
        # If P2P is not allowed, fall back to grid-only
        if not p2p_allowed[h]:
            for i in range(N):
                if NP[i, h] > 0:
                    Pim[i, h] = NP[i, h]
                elif NP[i, h] < 0:
                    Pex[i, h] = -NP[i, h]
            SDR[h] = np.nan
            Pr_buy[h] = lambda_buy_ref
            Pr_sell[h] = lambda_sell_ref
            continue

        buyers = NP[:, h] > 0
        sellers = NP[:, h] < 0

        TBP[h] = NP[buyers, h].sum() if np.any(buyers) else 0.0
        TSP[h] = (-NP[sellers, h]).sum() if np.any(sellers) else 0.0

        if TBP[h] <= epsilon:
            SDR[h] = np.inf
        else:
            SDR[h] = TSP[h] / TBP[h]

        # SDR-based pricing
        if not np.isfinite(SDR[h]) or SDR[h] >= 1.0:
            Pr_sell[h] = lambda_sell_ref
            Pr_buy[h] = lambda_sell_ref
        else:
            SDRh = SDR[h]
            Pr_sell[h] = (lambda_buy_ref * lambda_sell_ref) / (
                (lambda_buy_ref - lambda_sell_ref) * SDRh + lambda_sell_ref
            )
            Pr_buy[h] = Pr_sell[h] * SDRh + lambda_buy_ref * (1.0 - SDRh)

        # Allocate internal P2P energy
        internal_power = min(TSP[h], TBP[h])

        if internal_power > 0:
            if TSP[h] > 0 and np.any(sellers):
                seller_surplus = -NP[sellers, h]
                Pp2p_sell[sellers, h] = internal_power * seller_surplus / TSP[h]

            if TBP[h] > 0 and np.any(buyers):
                buyer_deficit = NP[buyers, h]
                Pp2p_buy[buyers, h] = internal_power * buyer_deficit / TBP[h]

        # Remaining goes to/from grid
        for i in range(N):
            net = NP[i, h]
            if net > 0:
                residual = net - Pp2p_buy[i, h]
                Pim[i, h] = max(residual, 0.0)
            elif net < 0:
                residual = -net - Pp2p_sell[i, h]
                Pex[i, h] = max(residual, 0.0)

    # Cost calculation
    cost_ts = np.zeros((N, H))
    for i in range(N):
        for h in range(H):
            # Grid trades
            cost_ts[i, h] += Pim[i, h] * dt * lambda_buy_ref
            cost_ts[i, h] -= Pex[i, h] * dt * lambda_sell_ref

            # P2P trades
            cost_ts[i, h] += Pp2p_buy[i, h] * dt * Pr_buy[h]
            cost_ts[i, h] -= Pp2p_sell[i, h] * dt * Pr_sell[h]

    bills = np.sum(cost_ts, axis=1)
    return SettlementResult(
        mechanism_name="SDR",
        bills=bills,
        cost_ts=cost_ts,
        extra_data={'Pr_buy': Pr_buy, 'Pr_sell': Pr_sell, 'SDR': SDR}
    )


def run_all_settlements(profiles: dict) -> dict:
    """
    Run all 4 settlement mechanisms on the given profiles.

    Args:
        profiles: Dictionary from generate_all_profiles()

    Returns:
        Dictionary mapping mechanism name to SettlementResult
    """
    L = profiles['L']
    PV = profiles['PV']
    dt = profiles['config'].dt
    lambda_buy_ind = profiles['lambda_buy_ind']
    lambda_sell_ind = profiles['lambda_sell_ind']
    lambda_buy_ref = profiles['lambda_buy_ref']
    lambda_sell_ref = profiles['lambda_sell_ref']
    p2p_allowed = profiles['p2p_allowed']

    results = {}

    results['Conventional'] = settle_conventional(
        L, PV, dt, lambda_buy_ind, lambda_sell_ind
    )

    results['MMR'] = settle_mmr(
        L, PV, dt, lambda_buy_ref, lambda_sell_ref, p2p_allowed
    )

    results['Bill-Sharing'] = settle_bill_sharing(
        L, PV, dt, lambda_buy_ind, lambda_sell_ind,
        lambda_buy_ref, lambda_sell_ref, p2p_allowed
    )

    results['SDR'] = settle_sdr(
        L, PV, dt, lambda_buy_ref, lambda_sell_ref, p2p_allowed
    )

    return results