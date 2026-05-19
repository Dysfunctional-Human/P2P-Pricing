"""Generate real-data load profile figures for the journal paper.
- Fig 1: Average daily load profile with variability (already exists as fig_real_load_profiles)
- Fig 2: Individual household load reshaping under SDR-DSM (fixed version)
"""
import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from run_scenarios import settle_sdr, DT, DAYS, PERIODS_PER_DAY
from run_real_data import (
    load_meter_data, aggregate_to_15min, find_best_window,
    clean_and_filter, assign_roles, build_load_matrix,
    generate_pv_for_real_loads, assign_tariffs
)

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})


def load_real_data(prosumer_frac=0.6):
    data_dir = '../electricity-smart-meter-data-from-india'
    bareilly_path = os.path.join(data_dir, 'CEEW - Smart meter data Bareilly 2020.csv')
    mathura_path = os.path.join(data_dir, 'CEEW - Smart meter data Mathura 2020.csv')

    print("  Loading CEEW meter data...")
    df = load_meter_data(bareilly_path, mathura_path)
    df = aggregate_to_15min(df)
    best_start = find_best_window(df, num_days=30)
    pivot = clean_and_filter(df, best_start, num_days=30)

    ordered_ids, nc, np_ = assign_roles(pivot, prosumer_fraction=prosumer_frac, seed=42)
    L = build_load_matrix(pivot, ordered_ids)
    PV = generate_pv_for_real_loads(L, nc, np_, num_days=30, seed=42)

    return L, PV, nc, np_


def run_dsm_get_adjusted(L_ref, PV, lbuy, lsell, alpha=0.12, max_iter=8):
    """Run SDR-DSM and return the adjusted load matrix."""
    N, H = L_ref.shape
    num_days = H // PERIODS_PER_DAY
    L_adj = L_ref.copy()
    damping = 0.3
    p2p_allowed = np.ones(H, dtype=bool)

    for iteration in range(max_iter):
        L_prev = L_adj.copy()
        _, _, Pr_buy, Pr_sell, SDR_ts = settle_sdr(L_adj, PV, DT, lbuy, lsell, p2p_allowed)

        for i in range(N):
            prices = np.zeros(H)
            NP_i = L_adj[i] - PV[i]
            for h in range(H):
                prices[h] = Pr_buy[h] if NP_i[h] > 0 else -Pr_sell[h]

            adjustment = -(prices * DT) / (2 * alpha)
            new_load = L_ref[i] + adjustment

            for d in range(num_days):
                s = d * PERIODS_PER_DAY
                e = (d + 1) * PERIODS_PER_DAY
                day_ref_energy = np.sum(L_ref[i, s:e]) * DT
                new_load[s:e] = np.clip(new_load[s:e], 0.5 * L_ref[i, s:e], 1.5 * L_ref[i, s:e])
                day_new_energy = np.sum(new_load[s:e]) * DT
                if day_new_energy > 0:
                    new_load[s:e] *= day_ref_energy / day_new_energy

            L_adj[i] = (1 - damping) * L_prev[i] + damping * new_load

        max_change = np.max(np.abs(L_adj - L_prev))
        if max_change < 1e-4:
            break

    return L_adj


def fig_real_individual_reshaping():
    print("Generating real data individual reshaping figure...")
    L, PV, nc, np_ = load_real_data(prosumer_frac=0.6)
    N = nc + np_

    lambda_buy_ref = 6.0
    lambda_sell_ref = 3.4

    print("  Running SDR-DSM on real data...")
    L_adj = run_dsm_get_adjusted(L, PV, lambda_buy_ref, lambda_sell_ref)

    # Verify: L_adj should be non-negative and bounded
    print(f"  L_adj min: {L_adj.min():.4f}, max: {L_adj.max():.4f}")
    print(f"  L_ref min: {L.min():.4f}, max: {L.max():.4f}")

    # Pick a day where load shifting is visible
    # Use day-averaged profiles for cleaner visualization
    day = 14  # middle of the 30-day window

    s = day * PERIODS_PER_DAY
    e = (day + 1) * PERIODS_PER_DAY
    hours = np.arange(PERIODS_PER_DAY) * DT

    # Pick 3 prosumers with different PV sizes
    prosumer_indices = list(range(nc, N))
    pv_daily = np.array([np.sum(PV[i]) * DT / 30 for i in prosumer_indices])
    sorted_prosumers = [prosumer_indices[j] for j in np.argsort(pv_daily)[::-1]]

    # Pick high, medium PV prosumers
    p1_idx = sorted_prosumers[2]   # high PV (not the very highest which may be extreme)
    p2_idx = sorted_prosumers[len(sorted_prosumers)//2]  # medium PV

    # Pick a consumer with reasonable load
    consumer_indices = list(range(nc))
    load_daily = np.array([np.sum(L[i]) * DT / 30 for i in consumer_indices])
    sorted_consumers = [consumer_indices[j] for j in np.argsort(load_daily)[::-1]]
    c1_idx = sorted_consumers[2]  # high-load consumer (not extreme)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)

    plot_data = [
        (axes[0], p1_idx, 'Prosumer A (High PV)', True),
        (axes[1], p2_idx, 'Prosumer B (Medium PV)', True),
        (axes[2], c1_idx, 'Consumer C', False),
    ]

    for ax, idx, title, show_pv in plot_data:
        ref = L[idx, s:e]
        adj = L_adj[idx, s:e]

        ax.plot(hours, ref, 'b-', linewidth=1.5, label='Reference')
        ax.plot(hours, adj, 'r--', linewidth=1.5, label='Adjusted')

        if show_pv:
            pv = PV[idx, s:e]
            ax.fill_between(hours, 0, pv, alpha=0.12, color='green', label='PV')

        # Shade regions where load increased vs decreased
        ax.fill_between(hours, ref, adj, where=(adj > ref),
                       alpha=0.2, color='red', label='Increased')
        ax.fill_between(hours, ref, adj, where=(adj < ref),
                       alpha=0.2, color='blue', label='Decreased')

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Hour of Day')
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc='upper right', ncol=2)

    axes[0].set_ylabel('Power (kW)')

    plt.tight_layout()
    plt.savefig('figures/fig_real_individual_reshaping.pdf')
    plt.close()
    print("  Saved figures/fig_real_individual_reshaping.pdf")


def fig_real_aggregate_reshaping():
    """Community aggregate load profile for real data: before and after DSM."""
    print("Generating real data aggregate reshaping figure...")
    L, PV, nc, np_ = load_real_data(prosumer_frac=0.6)
    N = nc + np_

    lambda_buy_ref = 6.0
    lambda_sell_ref = 3.4

    print("  Running SDR-DSM on real data...")
    L_adj = run_dsm_get_adjusted(L, PV, lambda_buy_ref, lambda_sell_ref)

    # Compute average daily profiles
    num_days = L.shape[1] // PERIODS_PER_DAY
    hours = np.arange(PERIODS_PER_DAY) * DT

    # Community aggregate
    agg_ref = np.sum(L, axis=0)
    agg_adj = np.sum(L_adj, axis=0)
    agg_pv = np.sum(PV, axis=0)

    # Average across days
    ref_daily = np.mean(agg_ref.reshape(num_days, PERIODS_PER_DAY), axis=0)
    adj_daily = np.mean(agg_adj.reshape(num_days, PERIODS_PER_DAY), axis=0)
    pv_daily = np.mean(agg_pv.reshape(num_days, PERIODS_PER_DAY), axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(hours, ref_daily, 'b-', linewidth=2, label='Reference Load')
    ax.plot(hours, adj_daily, 'r--', linewidth=2, label='Adjusted Load (SDR-DSM)')
    ax.plot(hours, pv_daily, 'g-', linewidth=1.5, alpha=0.7, label='PV Generation')

    # Shade surplus/deficit
    ax.fill_between(hours, adj_daily, pv_daily, where=(pv_daily > adj_daily),
                   alpha=0.1, color='green', label='Surplus (export)')
    ax.fill_between(hours, adj_daily, pv_daily, where=(pv_daily < adj_daily),
                   alpha=0.1, color='red', label='Deficit (import)')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power (kW)')
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title('Community Aggregate Profile (56 Real Indian Meters, 60% Prosumer)')

    plt.tight_layout()
    plt.savefig('figures/fig_real_aggregate_reshaping.pdf')
    plt.close()
    print("  Saved figures/fig_real_aggregate_reshaping.pdf")


if __name__ == "__main__":
    fig_real_individual_reshaping()
    fig_real_aggregate_reshaping()
