
"""Generate all publication-quality figures for the Springer journal paper.
Outputs PDF figures suitable for LaTeX inclusion.
"""
import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import make_interp_spline
import os

from run_scenarios import (
    generate_profiles, settle_conventional, settle_mmr, settle_billsharing,
    settle_sdr, settle_sdr_dsm, generate_daily_load_shape, generate_daily_pv_shape,
    DAYS, PERIODS_PER_DAY, DT
)

os.makedirs('figures', exist_ok=True)

# Style settings
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

COLORS = {
    'BSM': '#2ca02c',
    'MMR': '#1f77b4',
    'SDR': '#ff7f0e',
    'SDR-DSM': '#d62728',
    'Conventional': '#7f7f7f',
}


def get_primary_simulation():
    """Run N=20 primary case and return all data."""
    nc, np_, seed = 8, 12, 42
    p = generate_profiles(nc, np_, seed)
    lbuy = p['lambda_buy_ref']
    lsell = p['lambda_sell_ref']

    bills_conv, cost_conv = settle_conventional(p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'])
    bills_mmr, cost_mmr, c_im_mmr, c_ex_mmr = settle_mmr(p['L'], p['PV'], DT, lbuy, lsell, p['p2p_allowed'])
    bills_bs, cost_bs, cp2p_bs = settle_billsharing(p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'], lbuy, lsell, p['p2p_allowed'])
    bills_sdr, cost_sdr, Pr_buy_s, Pr_sell_s, SDR_s = settle_sdr(p['L'], p['PV'], DT, lbuy, lsell, p['p2p_allowed'])
    bills_dsm, cost_dsm, Pr_buy_d, Pr_sell_d, SDR_d, iters = settle_sdr_dsm(p['L'], p['PV'], DT, lbuy, lsell, p['p2p_allowed'])

    # Get adjusted load from DSM (re-run to capture L_adj)
    N, H = p['L'].shape
    L_adj = p['L'].copy()
    damping = 0.3
    for iteration in range(8):
        L_prev = L_adj.copy()
        _, _, Pr_b, Pr_s, _ = settle_sdr(L_adj, p['PV'], DT, lbuy, lsell, p['p2p_allowed'])
        for i in range(N):
            prices = np.zeros(H)
            NP_i = L_adj[i] - p['PV'][i]
            for h in range(H):
                prices[h] = Pr_b[h] if NP_i[h] > 0 else -Pr_s[h]
            adjustment = -(prices * DT) / (2 * 0.12)
            new_load = p['L'][i] + adjustment
            for d in range(DAYS):
                s = d * PERIODS_PER_DAY
                e = (d + 1) * PERIODS_PER_DAY
                day_ref_energy = np.sum(p['L'][i, s:e]) * DT
                new_load[s:e] = np.clip(new_load[s:e], 0.5 * p['L'][i, s:e], 1.5 * p['L'][i, s:e])
                day_new_energy = np.sum(new_load[s:e]) * DT
                if day_new_energy > 0:
                    new_load[s:e] *= day_ref_energy / day_new_energy
            L_adj[i] = (1 - damping) * L_prev[i] + damping * new_load

    return {
        'p': p, 'nc': nc, 'L_adj': L_adj,
        'Pr_buy_sdr': Pr_buy_s, 'Pr_sell_sdr': Pr_sell_s, 'SDR_sdr': SDR_s,
        'Pr_buy_dsm': Pr_buy_d, 'Pr_sell_dsm': Pr_sell_d, 'SDR_dsm': SDR_d,
        'c_im_mmr': c_im_mmr, 'c_ex_mmr': c_ex_mmr, 'cp2p_bs': cp2p_bs,
        'lbuy': p['lambda_buy_ref'], 'lsell': p['lambda_sell_ref'],
    }


def hourly_avg(ts, days=DAYS):
    """Convert per-slot timeseries to hourly average across days."""
    reshaped = ts.reshape(days, PERIODS_PER_DAY)
    hourly = np.zeros((days, 24))
    for h in range(24):
        hourly[:, h] = reshaped[:, h*4:(h+1)*4].mean(axis=1)
    return hourly.mean(axis=0)


def smooth_curve(hours, values):
    """Smooth a 24-point curve for nicer plotting."""
    hours_fine = np.linspace(0, 23, 200)
    try:
        spl = make_interp_spline(hours, values, k=3)
        return hours_fine, spl(hours_fine)
    except Exception:
        return np.array(hours, dtype=float), values


# ============================================================================
# FIGURE 1: Aggregate Power Profiles
# ============================================================================
def fig_aggregate_profiles(sim):
    print("  Generating fig_aggregate_profiles...")
    p = sim['p']
    L_ref_total = np.sum(p['L'], axis=0)
    L_adj_total = np.sum(sim['L_adj'], axis=0)
    PV_total = np.sum(p['PV'], axis=0)

    hours = list(range(24))
    L_ref_h = hourly_avg(L_ref_total)
    L_adj_h = hourly_avg(L_adj_total)
    PV_h = hourly_avg(PV_total)

    hrs_f, L_ref_s = smooth_curve(hours, L_ref_h)
    _, L_adj_s = smooth_curve(hours, L_adj_h)
    _, PV_s = smooth_curve(hours, PV_h)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Deficit fill
    ax.fill_between(hrs_f, L_adj_s, np.minimum(PV_s, L_adj_s),
                    alpha=0.3, color='#ff6b6b', label='Deficit (Grid Import)')
    # Surplus fill
    ax.fill_between(hrs_f, PV_s, np.minimum(PV_s, L_adj_s),
                    alpha=0.3, color='#51cf66', label='Excess PV (P2P Available)')

    ax.plot(hrs_f, PV_s, color='#2ca02c', linewidth=2, label='Total PV Generation')
    ax.plot(hrs_f, L_ref_s, color='#7f7f7f', linewidth=1.8, linestyle='--', label='Reference Load (no DSM)')
    ax.plot(hrs_f, L_adj_s, color='#1a1a1a', linewidth=2.2, label='Adjusted Load (with DSM)')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power (kW)')
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', framealpha=0.9)
    plt.savefig('figures/fig_aggregate_profiles.pdf')
    plt.close()


# ============================================================================
# FIGURE 2: Individual Load Reshaping
# ============================================================================
def fig_individual_reshaping(sim):
    print("  Generating fig_individual_reshaping...")
    p = sim['p']
    nc = sim['nc']
    hours = list(range(24))

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=False)

    # Top row: consumers
    for col, idx in enumerate(range(min(3, nc))):
        ax = axes[0, col]
        ref_h = hourly_avg(p['L'][idx])
        adj_h = hourly_avg(sim['L_adj'][idx])
        ax.fill_between(hours, ref_h, adj_h, alpha=0.2, color='#636EFA')
        ax.plot(hours, ref_h, '--', color='#AB63FA', linewidth=1.5, label='Reference')
        ax.plot(hours, adj_h, color='#636EFA', linewidth=2, label='Adjusted')
        ax.set_title(f'Consumer {col+1}')
        ax.set_xlabel('Hour')
        if col == 0:
            ax.set_ylabel('Power (kW)')
            ax.legend(fontsize=8)

    # Bottom row: prosumers
    prosumer_indices = [nc, nc+2, nc+4]
    for col, idx in enumerate(prosumer_indices[:3]):
        ax = axes[1, col]
        ref_h = hourly_avg(p['L'][idx])
        adj_h = hourly_avg(sim['L_adj'][idx])
        pv_h = hourly_avg(p['PV'][idx])
        ax.fill_between(hours, ref_h, adj_h, alpha=0.2, color='#636EFA')
        ax.plot(hours, ref_h, '--', color='#AB63FA', linewidth=1.5, label='Reference')
        ax.plot(hours, adj_h, color='#636EFA', linewidth=2, label='Adjusted')
        ax.plot(hours, pv_h, color='#00CC96', linewidth=1.8, label='PV')
        ax.set_title(f'Prosumer {col+1}')
        ax.set_xlabel('Hour')
        if col == 0:
            ax.set_ylabel('Power (kW)')
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/fig_individual_reshaping.pdf')
    plt.close()


# ============================================================================
# FIGURE 3: SDR Dynamics
# ============================================================================
def fig_sdr_hourly(sim):
    print("  Generating fig_sdr_hourly...")
    hours = list(range(24))
    SDR_ts = sim['SDR_dsm']

    sdr_hourly = np.zeros(24)
    count = np.zeros(24)
    for d in range(DAYS):
        for h in range(24):
            for slot in range(4):
                idx = d * PERIODS_PER_DAY + h * 4 + slot
                val = SDR_ts[idx]
                if np.isfinite(val):
                    sdr_hourly[h] += val
                    count[h] += 1
    mask = count > 0
    sdr_hourly[mask] /= count[mask]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hours, sdr_hourly, 'o-', color='#1565C0', linewidth=2, markersize=5)
    ax.axhline(y=1.0, linestyle='--', color='red', linewidth=1, alpha=0.7, label='SDR = 1 (balanced)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Supply-Demand Ratio')
    ax.set_xlim(0, 23)
    ax.legend()
    plt.savefig('figures/fig_sdr_hourly.pdf')
    plt.close()


# ============================================================================
# FIGURE 4: Comparative Price Analysis (all mechanisms)
# ============================================================================
def fig_prices_comparison(sim):
    print("  Generating fig_prices_comparison...")
    hours = list(range(24))
    lbuy = sim['lbuy']
    lsell = sim['lsell']

    buy_mmr = hourly_avg(sim['c_im_mmr'])
    sell_mmr = hourly_avg(sim['c_ex_mmr'])
    buy_sdr = hourly_avg(sim['Pr_buy_sdr'])
    sell_sdr = hourly_avg(sim['Pr_sell_sdr'])
    buy_dsm = hourly_avg(sim['Pr_buy_dsm'])
    sell_dsm = hourly_avg(sim['Pr_sell_dsm'])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(y=lbuy, linestyle=':', color='black', linewidth=1, alpha=0.5, label=f'Grid Buy (₹{lbuy:.2f})')
    ax.axhline(y=lsell, linestyle=':', color='black', linewidth=1, alpha=0.5, label=f'Grid Sell (₹{lsell:.2f})')

    ax.plot(hours, buy_mmr, 's-', color=COLORS['MMR'], linewidth=1.5, markersize=4, label='MMR Buy')
    ax.plot(hours, buy_sdr, '^-', color=COLORS['SDR'], linewidth=1.5, markersize=4, label='SDR Buy')
    ax.plot(hours, buy_dsm, 'o-', color=COLORS['SDR-DSM'], linewidth=1.5, markersize=4, label='SDR-DSM Buy')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Price (₹/kWh)')
    ax.set_xlim(0, 23)
    ax.legend(loc='upper right', fontsize=8)
    plt.savefig('figures/fig_prices_comparison.pdf')
    plt.close()


# ============================================================================
# FIGURE 5: SDR-DSM P2P Prices with spread shading
# ============================================================================
def fig_prices_sdr_dsm(sim):
    print("  Generating fig_prices_sdr_dsm...")
    hours = list(range(24))
    buy_h = hourly_avg(sim['Pr_buy_dsm'])
    sell_h = hourly_avg(sim['Pr_sell_dsm'])
    lbuy = sim['lbuy']
    lsell = sim['lsell']

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.fill_between(hours, buy_h, sell_h, alpha=0.15, color='#1f77b4', label='P2P Spread')
    ax.plot(hours, buy_h, 'o-', color='#1565C0', linewidth=2, markersize=5, label='P2P Buy Price')
    ax.plot(hours, sell_h, 's-', color='#C62828', linewidth=2, markersize=5, label='P2P Sell Price')
    ax.axhline(y=lbuy, linestyle=':', color='#1565C0', linewidth=1.2, label=f'Grid Buy ₹{lbuy:.2f}')
    ax.axhline(y=lsell, linestyle=':', color='#C62828', linewidth=1.2, label=f'Grid Sell ₹{lsell:.2f}')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Price (₹/kWh)')
    ax.set_xlim(0, 23)
    ax.legend(loc='upper right', fontsize=8)
    plt.savefig('figures/fig_prices_sdr_dsm.pdf')
    plt.close()


# ============================================================================
# FIGURE 6: Alpha Sensitivity
# ============================================================================
def fig_alpha_sensitivity():
    print("  Generating fig_alpha_sensitivity...")
    df = pd.read_csv('alpha_sweep_results.csv')

    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax2 = ax1.twinx()

    ax1.plot(df['alpha'], df['community_savings_pct'], 'o-', color='#1f77b4', linewidth=2, label='Community Savings (%)')
    ax1.plot(df['alpha'], df['fairness'] * 100, 's-', color='#2ca02c', linewidth=2, label='Fairness × 100')
    ax2.plot(df['alpha'], df['energy_shifted_kwh'], '^--', color='#d62728', linewidth=1.5, label='Energy Shifted (kWh)')

    ax1.set_xlabel('Discomfort Coefficient α (₹/kW)')
    ax1.set_ylabel('Savings (%) / Fairness Index ×100')
    ax2.set_ylabel('Energy Shifted (kWh)', color='#d62728')
    ax1.set_xscale('log')
    ax1.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='#d62728')

    plt.savefig('figures/fig_alpha_sensitivity.pdf')
    plt.close()


# ============================================================================
# FIGURE 7: Prosumer Ratio Sweep
# ============================================================================
def fig_prosumer_ratio():
    print("  Generating fig_prosumer_ratio...")
    df = pd.read_csv('prosumer_ratio_sweep_results.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for mech in ['BSM', 'MMR', 'SDR', 'SDR-DSM']:
        mdf = df[df['mechanism'] == mech]
        ax1.plot(mdf['prosumer_fraction'] * 100, mdf['community_savings_pct'],
                 'o-', color=COLORS[mech], linewidth=2, label=mech)
        ax2.plot(mdf['prosumer_fraction'] * 100, mdf['fairness'],
                 's-', color=COLORS[mech], linewidth=2, label=mech)

    ax1.set_xlabel('Prosumer Fraction (%)')
    ax1.set_ylabel('Community Savings (%)')
    ax1.legend(fontsize=8)

    ax2.set_xlabel('Prosumer Fraction (%)')
    ax2.set_ylabel('Fairness Index')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/fig_prosumer_ratio.pdf')
    plt.close()


# ============================================================================
# FIGURE 8: Tariff Spread Sensitivity
# ============================================================================
def fig_tariff_spread():
    print("  Generating fig_tariff_spread...")
    df = pd.read_csv('tariff_sweep_results.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for mech in ['BSM', 'MMR', 'SDR', 'SDR-DSM']:
        mdf = df[df['mechanism'] == mech]
        ax1.plot(mdf['spread'], mdf['community_savings_pct'],
                 'o-', color=COLORS[mech], linewidth=2, label=mech)
        ax2.plot(mdf['spread'], mdf['prosumer_delta_pct'],
                 's-', color=COLORS[mech], linewidth=2, label=mech)

    ax1.set_xlabel('Buy-Sell Spread (₹/kWh)')
    ax1.set_ylabel('Community Savings (%)')
    ax1.legend(fontsize=8)

    ax2.set_xlabel('Buy-Sell Spread (₹/kWh)')
    ax2.set_ylabel('Prosumer Savings (%)')
    ax2.axhline(y=0, linestyle='--', color='gray', linewidth=0.8)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/fig_tariff_spread.pdf')
    plt.close()


# ============================================================================
# FIGURE 9: Real Data Load Profiles
# ============================================================================
def fig_real_load_profiles():
    print("  Generating fig_real_load_profiles...")
    from run_real_data import (load_meter_data, aggregate_to_15min, find_best_window,
                               clean_and_filter, build_load_matrix, BAREILLY_FILE, MATHURA_FILE)

    df = load_meter_data(BAREILLY_FILE, MATHURA_FILE)
    df = aggregate_to_15min(df)
    start_date = find_best_window(df, num_days=30)
    pivot = clean_and_filter(df, start_date, num_days=30)

    all_meters = list(pivot.columns)
    L_all = build_load_matrix(pivot, all_meters)
    N, H = L_all.shape
    num_days = H // PERIODS_PER_DAY

    # Hourly average per meter
    avg_profile = np.mean(L_all, axis=0).reshape(num_days, PERIODS_PER_DAY)
    daily_hourly = np.zeros((num_days, 24))
    for h in range(24):
        daily_hourly[:, h] = avg_profile[:, h*4:(h+1)*4].mean(axis=1)

    mean_profile = daily_hourly.mean(axis=0)
    std_profile = daily_hourly.std(axis=0)

    # Synthetic comparison
    synth_shape = generate_daily_load_shape()
    synth_hourly = np.zeros(24)
    for h in range(24):
        synth_hourly[h] = synth_shape[h*4:(h+1)*4].mean()
    # Normalize synthetic to same energy as real
    synth_hourly *= mean_profile.sum() / synth_hourly.sum()

    hours = list(range(24))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.fill_between(hours, mean_profile - std_profile, mean_profile + std_profile,
                    alpha=0.2, color='#1f77b4', label='Real ± 1σ')
    ax.plot(hours, mean_profile, 'o-', color='#1f77b4', linewidth=2, markersize=4, label='Real (CEEW avg)')
    ax.plot(hours, synth_hourly, 's--', color='#d62728', linewidth=1.5, markersize=4, label='Synthetic (Gaussian peaks)')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Load per Meter (kW)')
    ax.set_xlim(0, 23)
    ax.legend()
    plt.savefig('figures/fig_real_load_profiles.pdf')
    plt.close()


# ============================================================================
# FIGURE 10: Convergence Plot
# ============================================================================
def fig_convergence():
    print("  Generating fig_convergence...")
    df = pd.read_csv('convergence_analysis.csv')

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax2 = ax1.twinx()

    ax1.plot(df['iteration'], df['community_savings_pct'], 'o-', color='#1f77b4', linewidth=2, label='Community Savings (%)')
    ax1.plot(df['iteration'], df['fairness'] * 100, 's-', color='#2ca02c', linewidth=2, label='Fairness ×100')
    ax2.plot(df['iteration'], df['max_change_kw'], '^--', color='#d62728', linewidth=1.5, label='Max Load Change (kW)')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Savings (%) / Fairness ×100')
    ax2.set_ylabel('Max Load Change (kW)', color='#d62728')
    ax1.legend(loc='center right', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_xlim(1, len(df))

    plt.savefig('figures/fig_convergence.pdf')
    plt.close()


# ============================================================================
# FIGURE 11: Community Size Sweep
# ============================================================================
def fig_community_sweep():
    print("  Generating fig_community_sweep...")
    df = pd.read_csv('community_sweep_results.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for mech in ['BSM', 'MMR', 'SDR', 'SDR-DSM']:
        mdf = df[df['mechanism'] == mech]
        ax1.plot(mdf['N'], mdf['community_savings_pct'],
                 'o-', color=COLORS[mech], linewidth=2, label=mech)
        ax2.plot(mdf['N'], mdf['fairness'],
                 's-', color=COLORS[mech], linewidth=2, label=mech)

    ax1.set_xlabel('Community Size (N)')
    ax1.set_ylabel('Community Savings (%)')
    ax1.legend(fontsize=8)

    ax2.set_xlabel('Community Size (N)')
    ax2.set_ylabel('Fairness Index')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/fig_community_sweep.pdf')
    plt.close()


# ============================================================================
# FIGURE 12: Real Data Community Sweep
# ============================================================================
def fig_real_community_sweep():
    print("  Generating fig_real_community_sweep...")
    df = pd.read_csv('real_data_sweep_results.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for mech in ['BSM', 'MMR', 'SDR', 'SDR-DSM']:
        mdf = df[df['mechanism'] == mech]
        ax1.plot(mdf['prosumer_fraction'] * 100, mdf['community_savings_pct'],
                 'o-', color=COLORS[mech], linewidth=2, label=mech)
        ax2.plot(mdf['prosumer_fraction'] * 100, mdf['fairness'],
                 's-', color=COLORS[mech], linewidth=2, label=mech)

    ax1.set_xlabel('Prosumer Fraction (%)')
    ax1.set_ylabel('Community Savings (%)')
    ax1.set_title('Real Data (N=56, CEEW)')
    ax1.legend(fontsize=8)

    ax2.set_xlabel('Prosumer Fraction (%)')
    ax2.set_ylabel('Fairness Index')
    ax2.set_title('Real Data (N=56, CEEW)')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/fig_real_community_sweep.pdf')
    plt.close()


# ============================================================================
# FIGURE 13: Savings by Group (bar chart for primary case)
# ============================================================================
def fig_savings_by_group(sim):
    print("  Generating fig_savings_by_group...")
    p = sim['p']
    nc = sim['nc']
    lbuy = sim['lbuy']
    lsell = sim['lsell']

    bills_conv, _ = settle_conventional(p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'])
    bills_mmr, _, _, _ = settle_mmr(p['L'], p['PV'], DT, lbuy, lsell, p['p2p_allowed'])
    bills_bs, _, _ = settle_billsharing(p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'], lbuy, lsell, p['p2p_allowed'])
    bills_sdr, _, _, _, _ = settle_sdr(p['L'], p['PV'], DT, lbuy, lsell, p['p2p_allowed'])
    bills_dsm, _, _, _, _, _ = settle_sdr_dsm(p['L'], p['PV'], DT, lbuy, lsell, p['p2p_allowed'])

    consumer_conv = bills_conv[:nc].sum()
    prosumer_conv = bills_conv[nc:].sum()

    mechanisms = ['BSM', 'MMR', 'SDR', 'SDR-DSM']
    all_bills = [bills_bs, bills_mmr, bills_sdr, bills_dsm]

    consumer_savings = []
    prosumer_savings = []
    for bills in all_bills:
        consumer_savings.append(((consumer_conv - bills[:nc].sum()) / abs(consumer_conv)) * 100)
        prosumer_savings.append(((prosumer_conv - bills[nc:].sum()) / abs(prosumer_conv)) * 100)

    x = np.arange(len(mechanisms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width/2, consumer_savings, width, label='Consumers', color='#2ca02c', alpha=0.8)
    bars2 = ax.bar(x + width/2, prosumer_savings, width, label='Prosumers', color='#1f77b4', alpha=0.8)

    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='-')
    ax.set_xlabel('Pricing Mechanism')
    ax.set_ylabel('Savings vs Conventional (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms)
    ax.legend()

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.savefig('figures/fig_savings_by_group.pdf')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Generating Journal Figures")
    print("=" * 60)

    print("\nRunning primary simulation (N=20)...")
    sim = get_primary_simulation()

    print("\nGenerating figures...")
    fig_aggregate_profiles(sim)
    fig_individual_reshaping(sim)
    fig_sdr_hourly(sim)
    fig_prices_comparison(sim)
    fig_prices_sdr_dsm(sim)
    fig_alpha_sensitivity()
    fig_prosumer_ratio()
    fig_tariff_spread()
    fig_convergence()
    fig_community_sweep()
    fig_real_community_sweep()
    fig_savings_by_group(sim)

    print("\nGenerating real data figure (loading CEEW data)...")
    fig_real_load_profiles()

    print("\n" + "=" * 60)
    print(f"Done! Generated {len(os.listdir('figures'))} figures in journal/figures/")
