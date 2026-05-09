"""
Generate all figures for the IEEE conference paper.
Uses the same simulation functions as run_scenarios.py (3C+7P, seed=42).

Figures:
  fig1.png  - P2P trading concept (community microgrid schematic)
  fig7.png  - Community aggregate power profiles (reference vs adjusted)
  fig8.png  - Consumer load reshaping under SDR-DSM
  fig11.png - Hourly SDR variation
  fig10.png - Internal P2P market clearing prices vs grid tariffs
  fig12.png - Comparative price validation (all 4 mechanisms)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_scenarios import (
    generate_profiles, settle_conventional, settle_mmr,
    settle_billsharing, settle_sdr, settle_sdr_dsm,
    DAYS, PERIODS_PER_DAY, DT
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 6,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def run_main_scenario():
    """Run the 3C+7P scenario and return all data needed for figures."""
    p = generate_profiles(3, 7, seed=42)

    bills_conv, cost_conv = settle_conventional(
        p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'])

    bills_mmr, cost_mmr, c_im_mmr, c_ex_mmr = settle_mmr(
        p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])

    bills_bs, cost_bs, cp2p_bs = settle_billsharing(
        p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'],
        p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])

    bills_sdr, cost_sdr, Pr_buy_sdr, Pr_sell_sdr, SDR_ts = settle_sdr(
        p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])

    bills_sdr_dsm, cost_sdr_dsm, Pr_buy_dsm, Pr_sell_dsm, SDR_ts_dsm, iters = settle_sdr_dsm(
        p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])

    # Also get adjusted loads from DSM (re-run to capture L_adj)
    L_adj = get_adjusted_loads(p)

    return {
        'p': p,
        'L_adj': L_adj,
        'SDR_ts': SDR_ts,
        'SDR_ts_dsm': SDR_ts_dsm,
        'Pr_buy_sdr': Pr_buy_sdr,
        'Pr_sell_sdr': Pr_sell_sdr,
        'Pr_buy_dsm': Pr_buy_dsm,
        'Pr_sell_dsm': Pr_sell_dsm,
        'c_im_mmr': c_im_mmr,
        'c_ex_mmr': c_ex_mmr,
        'cp2p_bs': cp2p_bs,
    }


def get_adjusted_loads(p):
    """Re-run SDR-DSM and capture adjusted loads."""
    L_ref = p['L'].copy()
    PV = p['PV']
    lbuy = p['lambda_buy_ref']
    lsell = p['lambda_sell_ref']
    p2p_allowed = p['p2p_allowed']
    N, H = L_ref.shape
    L_adj = L_ref.copy()
    damping = 0.3
    alpha = 0.12

    for iteration in range(8):
        L_prev = L_adj.copy()
        _, _, Pr_buy, Pr_sell, SDR_ts = settle_sdr(L_adj, PV, DT, lbuy, lsell, p2p_allowed)

        for i in range(N):
            prices = np.zeros(H)
            NP_i = L_adj[i] - PV[i]
            for h in range(H):
                if NP_i[h] > 0:
                    prices[h] = Pr_buy[h]
                else:
                    prices[h] = -Pr_sell[h]

            price_signal = prices * DT
            adjustment = -price_signal / (2 * alpha)
            new_load = L_ref[i] + adjustment

            for d in range(DAYS):
                s = d * PERIODS_PER_DAY
                e = (d + 1) * PERIODS_PER_DAY
                day_ref_energy = np.sum(L_ref[i, s:e]) * DT
                new_load[s:e] = np.clip(new_load[s:e], 0.5 * L_ref[i, s:e], 1.5 * L_ref[i, s:e])
                day_new_energy = np.sum(new_load[s:e]) * DT
                if day_new_energy > 0:
                    new_load[s:e] *= day_ref_energy / day_new_energy

            L_adj[i] = (1 - damping) * L_prev[i] + damping * new_load

    return L_adj


def hourly_avg(ts):
    """Convert 15-min time series to hourly averages over 30 days."""
    reshaped = ts.reshape(DAYS, PERIODS_PER_DAY)
    hourly = np.zeros((DAYS, 24))
    for h in range(24):
        hourly[:, h] = reshaped[:, h*4:(h+1)*4].mean(axis=1)
    return hourly.mean(axis=0)


def hourly_avg_sum(ts_2d):
    """Sum across households then hourly average."""
    total = np.sum(ts_2d, axis=0)
    return hourly_avg(total)


def generate_fig1():
    """P2P trading concept - community microgrid schematic."""
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # ESP/Aggregator in center
    esp = FancyBboxPatch((3.5, 3.2), 3, 1.6, boxstyle="round,pad=0.1",
                         facecolor='#2196F3', edgecolor='black', linewidth=1.5)
    ax.add_patch(esp)
    ax.text(5, 4.0, 'Energy Service\nProvider (ESP)', ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')

    # Grid
    grid = FancyBboxPatch((3.8, 6.5), 2.4, 1.0, boxstyle="round,pad=0.1",
                          facecolor='#FF9800', edgecolor='black', linewidth=1.2)
    ax.add_patch(grid)
    ax.text(5, 7.0, 'Utility Grid', ha='center', va='center', fontsize=8, fontweight='bold')

    # Prosumers (left)
    for i, y in enumerate([0.5, 2.0]):
        box = FancyBboxPatch((0.3, y), 2.0, 1.0, boxstyle="round,pad=0.1",
                             facecolor='#4CAF50', edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(1.3, y+0.5, f'Prosumer {i+1}\n(PV+Load)', ha='center', va='center', fontsize=6)

    # Consumers (right)
    for i, y in enumerate([0.5, 2.0]):
        box = FancyBboxPatch((7.7, y), 2.0, 1.0, boxstyle="round,pad=0.1",
                             facecolor='#F44336', edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(8.7, y+0.5, f'Consumer {i+1}\n(Load only)', ha='center', va='center', fontsize=6)

    # Arrows
    ax.annotate('', xy=(5, 6.5), xytext=(5, 4.8),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.annotate('', xy=(3.5, 3.5), xytext=(2.3, 1.5),
                arrowprops=dict(arrowstyle='<->', color='#4CAF50', lw=1.2))
    ax.annotate('', xy=(3.5, 3.8), xytext=(2.3, 2.8),
                arrowprops=dict(arrowstyle='<->', color='#4CAF50', lw=1.2))
    ax.annotate('', xy=(6.5, 3.5), xytext=(7.7, 1.5),
                arrowprops=dict(arrowstyle='<->', color='#F44336', lw=1.2))
    ax.annotate('', xy=(6.5, 3.8), xytext=(7.7, 2.8),
                arrowprops=dict(arrowstyle='<->', color='#F44336', lw=1.2))

    ax.text(5, 5.5, 'Grid Import/Export', ha='center', fontsize=6, style='italic')
    ax.text(2.2, 3.3, 'P2P\nSell', ha='center', fontsize=6, color='#4CAF50')
    ax.text(7.8, 3.3, 'P2P\nBuy', ha='center', fontsize=6, color='#F44336')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1.png'))
    plt.close()
    print("  fig1.png - concept diagram")


def generate_fig7(data):
    """Community aggregate power profiles (reference vs adjusted)."""
    p = data['p']
    L_adj = data['L_adj']

    L_ref_total = np.sum(p['L'], axis=0)
    L_adj_total = np.sum(L_adj, axis=0)
    PV_total = np.sum(p['PV'], axis=0)

    # Hourly averages
    hours = np.arange(24)
    L_ref_h = hourly_avg(L_ref_total)
    L_adj_h = hourly_avg(L_adj_total)
    PV_h = hourly_avg(PV_total)

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(hours, L_ref_h, 'b-', linewidth=1.5, label='Reference Load')
    ax.plot(hours, L_adj_h, 'r--', linewidth=1.5, label='Adjusted Load (SDR-DSM)')
    ax.plot(hours, PV_h, 'g-.', linewidth=1.5, label='PV Generation')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Power (kW)')
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig7.png'))
    plt.close()
    print("  fig7.png - aggregate power profiles")


def generate_fig8(data):
    """Consumer load reshaping under SDR-DSM."""
    p = data['p']
    L_adj = data['L_adj']

    # Pick 2 consumers (indices 0, 1) and 1 prosumer (index 3)
    hours = np.arange(24)

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2), sharey=True)

    for ax_idx, hh_idx in enumerate([0, 1]):
        ref_h = hourly_avg(p['L'][hh_idx])
        adj_h = hourly_avg(L_adj[hh_idx])
        axes[ax_idx].plot(hours, ref_h, 'b-', linewidth=1.2, label='Reference')
        axes[ax_idx].plot(hours, adj_h, 'r--', linewidth=1.2, label='Adjusted')
        axes[ax_idx].set_xlabel('Hour')
        axes[ax_idx].set_title(f'Consumer {hh_idx+1}', fontsize=9)
        axes[ax_idx].set_xlim(0, 23)
        axes[ax_idx].set_xticks([0, 6, 12, 18])
        axes[ax_idx].grid(True, alpha=0.3)

    axes[0].set_ylabel('Load (kW)')
    axes[0].legend(loc='upper left', fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig8.png'))
    plt.close()
    print("  fig8.png - consumer load reshaping")


def generate_fig11(data):
    """Hourly SDR variation."""
    SDR_ts = data['SDR_ts_dsm']
    hours = np.arange(24)

    # Compute hourly average SDR
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

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.bar(hours, sdr_hourly, color='#2196F3', alpha=0.7, width=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='SDR = 1 (balanced)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Supply-Demand Ratio')
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig11.png'))
    plt.close()

    peak_sdr = sdr_hourly.max()
    peak_hour = hours[np.argmax(sdr_hourly)]
    print(f"  fig11.png - SDR dynamics (peak={peak_sdr:.2f} at hour {peak_hour})")
    return peak_sdr


def generate_fig10(data):
    """Internal P2P market clearing prices vs grid tariffs."""
    p = data['p']
    Pr_buy = data['Pr_buy_dsm']
    Pr_sell = data['Pr_sell_dsm']
    lbuy = p['lambda_buy_ref']
    lsell = p['lambda_sell_ref']

    hours = np.arange(24)
    buy_h = hourly_avg(Pr_buy)
    sell_h = hourly_avg(Pr_sell)

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(hours, buy_h, 'b-o', linewidth=1.2, markersize=3, label='P2P Buy Price')
    ax.plot(hours, sell_h, 'r-s', linewidth=1.2, markersize=3, label='P2P Sell Price')
    ax.axhline(y=lbuy, color='blue', linestyle=':', linewidth=0.8, alpha=0.6, label=f'Grid Buy ({lbuy:.2f})')
    ax.axhline(y=lsell, color='red', linestyle=':', linewidth=0.8, alpha=0.6, label=f'Grid Sell ({lsell:.2f})')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Price (Rs/kWh)')
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.legend(loc='right', fontsize=6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig10.png'))
    plt.close()
    print(f"  fig10.png - P2P prices (buy range: {buy_h.min():.4f}-{buy_h.max():.4f})")


def generate_fig12(data):
    """Comparative price validation (all 4 mechanisms)."""
    p = data['p']
    lbuy = p['lambda_buy_ref']
    lsell = p['lambda_sell_ref']

    hours = np.arange(24)

    # SDR buy prices
    sdr_buy_h = hourly_avg(data['Pr_buy_dsm'])

    # MMR buy prices
    mmr_buy_h = hourly_avg(data['c_im_mmr'])

    # BSM prices
    bs_h = hourly_avg(data['cp2p_bs'])

    # Conventional is flat at grid buy
    conv_h = np.full(24, lbuy)

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(hours, conv_h, 'k-', linewidth=1.0, label='Conventional')
    ax.plot(hours, bs_h, 'g-^', linewidth=1.0, markersize=3, label='BSM')
    ax.plot(hours, mmr_buy_h, 'm-d', linewidth=1.0, markersize=3, label='MMR')
    ax.plot(hours, sdr_buy_h, 'b-o', linewidth=1.2, markersize=3, label='SDR-DSM')
    ax.axhline(y=lsell, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label=f'Export floor ({lsell:.2f})')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Buy Price (Rs/kWh)')
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.legend(loc='lower left', fontsize=6)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig12.png'))
    plt.close()
    print("  fig12.png - comparative price validation")


if __name__ == "__main__":
    print("Running main scenario (3C+7P, seed=42)...")
    data = run_main_scenario()
    print("\nGenerating figures:")
    generate_fig1()
    generate_fig7(data)
    generate_fig8(data)
    peak_sdr = generate_fig11(data)
    generate_fig10(data)
    generate_fig12(data)
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print(f"\nKey values for paper text:")
    print(f"  SDR peak: {peak_sdr:.2f}")
    print(f"  Grid buy: {data['p']['lambda_buy_ref']:.4f}")
    print(f"  Grid sell: {data['p']['lambda_sell_ref']:.4f}")
