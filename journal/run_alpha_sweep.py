"""Alpha sensitivity sweep for SDR-DSM mechanism.
Varies discomfort coefficient α from aggressive DSM to effectively no DSM.
"""
import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from run_scenarios import (
    generate_profiles, settle_conventional, settle_sdr, settle_sdr_dsm,
    DAYS, PERIODS_PER_DAY, DT
)


def run_alpha_sweep():
    alphas = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    num_consumers, num_prosumers, seed = 8, 12, 42  # N=20

    p = generate_profiles(num_consumers, num_prosumers, seed)
    nc = num_consumers

    bills_conv, _ = settle_conventional(p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'])
    total_conv = bills_conv.sum()
    consumer_conv = bills_conv[:nc].sum()
    prosumer_conv = bills_conv[nc:].sum()

    # Static SDR (no DSM) as reference
    bills_sdr, _, _, _, _ = settle_sdr(p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])

    results = []
    for alpha in alphas:
        bills_dsm, cost_ts, Pr_buy, Pr_sell, SDR_ts, iters = settle_sdr_dsm(
            p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'],
            p['p2p_allowed'], alpha=alpha
        )

        total_dsm = bills_dsm.sum()
        consumer_dsm = bills_dsm[:nc].sum()
        prosumer_dsm = bills_dsm[nc:].sum()

        community_savings = ((total_conv - total_dsm) / abs(total_conv)) * 100
        consumer_delta = ((consumer_conv - consumer_dsm) / abs(consumer_conv)) * 100
        prosumer_delta = ((prosumer_conv - prosumer_dsm) / abs(prosumer_conv)) * 100

        denom = abs(consumer_delta) + abs(prosumer_delta)
        fairness = 1.0 - abs(consumer_delta - prosumer_delta) / (denom + 1e-9) if denom > 1e-9 else 1.0

        # Compute energy shifted (kWh total over 30 days)
        # We need to re-run to get L_adj - recompute manually
        N, H = p['L'].shape
        L_adj = p['L'].copy()
        damping = 0.3
        for iteration in range(8):
            L_prev = L_adj.copy()
            _, _, Pr_b, Pr_s, _ = settle_sdr(L_adj, p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])
            for i in range(N):
                prices = np.zeros(H)
                NP_i = L_adj[i] - p['PV'][i]
                for h in range(H):
                    prices[h] = Pr_b[h] if NP_i[h] > 0 else -Pr_s[h]
                adjustment = -(prices * DT) / (2 * alpha)
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

        energy_shifted = np.sum(np.abs(L_adj - p['L'])) * DT / 2  # total kWh shifted

        results.append({
            'alpha': alpha,
            'community_savings_pct': community_savings,
            'consumer_delta_pct': consumer_delta,
            'prosumer_delta_pct': prosumer_delta,
            'fairness': fairness,
            'energy_shifted_kwh': energy_shifted,
            'iterations': iters,
        })
        print(f"  α={alpha:.2f}: savings={community_savings:.2f}%, F={fairness:.3f}, shifted={energy_shifted:.1f} kWh")

    df = pd.DataFrame(results)
    df.to_csv('alpha_sweep_results.csv', index=False)
    print(f"\nSaved alpha_sweep_results.csv ({len(df)} rows)")
    return df


if __name__ == "__main__":
    print("Alpha Sensitivity Sweep (N=10, 3C+7P)")
    print("=" * 60)
    run_alpha_sweep()
