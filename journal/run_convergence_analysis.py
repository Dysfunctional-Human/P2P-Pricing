"""Convergence analysis: track iteration-by-iteration metrics for SDR-DSM.
Records max_change, community savings, and fairness at each iteration.
"""
import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from run_scenarios import (
    generate_profiles, settle_conventional, settle_sdr,
    DAYS, PERIODS_PER_DAY, DT
)


def run_convergence_analysis(alpha=0.12, max_iter=15):
    num_consumers, num_prosumers, seed = 8, 12, 42  # N=20
    p = generate_profiles(num_consumers, num_prosumers, seed)
    nc = num_consumers
    N, H = p['L'].shape

    bills_conv, _ = settle_conventional(p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'])
    total_conv = bills_conv.sum()
    consumer_conv = bills_conv[:nc].sum()
    prosumer_conv = bills_conv[nc:].sum()

    L_adj = p['L'].copy()
    damping = 0.3
    lbuy = p['lambda_buy_ref']
    lsell = p['lambda_sell_ref']

    results = []

    for iteration in range(max_iter):
        L_prev = L_adj.copy()

        bills_iter, _, Pr_buy, Pr_sell, SDR_ts = settle_sdr(
            L_adj, p['PV'], DT, lbuy, lsell, p['p2p_allowed']
        )

        # Compute metrics at this iteration
        total = bills_iter.sum()
        consumer = bills_iter[:nc].sum()
        prosumer = bills_iter[nc:].sum()

        cs = ((total_conv - total) / abs(total_conv)) * 100
        cd = ((consumer_conv - consumer) / abs(consumer_conv)) * 100
        pd_ = ((prosumer_conv - prosumer) / abs(prosumer_conv)) * 100
        denom = abs(cd) + abs(pd_)
        fairness = 1.0 - abs(cd - pd_) / (denom + 1e-9) if denom > 1e-9 else 1.0

        # DSM update step
        for i in range(N):
            prices = np.zeros(H)
            NP_i = L_adj[i] - p['PV'][i]
            for h in range(H):
                prices[h] = Pr_buy[h] if NP_i[h] > 0 else -Pr_sell[h]

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

        max_change = np.max(np.abs(L_adj - L_prev))
        energy_shifted = np.sum(np.abs(L_adj - p['L'])) * DT / 2

        results.append({
            'iteration': iteration + 1,
            'max_change_kw': max_change,
            'community_savings_pct': cs,
            'consumer_delta_pct': cd,
            'prosumer_delta_pct': pd_,
            'fairness': fairness,
            'energy_shifted_kwh': energy_shifted,
        })
        print(f"  Iter {iteration+1}: max_change={max_change:.6f} kW, savings={cs:.2f}%, F={fairness:.3f}")

    df = pd.DataFrame(results)
    df.to_csv('convergence_analysis.csv', index=False)
    print(f"\nSaved convergence_analysis.csv ({len(df)} rows)")
    return df


if __name__ == "__main__":
    print("Convergence Analysis (N=10, 3C+7P, α=0.12)")
    print("=" * 60)
    run_convergence_analysis()
