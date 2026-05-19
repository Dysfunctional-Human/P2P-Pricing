"""Tariff spread sensitivity: vary grid buy-sell spread.
Fixed N=10 (3C+7P), all 5 mechanisms.
"""
import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from run_scenarios import (
    generate_profiles, settle_conventional, settle_mmr, settle_billsharing,
    settle_sdr, settle_sdr_dsm, DAYS, PERIODS_PER_DAY, DT
)


def run_tariff_sweep():
    tariff_configs = [
        (4.5, 3.5, "Narrow (1.0 Rs)"),
        (5.5, 3.4, "Medium-low (2.1 Rs)"),
        (6.0, 3.4, "Default (2.6 Rs)"),
        (7.0, 3.2, "Medium-high (3.8 Rs)"),
        (8.0, 3.0, "Wide (5.0 Rs)"),
        (10.0, 2.5, "Very wide (7.5 Rs)"),
    ]

    num_consumers, num_prosumers, seed = 8, 12, 42  # N=20

    results = []
    for lbuy, lsell, label in tariff_configs:
        print(f"  Running buy={lbuy}, sell={lsell} ({label})...")

        # Generate profiles with custom tariffs
        p = generate_profiles(num_consumers, num_prosumers, seed)
        nc = num_consumers
        N = p['N']

        # Override tariffs
        lambda_buy_ind = np.full(N, lbuy)
        lambda_sell_ind = np.zeros(N)
        for i in range(nc, N):
            lambda_sell_ind[i] = lsell
        lambda_buy_ref = lbuy
        lambda_sell_ref = lsell

        bills_conv, _ = settle_conventional(p['L'], p['PV'], DT, lambda_buy_ind, lambda_sell_ind)
        bills_mmr, _, _, _ = settle_mmr(p['L'], p['PV'], DT, lambda_buy_ref, lambda_sell_ref, p['p2p_allowed'])
        bills_bs, _, _ = settle_billsharing(p['L'], p['PV'], DT, lambda_buy_ind, lambda_sell_ind,
                                             lambda_buy_ref, lambda_sell_ref, p['p2p_allowed'])
        bills_sdr, _, _, _, _ = settle_sdr(p['L'], p['PV'], DT, lambda_buy_ref, lambda_sell_ref, p['p2p_allowed'])
        bills_dsm, _, _, _, _, iters = settle_sdr_dsm(p['L'], p['PV'], DT, lambda_buy_ref, lambda_sell_ref, p['p2p_allowed'])

        total_conv = bills_conv.sum()
        consumer_conv = bills_conv[:nc].sum()
        prosumer_conv = bills_conv[nc:].sum()

        for name, bills in [('BSM', bills_bs), ('MMR', bills_mmr), ('SDR', bills_sdr), ('SDR-DSM', bills_dsm)]:
            total = bills.sum()
            consumer = bills[:nc].sum()
            prosumer = bills[nc:].sum()

            cs = ((total_conv - total) / abs(total_conv)) * 100
            cd = ((consumer_conv - consumer) / abs(consumer_conv)) * 100 if consumer_conv != 0 else 0
            pd_ = ((prosumer_conv - prosumer) / abs(prosumer_conv)) * 100 if prosumer_conv != 0 else 0

            denom = abs(cd) + abs(pd_)
            fairness = 1.0 - abs(cd - pd_) / (denom + 1e-9) if denom > 1e-9 else 1.0

            results.append({
                'buy_price': lbuy,
                'sell_price': lsell,
                'spread': lbuy - lsell,
                'label': label,
                'mechanism': name,
                'community_savings_pct': cs,
                'consumer_delta_pct': cd,
                'prosumer_delta_pct': pd_,
                'fairness': fairness,
            })

    df = pd.DataFrame(results)
    df.to_csv('tariff_sweep_results.csv', index=False)
    print(f"\nSaved tariff_sweep_results.csv ({len(df)} rows)")
    return df


if __name__ == "__main__":
    print("Tariff Spread Sensitivity Sweep (N=10, 3C+7P)")
    print("=" * 60)
    run_tariff_sweep()
