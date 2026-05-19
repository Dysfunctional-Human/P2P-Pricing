"""Community size sweep with all 5 mechanisms (including static SDR separately).
Generates data for N=7, 10, 20, 30.
"""
import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from run_scenarios import (
    generate_profiles, settle_conventional, settle_mmr, settle_billsharing,
    settle_sdr, settle_sdr_dsm, DAYS, PERIODS_PER_DAY, DT
)


def run_community_sweep():
    scenarios = [
        (2, 5, 77, "N=7"),
        (3, 7, 42, "N=10"),
        (5, 15, 123, "N=20"),
        (10, 20, 99, "N=30"),
        (15, 35, 201, "N=50"),
    ]

    results = []
    for nc, np_, seed, label in scenarios:
        print(f"  Running {nc}C + {np_}P ({label})...")
        p = generate_profiles(nc, np_, seed)

        bills_conv, _ = settle_conventional(p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'])
        bills_mmr, _, _, _ = settle_mmr(p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])
        bills_bs, _, _ = settle_billsharing(p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'],
                                             p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])
        bills_sdr, _, _, _, _ = settle_sdr(p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])
        bills_dsm, _, _, _, _, iters = settle_sdr_dsm(p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])

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
                'config': f"{nc}C+{np_}P",
                'N': nc + np_,
                'num_consumers': nc,
                'num_prosumers': np_,
                'mechanism': name,
                'community_savings_pct': cs,
                'consumer_delta_pct': cd,
                'prosumer_delta_pct': pd_,
                'fairness': fairness,
            })

    df = pd.DataFrame(results)
    df.to_csv('community_sweep_results.csv', index=False)
    print(f"\nSaved community_sweep_results.csv ({len(df)} rows)")
    return df


if __name__ == "__main__":
    print("Community Size Sweep (N=7, 10, 20, 30)")
    print("=" * 60)
    run_community_sweep()
