"""Real data community composition sweep.
Uses CEEW smart meter data, varies prosumer fraction: 20%, 40%, 60%, 80%.
All 5 mechanisms.
"""
import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from run_real_data import (
    load_meter_data, aggregate_to_15min, find_best_window, clean_and_filter,
    build_load_matrix, generate_pv_for_real_loads, assign_tariffs,
    settle_conventional, settle_mmr, settle_billsharing, settle_sdr, settle_sdr_dsm,
    BAREILLY_FILE, MATHURA_FILE, PERIODS_PER_DAY, DT
)


def run_real_data_sweep():
    print("Loading real meter data...")
    df = load_meter_data(BAREILLY_FILE, MATHURA_FILE)
    print(f"  Loaded {len(df)} rows, {df['meter_id'].nunique()} meters")

    print("Aggregating to 15-min...")
    df = aggregate_to_15min(df)

    print("Finding best 30-day window...")
    start_date = find_best_window(df, num_days=30)

    print("Cleaning and filtering...")
    pivot = clean_and_filter(df, start_date, num_days=30)

    N_available = pivot.shape[1]
    print(f"  Available meters: {N_available}")

    prosumer_fractions = [0.20, 0.40, 0.60, 0.80]
    results = []

    for frac in prosumer_fractions:
        num_prosumers = int(frac * N_available)
        num_consumers = N_available - num_prosumers
        N = N_available
        num_days = pivot.shape[0] // PERIODS_PER_DAY

        print(f"\n  Running {num_consumers}C + {num_prosumers}P ({frac*100:.0f}% prosumer)...")

        # Assign roles: top consumers become prosumers
        total_consumption = pivot.sum().sort_values(ascending=False)
        all_meters = list(total_consumption.index)
        prosumer_ids = all_meters[:num_prosumers]
        consumer_ids = all_meters[num_prosumers:]
        ordered_ids = consumer_ids + prosumer_ids

        L = build_load_matrix(pivot, ordered_ids)
        PV = generate_pv_for_real_loads(L, num_consumers, num_prosumers, num_days, seed=42)

        lambda_buy_ind, lambda_sell_ind, lambda_buy_ref, lambda_sell_ref = assign_tariffs(N, num_consumers, seed=42)

        H = L.shape[1]
        p2p_allowed = np.ones(H, dtype=bool)
        np.random.seed(142)
        for _ in range(3):
            s = np.random.randint(0, H - 8)
            p2p_allowed[s:s + 8] = False

        nc = num_consumers

        bills_conv, _ = settle_conventional(L, PV, DT, lambda_buy_ind, lambda_sell_ind)
        bills_mmr, _, _, _ = settle_mmr(L, PV, DT, lambda_buy_ref, lambda_sell_ref, p2p_allowed)
        bills_bs, _, _ = settle_billsharing(L, PV, DT, lambda_buy_ind, lambda_sell_ind,
                                             lambda_buy_ref, lambda_sell_ref, p2p_allowed)
        bills_sdr, _, _, _, _ = settle_sdr(L, PV, DT, lambda_buy_ref, lambda_sell_ref, p2p_allowed)
        bills_dsm, _, _, _, _, iters = settle_sdr_dsm(L, PV, DT, lambda_buy_ref, lambda_sell_ref, p2p_allowed)

        total_conv = bills_conv.sum()
        consumer_conv = bills_conv[:nc].sum()
        prosumer_conv = bills_conv[nc:].sum()

        for name, bills in [('BSM', bills_bs), ('MMR', bills_mmr), ('SDR', bills_sdr), ('SDR-DSM', bills_dsm)]:
            total = bills.sum()
            consumer = bills[:nc].sum()
            prosumer = bills[nc:].sum()

            cs = ((total_conv - total) / abs(total_conv)) * 100
            cd = ((consumer_conv - consumer) / abs(consumer_conv)) * 100 if abs(consumer_conv) > 1e-9 else 0
            pd_ = ((prosumer_conv - prosumer) / abs(prosumer_conv)) * 100 if abs(prosumer_conv) > 1e-9 else 0

            denom = abs(cd) + abs(pd_)
            fairness = 1.0 - abs(cd - pd_) / (denom + 1e-9) if denom > 1e-9 else 1.0

            results.append({
                'config': f"{nc}C+{num_prosumers}P",
                'N': N,
                'prosumer_fraction': frac,
                'mechanism': name,
                'community_savings_pct': cs,
                'consumer_delta_pct': cd,
                'prosumer_delta_pct': pd_,
                'fairness': fairness,
                'total_conv_cost': total_conv,
            })

        print(f"    SDR-DSM: savings={results[-1]['community_savings_pct']:.2f}%, F={results[-1]['fairness']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv('real_data_sweep_results.csv', index=False)
    print(f"\nSaved real_data_sweep_results.csv ({len(df)} rows)")

    # Also save load profile statistics for the paper
    print("\nComputing load profile statistics...")
    all_meters = list(pivot.columns)
    L_all = build_load_matrix(pivot, all_meters)
    stats = {
        'num_meters': N_available,
        'num_days': pivot.shape[0] // PERIODS_PER_DAY,
        'avg_daily_load_kwh': np.sum(L_all * DT) / (N_available * (pivot.shape[0] // PERIODS_PER_DAY)),
        'std_daily_load_kwh': np.std([np.sum(L_all[:, d*PERIODS_PER_DAY:(d+1)*PERIODS_PER_DAY] * DT)
                                       for d in range(pivot.shape[0] // PERIODS_PER_DAY)]) / N_available,
        'peak_hour_load': np.argmax(np.mean(L_all, axis=0).reshape(-1, PERIODS_PER_DAY).mean(axis=0)) * 0.25,
        'min_hour_load': np.argmin(np.mean(L_all, axis=0).reshape(-1, PERIODS_PER_DAY).mean(axis=0)) * 0.25,
    }
    pd.DataFrame([stats]).to_csv('real_data_stats.csv', index=False)
    print(f"  Avg daily load per meter: {stats['avg_daily_load_kwh']:.2f} kWh")
    print(f"  Peak hour: {stats['peak_hour_load']:.1f}")

    # Save hourly average load profile for plotting
    avg_load_profile = np.mean(L_all, axis=0).reshape(-1, PERIODS_PER_DAY).mean(axis=0)
    hourly_profile = np.zeros(24)
    for h in range(24):
        hourly_profile[h] = avg_load_profile[h*4:(h+1)*4].mean()
    pd.DataFrame({'hour': range(24), 'avg_load_kw': hourly_profile}).to_csv('real_data_hourly_profile.csv', index=False)

    return df


if __name__ == "__main__":
    print("Real Data Community Composition Sweep")
    print("=" * 60)
    run_real_data_sweep()
