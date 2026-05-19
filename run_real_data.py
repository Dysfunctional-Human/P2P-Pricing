"""
P2P Pricing Framework with Real Indian Smart Meter Data.
Uses CEEW smart meter datasets (Bareilly + Mathura, 2020) instead of synthetic profiles.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from run_scenarios import (
    PERIODS_PER_DAY, DT,
    generate_daily_pv_shape,
    settle_conventional, settle_mmr, settle_billsharing, settle_sdr, settle_sdr_dsm,
)

DATA_DIR = Path(__file__).parent / "electricity-smart-meter-data-from-india"
BAREILLY_FILE = DATA_DIR / "CEEW - Smart meter data Bareilly 2020.csv"
MATHURA_FILE = DATA_DIR / "CEEW - Smart meter data Mathura 2020.csv"


def load_meter_data(bareilly_path, mathura_path):
    df_br = pd.read_csv(bareilly_path, usecols=['x_Timestamp', 't_kWh', 'meter'],
                        parse_dates=['x_Timestamp'])
    df_mh = pd.read_csv(mathura_path, usecols=['x_Timestamp', 't_kWh', 'meter'],
                        parse_dates=['x_Timestamp'])
    df = pd.concat([df_br, df_mh], ignore_index=True)
    df = df.rename(columns={'x_Timestamp': 'timestamp', 't_kWh': 'kwh', 'meter': 'meter_id'})
    return df


def aggregate_to_15min(df):
    df = df.set_index('timestamp')
    agg = df.groupby('meter_id').resample('15min')['kwh'].sum().reset_index()
    return agg


def find_best_window(df, num_days=30):
    expected_slots = num_days * PERIODS_PER_DAY
    threshold = 0.90

    min_date = df['timestamp'].min().normalize()
    max_date = df['timestamp'].max().normalize()

    best_start = None
    best_count = 0

    candidate = min_date
    while candidate + pd.Timedelta(days=num_days) <= max_date:
        end = candidate + pd.Timedelta(days=num_days)
        subset = df[(df['timestamp'] >= candidate) & (df['timestamp'] < end)]
        counts = subset.groupby('meter_id')['kwh'].count()
        good = (counts >= threshold * expected_slots).sum()
        if good > best_count:
            best_count = good
            best_start = candidate
        candidate += pd.Timedelta(days=7)

    print(f"  Best window: {best_start.date()} to {(best_start + pd.Timedelta(days=num_days)).date()} "
          f"({best_count} meters with ≥90% coverage)")
    return best_start


def clean_and_filter(df, start_date, num_days):
    end_date = start_date + pd.Timedelta(days=num_days)
    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] < end_date)]

    expected_slots = num_days * PERIODS_PER_DAY

    # Filter meters with at least 90% coverage
    counts = df.groupby('meter_id')['kwh'].count()
    good_meters = counts[counts >= 0.90 * expected_slots].index
    df = df[df['meter_id'].isin(good_meters)]

    # Pivot to wide form
    full_index = pd.date_range(start_date, end_date, freq='15min', inclusive='left')
    pivot = df.pivot_table(index='timestamp', columns='meter_id', values='kwh', aggfunc='sum')
    pivot = pivot.reindex(full_index)

    # Forward/backward fill short gaps (up to 1 hour = 4 slots)
    pivot = pivot.ffill(limit=4).bfill(limit=4)

    # Drop meters that still have >2% NaN
    valid = pivot.columns[pivot.isna().sum() < 0.02 * len(pivot)]
    pivot = pivot[valid].fillna(0)

    # Clip negatives and cap outliers per meter
    pivot = pivot.clip(lower=0)
    caps = pivot.quantile(0.995)
    pivot = pivot.clip(upper=caps, axis=1)

    # Drop meters with near-zero median (dead meters)
    medians = pivot.median()
    pivot = pivot.loc[:, medians > 0.0001]

    print(f"  After cleaning: {pivot.shape[1]} meters, {pivot.shape[0]} time slots")
    return pivot


def assign_roles(pivot_df, prosumer_fraction=0.4, seed=42):
    rng = np.random.default_rng(seed)
    meters = list(pivot_df.columns)
    N = len(meters)
    num_prosumers = int(prosumer_fraction * N)
    num_consumers = N - num_prosumers

    total_consumption = pivot_df.sum().sort_values(ascending=False)
    # Top 70% consumers are candidates for PV
    candidate_pool = list(total_consumption.index[:int(0.7 * N)])
    prosumer_ids = list(rng.choice(candidate_pool, size=num_prosumers, replace=False))
    consumer_ids = [m for m in meters if m not in prosumer_ids]

    ordered_ids = consumer_ids + prosumer_ids
    print(f"  Roles: {num_consumers} consumers + {num_prosumers} prosumers")
    return ordered_ids, num_consumers, num_prosumers


def build_load_matrix(pivot_df, ordered_ids):
    # pivot values are kWh per 15-min; convert to kW: kW = kWh / 0.25
    L = pivot_df[ordered_ids].values.T * (1.0 / DT)
    return L


def generate_pv_for_real_loads(L, num_consumers, num_prosumers, num_days, seed=42):
    np.random.seed(seed)
    N, H = L.shape

    pv_shape = generate_daily_pv_shape()
    shape_energy_pv = np.sum(pv_shape * DT)
    PV = np.zeros((N, H))

    for idx in range(num_consumers, N):
        total_load = np.sum(L[idx, :] * DT)
        avg_daily = total_load / num_days
        sizing = np.random.uniform(0.4, 1.5)
        target_pv = sizing * avg_daily
        yield_kwp = np.random.uniform(4.5, 5.5)
        capacity = target_pv / yield_kwp
        base_pv = pv_shape * capacity * (yield_kwp / shape_energy_pv)
        for d in range(num_days):
            cloud = np.random.normal(1.0, 0.15)
            noise = np.random.normal(0.0, 0.05, size=PERIODS_PER_DAY)
            profile = base_pv * cloud * (1.0 + noise)
            profile = np.clip(profile, 0.0, None)
            PV[idx, d * PERIODS_PER_DAY:(d + 1) * PERIODS_PER_DAY] = profile

    # Cap PV and create exporters (same logic as run_scenarios.py)
    L_r = L.reshape(N, num_days, PERIODS_PER_DAY)
    PV_r = PV.reshape(N, num_days, PERIODS_PER_DAY)

    for i in range(N):
        for d in range(num_days):
            load_day = L_r[i, d].sum() * DT
            pv_day = PV_r[i, d].sum() * DT
            if load_day <= 0 or pv_day <= 0:
                continue
            if pv_day > 0.8 * load_day:
                PV_r[i, d] *= (0.8 * load_day) / pv_day

    # ~30% of prosumers as exporters
    num_exporters = max(2, int(0.3 * num_prosumers))
    exporter_indices = [num_consumers + i for i in range(num_exporters)]
    super_exporter = num_consumers + num_exporters if num_exporters < num_prosumers else None

    for i in exporter_indices:
        for d in range(num_days):
            load_day = L_r[i, d].sum() * DT
            pv_day = PV_r[i, d].sum() * DT
            if load_day <= 0 or pv_day <= 0:
                continue
            if pv_day < 1.1 * load_day:
                PV_r[i, d] *= (1.1 * load_day) / pv_day

    if super_exporter is not None:
        for d in range(num_days):
            load_day = L_r[super_exporter, d].sum() * DT
            pv_day = PV_r[super_exporter, d].sum() * DT
            if load_day <= 0 or pv_day <= 0:
                continue
            if pv_day < 1.5 * load_day:
                PV_r[super_exporter, d] *= (1.5 * load_day) / pv_day

    PV = PV_r.reshape(N, H)

    # Random inverter failures
    for _ in range(2):
        pidx = num_consumers + np.random.randint(0, num_prosumers)
        day = np.random.randint(0, num_days)
        PV[pidx, day * PERIODS_PER_DAY:(day + 1) * PERIODS_PER_DAY] = 0.0

    return PV


def assign_tariffs(N, num_consumers, seed=42):
    np.random.seed(seed)
    lambda_buy_ind = np.zeros(N)
    lambda_sell_ind = np.zeros(N)

    for i in range(N):
        if i < num_consumers:
            lambda_buy_ind[i] = np.random.uniform(6.0, 7.0)
            lambda_sell_ind[i] = 0.0
        else:
            lambda_buy_ind[i] = np.random.uniform(5.0, 6.0)
            lambda_sell_ind[i] = np.random.uniform(3.0, 3.8)

    lambda_buy_ref = np.mean(lambda_buy_ind)
    lambda_sell_ref = np.mean(lambda_sell_ind[lambda_sell_ind > 0])
    return lambda_buy_ind, lambda_sell_ind, lambda_buy_ref, lambda_sell_ref


def load_real_profiles(num_days=30, prosumer_fraction=0.4, seed=42):
    print("Loading smart meter data...")
    df = load_meter_data(BAREILLY_FILE, MATHURA_FILE)
    print(f"  Loaded {len(df)} rows, {df['meter_id'].nunique()} meters")

    print("Aggregating to 15-min resolution...")
    df = aggregate_to_15min(df)

    print("Finding best 30-day window...")
    start_date = find_best_window(df, num_days)

    print("Cleaning and filtering...")
    pivot = clean_and_filter(df, start_date, num_days)

    ordered_ids, num_consumers, num_prosumers = assign_roles(pivot, prosumer_fraction, seed)

    L = build_load_matrix(pivot, ordered_ids)
    N, H = L.shape

    print("Generating PV profiles...")
    PV = generate_pv_for_real_loads(L, num_consumers, num_prosumers, num_days, seed)

    lambda_buy_ind, lambda_sell_ind, lambda_buy_ref, lambda_sell_ref = assign_tariffs(N, num_consumers, seed)

    # P2P availability with random maintenance outages
    p2p_allowed = np.ones(H, dtype=bool)
    np.random.seed(seed + 100)
    for _ in range(3):
        start = np.random.randint(0, H - 8)
        p2p_allowed[start:start + 8] = False

    return {
        'L': L, 'PV': PV, 'N': N, 'H': H,
        'num_consumers': num_consumers, 'num_prosumers': num_prosumers,
        'lambda_buy_ind': lambda_buy_ind, 'lambda_sell_ind': lambda_sell_ind,
        'lambda_buy_ref': lambda_buy_ref, 'lambda_sell_ref': lambda_sell_ref,
        'p2p_allowed': p2p_allowed,
        'meter_ids': ordered_ids,
        'num_days': num_days,
        'start_date': start_date,
    }


def run_real_scenario(p):
    num_days = p['num_days']
    nc = p['num_consumers']

    bills_conv, _ = settle_conventional(
        p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'])

    bills_mmr, _, _, _ = settle_mmr(
        p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])

    bills_bs, _, _ = settle_billsharing(
        p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'],
        p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])

    bills_sdr, _, Pr_buy, Pr_sell, SDR_ts = settle_sdr(
        p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])

    bills_sdr_dsm, _, _, _, _, iters = settle_sdr_dsm(
        p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])

    def savings_pct(new, baseline):
        return ((baseline - new) / abs(baseline)) * 100 if baseline != 0 else 0.0

    def fairness(dc, dp):
        denom = abs(dc) + abs(dp)
        if denom < 1e-9:
            return 1.0
        return 1.0 - abs(dc - dp) / denom

    consumer_conv = bills_conv[:nc].sum()
    prosumer_conv = bills_conv[nc:].sum()
    total_conv = bills_conv.sum()

    mechanisms = {
        'MMR': bills_mmr,
        'BSM': bills_bs,
        'SDR': bills_sdr,
        'SDR-DSM': bills_sdr_dsm,
    }

    results = {
        'total_conv': total_conv,
        'avg_daily_load_kwh': np.sum(p['L'] * DT) / num_days,
        'avg_daily_pv_kwh': np.sum(p['PV'] * DT) / num_days,
        'iterations': iters,
    }

    for name, bills in mechanisms.items():
        key = name.lower().replace('-', '_')
        total = bills.sum()
        consumer = bills[:nc].sum()
        prosumer = bills[nc:].sum()
        results[f'community_savings_{key}'] = savings_pct(total, total_conv)
        results[f'consumer_delta_{key}'] = savings_pct(consumer, consumer_conv)
        results[f'prosumer_delta_{key}'] = savings_pct(prosumer, prosumer_conv)
        results[f'fairness_{key}'] = fairness(
            savings_pct(consumer, consumer_conv),
            savings_pct(prosumer, prosumer_conv))

    # Hourly SDR table
    hourly_sdr = np.zeros(24)
    hourly_buy = np.zeros(24)
    hourly_sell = np.zeros(24)

    for d in range(num_days):
        for hour in range(24):
            slots = range(d * PERIODS_PER_DAY + hour * 4, d * PERIODS_PER_DAY + (hour + 1) * 4)
            for s in slots:
                hourly_sdr[hour] += SDR_ts[s] if np.isfinite(SDR_ts[s]) else 0.0
                hourly_buy[hour] += Pr_buy[s]
                hourly_sell[hour] += Pr_sell[s]

    divisor = num_days * 4
    hourly_sdr /= divisor
    hourly_buy /= divisor
    hourly_sell /= divisor

    results['hourly'] = pd.DataFrame({
        'Hour': range(24),
        'SDR': hourly_sdr,
        'Buy Price (Rs/kWh)': hourly_buy,
        'Sell Price (Rs/kWh)': hourly_sell,
    })

    return results


def generate_report(p, results):
    nc = p['num_consumers']
    np_ = p['num_prosumers']
    N = p['N']
    num_days = p['num_days']

    md = []
    md.append("# P2P Pricing Framework: Real Indian Smart Meter Data Results\n")
    md.append("## Dataset Overview\n")
    md.append(f"- **Source**: CEEW Smart Meter Data (Bareilly + Mathura, 2020)")
    md.append(f"- **Households used**: {N} ({nc} consumers + {np_} prosumers)")
    md.append(f"- **Date window**: {p['start_date'].date()} to {(p['start_date'] + pd.Timedelta(days=num_days)).date()}")
    md.append(f"- **Resolution**: 15-minute intervals ({PERIODS_PER_DAY} slots/day)")
    md.append(f"- **Duration**: {num_days} days")
    md.append(f"- **Avg daily community load**: {results['avg_daily_load_kwh']:.1f} kWh")
    md.append(f"- **Avg daily community PV**: {results['avg_daily_pv_kwh']:.1f} kWh")
    md.append(f"- **PV sizing**: Synthetic clear-sky model, sized 40–150% of each prosumer's real load\n")

    md.append("\n## Mechanism Comparison\n")
    md.append("| Metric | BSM | MMR | SDR | SDR-DSM |")
    md.append("|:---|:---:|:---:|:---:|:---:|")
    md.append(f"| Community Savings (%) | {results['community_savings_bsm']:.2f} | {results['community_savings_mmr']:.2f} | {results['community_savings_sdr']:.2f} | **{results['community_savings_sdr_dsm']:.2f}** |")
    md.append(f"| Consumer Δ (%) | {results['consumer_delta_bsm']:.2f} | {results['consumer_delta_mmr']:.2f} | {results['consumer_delta_sdr']:.2f} | **{results['consumer_delta_sdr_dsm']:.2f}** |")
    md.append(f"| Prosumer Δ (%) | {results['prosumer_delta_bsm']:.2f} | {results['prosumer_delta_mmr']:.2f} | {results['prosumer_delta_sdr']:.2f} | **{results['prosumer_delta_sdr_dsm']:.2f}** |")
    md.append(f"| Fairness F | {results['fairness_bsm']:.3f} | {results['fairness_mmr']:.3f} | {results['fairness_sdr']:.3f} | **{results['fairness_sdr_dsm']:.3f}** |")
    md.append(f"| Convergence (iters) | — | — | — | {results['iterations']} |")

    md.append(f"\n\nBaseline conventional cost: ₹{results['total_conv']:.2f} over {num_days} days\n")

    md.append("\n## Hourly SDR and Price Dynamics (30-day average)\n")
    md.append("| Hour | SDR | Buy Price (₹/kWh) | Sell Price (₹/kWh) |")
    md.append("|:---:|:---:|:---:|:---:|")
    for _, row in results['hourly'].iterrows():
        md.append(f"| {int(row['Hour'])} | {row['SDR']:.3f} | {row['Buy Price (Rs/kWh)']:.4f} | {row['Sell Price (Rs/kWh)']:.4f} |")

    md.append("\n\n## Key Observations\n")
    md.append("1. Results are derived from **real Indian household consumption patterns** (not synthetic data).")
    md.append(f"2. Community of {N} homes from two North Indian cities provides realistic load heterogeneity.")
    md.append("3. PV profiles are synthetic (clear-sky + weather noise) but sized proportionally to each home's actual consumption.")
    md.append("4. Settlement mechanisms are identical to the toy-system analysis — only the input data changed.\n")

    return "\n".join(md)


if __name__ == "__main__":
    p = load_real_profiles(num_days=30, prosumer_fraction=0.4, seed=42)

    print(f"\nRunning settlements for {p['N']} households over {p['num_days']} days...")
    results = run_real_scenario(p)

    report = generate_report(p, results)
    output_path = Path(__file__).parent / "real_data_results.md"
    output_path.write_text(report)

    print(f"\nResults written to {output_path.name}")
    print(f"\nSummary ({p['num_consumers']}C + {p['num_prosumers']}P, N={p['N']}):")
    print(f"  Conventional baseline: ₹{results['total_conv']:.2f}")
    print(f"  SDR-DSM savings: {results['community_savings_sdr_dsm']:.2f}%")
    print(f"  SDR-DSM fairness: {results['fairness_sdr_dsm']:.3f}")
    print(f"  Convergence: {results['iterations']} iterations")
