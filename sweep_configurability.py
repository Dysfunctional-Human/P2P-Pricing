"""
Community size sweep using toysystem.py methodology.
Generates results for N=7, N=20, N=30 (skipping N=10 which is the main study).
All mechanisms: Conventional, BSM, MMR, SDR, SDR-DSM.
"""

import numpy as np
import pandas as pd

DAYS = 30
PERIODS_PER_DAY = 96
DT = 0.25


def generate_daily_load_shape():
    t = np.arange(PERIODS_PER_DAY)
    hours = t * DT
    base = 0.2
    morning_peak = np.exp(-0.5 * ((hours - 8) / 1.5) ** 2)
    evening_peak = 1.3 * np.exp(-0.5 * ((hours - 20) / 2.0) ** 2)
    shape = base + 0.7 * morning_peak + 1.0 * evening_peak
    shape = np.maximum(shape, 0.05)
    return shape


def generate_daily_pv_shape():
    t = np.arange(PERIODS_PER_DAY)
    hours = t * DT
    angle = np.pi * (hours - 6) / 12.0
    irradiance = np.sin(angle)
    irradiance[irradiance < 0] = 0.0
    return irradiance


def setup_community(num_consumers, num_prosumers, seed=42):
    """Generate profiles using toysystem.py methodology exactly."""
    np.random.seed(seed)
    N = num_consumers + num_prosumers
    H = DAYS * PERIODS_PER_DAY

    # Tariffs (same logic as toysystem.py)
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

    # Load profiles
    shape_day = generate_daily_load_shape()
    shape_energy = np.sum(shape_day * DT)
    L = np.zeros((N, H))

    for idx in range(N):
        if idx >= num_consumers:
            target = np.random.uniform(5.0, 9.0)
        else:
            target = np.random.uniform(4.0, 7.0)
        scale = target / shape_energy
        base_profile = shape_day * scale
        for d in range(DAYS):
            day_noise = np.random.normal(1.0, 0.05)
            noise = np.random.normal(0.0, 0.05, size=PERIODS_PER_DAY)
            profile = base_profile * day_noise * (1.0 + noise)
            profile = np.clip(profile, 0.05, None)
            L[idx, d*PERIODS_PER_DAY:(d+1)*PERIODS_PER_DAY] = profile

    # PV profiles (same as toysystem.py)
    pv_shape = generate_daily_pv_shape()
    shape_energy_pv = np.sum(pv_shape * DT)
    PV = np.zeros((N, H))

    for idx in range(num_consumers, N):
        total_load = np.sum(L[idx, :] * DT)
        avg_daily = total_load / DAYS
        sizing = np.random.uniform(0.4, 1.5)
        target_pv = sizing * avg_daily
        yield_kwp = np.random.uniform(4.5, 5.5)
        capacity = target_pv / yield_kwp
        base_pv = pv_shape * capacity * (yield_kwp / shape_energy_pv)
        for d in range(DAYS):
            cloud = np.random.normal(1.0, 0.15)
            noise = np.random.normal(0.0, 0.05, size=PERIODS_PER_DAY)
            profile = base_pv * cloud * (1.0 + noise)
            profile = np.clip(profile, 0.0, None)
            PV[idx, d*PERIODS_PER_DAY:(d+1)*PERIODS_PER_DAY] = profile

    # Cap PV at 80% of load (toysystem.py STEP 0)
    L_r = L.reshape(N, DAYS, PERIODS_PER_DAY)
    PV_r = PV.reshape(N, DAYS, PERIODS_PER_DAY)

    for i in range(N):
        for d in range(DAYS):
            load_day = L_r[i, d].sum() * DT
            pv_day = PV_r[i, d].sum() * DT
            if load_day <= 0 or pv_day <= 0:
                continue
            if pv_day > 0.8 * load_day:
                PV_r[i, d] *= (0.8 * load_day) / pv_day

    # Exporters: first 2 prosumers at 1.1x, third at 1.5x (toysystem.py STEP 1)
    exporters = [num_consumers + 0, num_consumers + 1]
    super_exporter = num_consumers + 2 if num_prosumers > 2 else None

    for i in exporters:
        if i >= N:
            break
        for d in range(DAYS):
            load_day = L_r[i, d].sum() * DT
            pv_day = PV_r[i, d].sum() * DT
            if load_day <= 0 or pv_day <= 0:
                continue
            if pv_day < 1.1 * load_day:
                PV_r[i, d] *= (1.1 * load_day) / pv_day

    if super_exporter is not None and super_exporter < N:
        for d in range(DAYS):
            load_day = L_r[super_exporter, d].sum() * DT
            pv_day = PV_r[super_exporter, d].sum() * DT
            if load_day <= 0 or pv_day <= 0:
                continue
            if pv_day < 1.5 * load_day:
                PV_r[super_exporter, d] *= (1.5 * load_day) / pv_day

    PV = PV_r.reshape(N, H)

    # Random events
    for _ in range(2):
        pidx = num_consumers + np.random.randint(0, num_prosumers)
        day = np.random.randint(0, DAYS)
        PV[pidx, day*PERIODS_PER_DAY:(day+1)*PERIODS_PER_DAY] = 0.0

    p2p_allowed = np.ones(H, dtype=bool)
    for _ in range(3):
        start = np.random.randint(0, H - 8)
        p2p_allowed[start:start+8] = False

    return {
        'L': L, 'PV': PV, 'N': N, 'H': H,
        'num_consumers': num_consumers, 'num_prosumers': num_prosumers,
        'lambda_buy_ind': lambda_buy_ind, 'lambda_sell_ind': lambda_sell_ind,
        'lambda_buy_ref': lambda_buy_ref, 'lambda_sell_ref': lambda_sell_ref,
        'p2p_allowed': p2p_allowed
    }


# Settlement functions (identical to toysystem.py)

def settle_conventional(L, PV, dt, lbuy_ind, lsell_ind):
    N, H = L.shape
    M = np.minimum(L, PV)
    Pim = np.maximum(L - M, 0.0)
    Pex = np.maximum(PV - M, 0.0)
    cost_ts = np.zeros((N, H))
    for i in range(N):
        cost_ts[i] = Pim[i] * dt * lbuy_ind[i] - Pex[i] * dt * lsell_ind[i]
    return np.sum(cost_ts, axis=1), cost_ts


def compute_cp2p_dynamic(L_total, PV_total, lbuy, lsell):
    H = len(L_total)
    cp2p = np.zeros(H)
    mid = 0.5 * (lbuy + lsell)
    spread = 0.5 * (lbuy - lsell)
    for h in range(H):
        denom = L_total[h] + PV_total[h]
        if denom < 1e-9:
            cp2p[h] = mid
        else:
            net = (L_total[h] - PV_total[h]) / denom
            cp2p[h] = mid + spread * net
    return cp2p


def settle_mmr(L, PV, dt, lbuy, lsell, p2p_allowed):
    N, H = L.shape
    L_total = np.sum(L, axis=0)
    PV_total = np.sum(PV, axis=0)
    cp2p = compute_cp2p_dynamic(L_total, PV_total, lbuy, lsell)

    Pim = np.maximum(L - np.minimum(L, PV), 0.0)
    Pex = np.maximum(PV - np.minimum(L, PV), 0.0)

    c_im = np.zeros(H)
    c_ex = np.zeros(H)
    for h in range(H):
        if not p2p_allowed[h]:
            c_im[h], c_ex[h] = lbuy, lsell
            continue
        if abs(PV_total[h] - L_total[h]) <= 1e-6:
            c_im[h] = c_ex[h] = cp2p[h]
        elif PV_total[h] > L_total[h]:
            c_im[h] = cp2p[h]
            c_ex[h] = (L_total[h] * cp2p[h] + (PV_total[h] - L_total[h]) * lsell) / PV_total[h] if PV_total[h] > 0 else lsell
        else:
            c_ex[h] = cp2p[h]
            c_im[h] = (PV_total[h] * cp2p[h] + (L_total[h] - PV_total[h]) * lbuy) / L_total[h] if L_total[h] > 0 else lbuy

    cost_ts = np.zeros((N, H))
    for i in range(N):
        cost_ts[i] = Pim[i] * dt * c_im - Pex[i] * dt * c_ex
    return np.sum(cost_ts, axis=1), cost_ts


def settle_billsharing(L, PV, dt, lbuy_ind, lsell_ind, lbuy, lsell, p2p_allowed):
    N, H = L.shape
    L_total = np.sum(L, axis=0)
    PV_total = np.sum(PV, axis=0)
    cp2p = compute_cp2p_dynamic(L_total, PV_total, lbuy, lsell)

    Pim = np.maximum(L - np.minimum(L, PV), 0.0)
    Pex = np.maximum(PV - np.minimum(L, PV), 0.0)

    cost_ts = np.zeros((N, H))
    for h in range(H):
        import_tot = np.sum(Pim[:, h])
        export_tot = np.sum(Pex[:, h])
        if import_tot <= 0 and export_tot <= 0:
            continue
        if not p2p_allowed[h]:
            for i in range(N):
                cost_ts[i, h] = Pim[i, h] * dt * lbuy_ind[i] - Pex[i, h] * dt * lsell_ind[i]
            continue
        p2p_energy = min(import_tot, export_tot)
        p2p_im_frac = p2p_energy / import_tot if import_tot > 0 else 0.0
        p2p_ex_frac = p2p_energy / export_tot if export_tot > 0 else 0.0
        for i in range(N):
            e_im = Pim[i, h] * dt
            e_ex = Pex[i, h] * dt
            e_im_p2p = Pim[i, h] * p2p_im_frac * dt
            e_ex_p2p = Pex[i, h] * p2p_ex_frac * dt
            cost_ts[i, h] = (e_im_p2p * cp2p[h] - e_ex_p2p * cp2p[h] +
                             (e_im - e_im_p2p) * lbuy_ind[i] - (e_ex - e_ex_p2p) * lsell_ind[i])
    return np.sum(cost_ts, axis=1), cost_ts


def settle_sdr(L, PV, dt, lbuy, lsell, p2p_allowed):
    N, H = L.shape
    NP = L - PV
    Pim = np.zeros_like(L)
    Pex = np.zeros_like(L)
    Pp2p_buy = np.zeros_like(L)
    Pp2p_sell = np.zeros_like(L)
    SDR = np.zeros(H)
    Pr_buy = np.zeros(H)
    Pr_sell = np.zeros(H)

    for h in range(H):
        if not p2p_allowed[h]:
            for i in range(N):
                if NP[i, h] > 0:
                    Pim[i, h] = NP[i, h]
                elif NP[i, h] < 0:
                    Pex[i, h] = -NP[i, h]
            SDR[h] = np.nan
            Pr_buy[h], Pr_sell[h] = lbuy, lsell
            continue

        buyers = NP[:, h] > 0
        sellers = NP[:, h] < 0
        TBP = NP[buyers, h].sum() if np.any(buyers) else 0.0
        TSP = (-NP[sellers, h]).sum() if np.any(sellers) else 0.0

        if TBP <= 1e-9:
            SDR[h] = np.inf
        else:
            SDR[h] = TSP / TBP

        if not np.isfinite(SDR[h]) or SDR[h] >= 1.0:
            Pr_sell[h] = Pr_buy[h] = lsell
        else:
            Pr_sell[h] = (lbuy * lsell) / ((lbuy - lsell) * SDR[h] + lsell)
            Pr_buy[h] = Pr_sell[h] * SDR[h] + lbuy * (1.0 - SDR[h])

        internal = min(TSP, TBP)
        if internal > 0:
            if TSP > 0 and np.any(sellers):
                Pp2p_sell[sellers, h] = internal * (-NP[sellers, h]) / TSP
            if TBP > 0 and np.any(buyers):
                Pp2p_buy[buyers, h] = internal * NP[buyers, h] / TBP

        for i in range(N):
            if NP[i, h] > 0:
                Pim[i, h] = max(NP[i, h] - Pp2p_buy[i, h], 0.0)
            elif NP[i, h] < 0:
                Pex[i, h] = max(-NP[i, h] - Pp2p_sell[i, h], 0.0)

    cost_ts = np.zeros((N, H))
    for i in range(N):
        for h in range(H):
            cost_ts[i, h] = (Pim[i, h] * dt * lbuy - Pex[i, h] * dt * lsell +
                             Pp2p_buy[i, h] * dt * Pr_buy[h] - Pp2p_sell[i, h] * dt * Pr_sell[h])

    return np.sum(cost_ts, axis=1), cost_ts, Pr_buy, Pr_sell, SDR


def settle_sdr_dsm(L_ref, PV, dt, lbuy, lsell, p2p_allowed,
                   alpha=0.12, max_iter=8, damping=0.3):
    """SDR with DSM iterative algorithm."""
    N, H = L_ref.shape
    L_adj = L_ref.copy()

    for iteration in range(max_iter):
        L_prev = L_adj.copy()
        _, _, Pr_buy, Pr_sell, _ = settle_sdr(L_adj, PV, dt, lbuy, lsell, p2p_allowed)

        for i in range(N):
            prices = np.zeros(H)
            NP_i = L_adj[i] - PV[i]
            for h in range(H):
                if NP_i[h] > 0:
                    prices[h] = Pr_buy[h]
                else:
                    prices[h] = -Pr_sell[h]

            adjustment = -(prices * dt) / (2 * alpha)
            new_load = L_ref[i] + adjustment

            for d in range(DAYS):
                s = d * PERIODS_PER_DAY
                e = (d + 1) * PERIODS_PER_DAY
                day_ref_energy = np.sum(L_ref[i, s:e]) * dt
                new_load[s:e] = np.clip(new_load[s:e], 0.5 * L_ref[i, s:e], 1.5 * L_ref[i, s:e])
                day_new_energy = np.sum(new_load[s:e]) * dt
                if day_new_energy > 0:
                    new_load[s:e] *= day_ref_energy / day_new_energy

            L_adj[i] = (1 - damping) * L_prev[i] + damping * new_load

    bills, cost_ts, Pr_buy, Pr_sell, SDR_ts = settle_sdr(L_adj, PV, dt, lbuy, lsell, p2p_allowed)
    return bills, cost_ts


def run_sweep(num_consumers, num_prosumers, seed):
    """Run all mechanisms and return metrics."""
    p = setup_community(num_consumers, num_prosumers, seed)
    nc = num_consumers

    bills_conv, _ = settle_conventional(p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'])
    bills_mmr, _ = settle_mmr(p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])
    bills_bs, _ = settle_billsharing(p['L'], p['PV'], DT, p['lambda_buy_ind'], p['lambda_sell_ind'],
                                     p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])
    bills_sdr, _, _, _, _ = settle_sdr(p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])
    bills_dsm, _ = settle_sdr_dsm(p['L'], p['PV'], DT, p['lambda_buy_ref'], p['lambda_sell_ref'], p['p2p_allowed'])

    def pct(new, base):
        return ((base - new) / abs(base)) * 100 if base != 0 else 0.0

    total_conv = bills_conv.sum()

    results = {}
    for name, bills in [('BSM', bills_bs), ('MMR', bills_mmr), ('SDR', bills_sdr), ('SDR-DSM', bills_dsm)]:
        consumer_conv = bills_conv[:nc].sum()
        prosumer_conv = bills_conv[nc:].sum()
        results[name] = {
            'community': pct(bills.sum(), total_conv),
            'consumer': pct(bills[:nc].sum(), consumer_conv),
            'prosumer': pct(bills[nc:].sum(), prosumer_conv),
        }

    return results


if __name__ == "__main__":
    # Use different seeds to avoid any overlap with main study (seed=42)
    scenarios = [
        (2, 5, 77),    # N=7
        (7, 13, 123),  # N=20
        (10, 20, 99),  # N=30
    ]

    print("Community Size Sweep Results")
    print("=" * 80)
    print(f"{'Config':<15} {'Mechanism':<10} {'Community %':>12} {'Consumer %':>12} {'Prosumer %':>12}")
    print("-" * 80)

    for nc, np_, seed in scenarios:
        N = nc + np_
        results = run_sweep(nc, np_, seed)
        for mech, vals in results.items():
            sign_p = '+' if vals['prosumer'] >= 0 else ''
            cfg = f"{nc}C+{np_}P (N={N})"
            print(f"{cfg:<15} {mech:<10} {vals['community']:>11.2f}% {vals['consumer']:>11.2f}% {sign_p}{vals['prosumer']:>10.2f}%")
        print()
