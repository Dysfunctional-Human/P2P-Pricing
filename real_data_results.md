# P2P Pricing Framework: Real Indian Smart Meter Data Results

## Dataset Overview

- **Source**: CEEW Smart Meter Data (Bareilly + Mathura, 2020)
- **Households used**: 56 (34 consumers + 22 prosumers)
- **Date window**: 2020-02-05 to 2020-03-06
- **Resolution**: 15-minute intervals (96 slots/day)
- **Duration**: 30 days
- **Avg daily community load**: 232.2 kWh
- **Avg daily community PV**: 100.4 kWh
- **PV sizing**: Synthetic clear-sky model, sized 40–150% of each prosumer's real load


## Mechanism Comparison

| Metric | BSM | MMR | SDR | SDR-DSM |
|:---|:---:|:---:|:---:|:---:|
| Community Savings (%) | 13.67 | 14.51 | 11.95 | **13.85** |
| Consumer Δ (%) | 8.22 | 13.29 | 16.15 | **12.76** |
| Prosumer Δ (%) | 35.42 | 19.37 | -4.81 | **18.23** |
| Fairness F | 0.377 | 0.814 | 0.000 | **0.823** |
| Convergence (iters) | — | — | — | 8 |


Baseline conventional cost: ₹28578.58 over 30 days


## Hourly SDR and Price Dynamics (30-day average)

| Hour | SDR | Buy Price (₹/kWh) | Sell Price (₹/kWh) |
|:---:|:---:|:---:|:---:|
| 0 | 0.000 | 6.0498 | 6.0498 |
| 1 | 0.000 | 6.0498 | 6.0498 |
| 2 | 0.000 | 6.0498 | 6.0498 |
| 3 | 0.000 | 6.0498 | 6.0498 |
| 4 | 0.000 | 6.0498 | 6.0498 |
| 5 | 0.000 | 6.0498 | 6.0498 |
| 6 | 0.031 | 6.0373 | 5.9184 |
| 7 | 0.221 | 5.7992 | 5.2078 |
| 8 | 0.432 | 5.3329 | 4.5207 |
| 9 | 0.643 | 4.8442 | 4.1840 |
| 10 | 0.998 | 4.0045 | 3.7087 |
| 11 | 1.373 | 3.6105 | 3.5129 |
| 12 | 1.477 | 3.6302 | 3.5226 |
| 13 | 1.451 | 3.5477 | 3.4840 |
| 14 | 1.436 | 3.6052 | 3.5008 |
| 15 | 1.079 | 3.9716 | 3.6394 |
| 16 | 0.538 | 5.0545 | 4.3580 |
| 17 | 0.088 | 5.9862 | 5.6928 |
| 18 | 0.000 | 6.0498 | 6.0498 |
| 19 | 0.000 | 6.0498 | 6.0498 |
| 20 | 0.000 | 6.0498 | 6.0498 |
| 21 | 0.000 | 6.0498 | 5.9845 |
| 22 | 0.000 | 6.0498 | 5.9627 |
| 23 | 0.000 | 6.0498 | 6.0280 |


## Key Observations

1. Results are derived from **real Indian household consumption patterns** (not synthetic data).
2. Community of 56 homes from two North Indian cities provides realistic load heterogeneity.
3. PV profiles are synthetic (clear-sky + weather noise) but sized proportionally to each home's actual consumption.
4. Settlement mechanisms are identical to the toy-system analysis — only the input data changed.
