# Fraud Detection Testing Guide

Realistic input values for the **Live Inference** tab of the dashboard, and the logic
the (illustrative) scoring is built around.

## Legitimate transaction examples

| Scenario | Amount | Merchant | Hour | Distance | Time since last | Card age | Txns/24h | Fraud prob | Anomaly |
|---|---|---|---|---|---|---|---|---|---|
| Grocery | $45.50 | Grocery | 14 | 2 km | 180 min | 730 days | 2 | ~5% | ~12% |
| Morning coffee | $5.50 | Restaurant | 7 | 1 km | 720 min | 912 days | 1 | ~3% | ~8% |
| Gas fill-up | $65.00 | Gas Station | 8 | 5 km | 240 min | 1095 days | 1 | ~4% | ~10% |
| Evening dinner | $120.00 | Restaurant | 19 | 3 km | 120 min | 365 days | 3 | ~8% | ~15% |
| Weekend shopping | $89.99 | Electronics | 16 | 8 km | 90 min | 456 days | 2 | ~7% | ~14% |

## Fraudulent transaction examples

| Scenario | Amount | Merchant | Hour | Distance | Time since last | Card age | Txns/24h | Fraud prob | Anomaly | Why |
|---|---|---|---|---|---|---|---|---|---|---|
| Impossible velocity | $850 | Electronics | 16 | 450 km | 25 min | 365 days | 3 | ~95% | ~88% | 1,080 km/h implied speed |
| High-value night | $8,500 | Online | 2 | 5 km | 120 min | 45 days | 1 | ~92% | ~85% | large amount, 2 AM, new card |
| Card testing | $1.00 | Online | 3 | 800 km | 5 min | 10 days | 25 | ~78% | ~91% | $1 probe, high frequency, new card |
| Travel fraud | $3,200 | Travel | 23 | 1200 km | 15 min | 3 days | 2 | ~88% | ~79% | impossible velocity, brand-new card |
| Multiple small online | $12.50 | Online | 1 | 650 km | 8 min | 15 days | 18 | ~82% | ~94% | high frequency, new card, distance |
| Jewelry/electronics | $5,800 | Electronics | 3 | 380 km | 35 min | 5 days | 4 | ~94% | ~87% | high amount + velocity + new card |

## Risk factor thresholds

- **Amount:** low <$200, medium $200–1k, high $1k–5k, very high >$5k
- **Time of day:** normal 6 AM–11 PM, suspicious 11 PM–2 AM, high-risk 2 AM–6 AM
- **Velocity (implied km/h):** normal <100, suspicious 100–300, impossible >300
- **Frequency:** normal 1–5/day, suspicious 6–10/day, high-risk >10/day
- **Card age:** very new (high risk) <30 days, new (medium) 30–90 days, established (low) >90 days
- **Merchant risk:** low = grocery/gas/pharmacy; medium = restaurant/ATM/retail; high = online/travel/jewelry/electronics

## Decision logic

```
if fraud_prob > 80% OR anomaly_score > 75%:
    DECLINE
elif fraud_prob > 50% OR anomaly_score > 50%:
    REVIEW (manual)
else:
    APPROVE
```

`fraud_prob` comes from the supervised head (trained on labeled fraud — good at known
patterns). `anomaly_score` comes from the contrastive head (trained on normal
behavior only — good at *novel* patterns it's never explicitly seen). The system
combines both because they catch different things.

## To deliberately trigger each outcome

- **Legitimate:** amount <$200, hours 7 AM–10 PM, distance <20 km, gap >60 min, card
  age >90 days, <5 txns/day, low-risk merchant.
- **Fraud:** extreme amount, hours 2–6 AM, implied velocity >300 km/h, gap <10 min,
  card age <30 days, >10 txns/day, high-risk merchant.
- **Edge case:** amount $800–1,200, hours 10 PM–midnight, distance 50–100 km, gap
  30–60 min, card age 30–90 days, 5–8 txns/day.
