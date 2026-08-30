# Dashboard Guide

`dashboard/app_enhanced.py` is the primary demo surface for this project — an 8-tab
Streamlit app that walks through the whole system without needing the real trained
model or dataset (it runs on illustrative/synthetic data, seeded for repeatability).
`dashboard/app.py` is an earlier, simpler 5-tab version kept for reference.

## Launch

```bash
./run_enhanced_dashboard.sh
# or manually:
cd dashboard && streamlit run app_enhanced.py --server.port 8501
```

Open `http://localhost:8501`.

## Tabs

| Tab | Purpose | Key message |
|---|---|---|
| Overview | Headline metrics | Entry point / key numbers |
| How It Works | Animated 6-step FL round, network topology | Data never leaves client devices — only model updates are shared |
| Privacy Guarantees | DP noise before/after, gradient-inversion attack simulation, ε slider | Privacy and accuracy aren't a trade-off here — both hold simultaneously |
| Graph Intelligence | Heterogeneous graph (users/cards/merchants/devices), impossible-velocity map | Graph methods catch coordinated fraud rings that flat ML misses |
| Contrastive Learning | t-SNE embedding space, dual-head decision logic | The contrastive head catches *novel* fraud patterns supervised learning hasn't seen |
| Training Metrics | Convergence curves, per-client comparison | Standard FL training diagnostics |
| Live Inference | Interactive transaction form → fraud probability + anomaly score + feature importance | Explainable, real-time-shaped decisioning (illustrative rules — see note below) |
| Comparison | Rule-based vs centralized ML vs plain FL vs this system | Where this approach sits relative to the alternatives |
| **Test Your Data** | Upload a CSV of transactions, or simulate a live feed, run through the **real trained checkpoint** | This is the one tab where the numbers are real model output, not illustrative math |

## Suggested demo flow (~15 min)

1. **Overview** (2 min) — set the stage.
2. **How It Works** (3 min) — explain the FL round.
3. **Privacy Guarantees** (3 min) — DP/attack-resistance demo.
4. **Graph Intelligence** (2 min) — fraud ring detection.
5. **Live Inference** (3 min) — interactive, audience picks values (see
   [TESTING_GUIDE.md](TESTING_GUIDE.md) for realistic inputs).
6. **Comparison** (2 min) — where this sits vs alternatives.
7. **Test Your Data** (2 min) — close by proving it's real: upload a CSV or
   run the live-feed simulation against the actual trained model.

For a technical audience, lead with *How It Works → Privacy Guarantees → Graph
Intelligence → Contrastive Learning*. For a business audience, lead with *Overview →
Comparison → Live Inference*.

## A note on the numbers shown

Most figures in this dashboard (accuracy %, ROI, business-impact dollar amounts, the
"+29% on card sharing rings" style deltas) are **illustrative placeholders**, not
numbers pulled from a real evaluation run — the dashboard generates them synthetically
(`np.random` with a fixed seed) so it renders without needing the trained model or
dataset present. `experiments/generate_matrices.py` is the clearest example: it
hard-codes "99.85% accuracy" style rows straight into
`results/production_matrices_summary.csv` — nothing there was ever measured.

**The real, measured numbers** live in `results/paysim/TRAINING_RESULTS.md` (76.6%
accuracy, 95.3% ROC-AUC, on the real PaySim data, before the graph-branch/DP fixes)
and `results/synthetic_paysim/` / `results/synthetic_paysim_dp/` (after the fixes, on
the rule-based synthetic stand-in dataset — see [ARCHITECTURE.md](ARCHITECTURE.md)).
Before using specific numbers in a report or presentation, pull from one of those, or
caveat clearly which tab's numbers are illustrative.

**The Test Your Data tab is the exception** — it loads the actual checkpoint
(`results/synthetic_paysim/global_model.pt` by default) and runs real forward passes
on whatever you upload. It's the place to point someone who asks "but does this
actually work, or is it just charts?"

## Troubleshooting

- **Port in use:** `./run_enhanced_dashboard.sh` binds `:8501`; stop whatever else is
  using it or pass `--server.port <other>`.
- **Missing deps:** `pip install -r requirements.txt` from the repo root (the pinned
  venv at `venv_fl/` already has everything needed).
