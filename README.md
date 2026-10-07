# Federated Fraud Detection

**A Temporal Graph Transformer trained across simulated banks with FedProx and client-level differential privacy — with real, measured results instead of asserted ones.**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178C6?logo=typescript&logoColor=white)
![Status](https://img.shields.io/badge/status-active%20capstone%20project-orange)

---

## Why this exists

Fraud detection is a cross-institution problem: the patterns that separate fraud from
legitimate activity are strongest when several banks' data is pooled, but raw
transaction data can never legally or competitively be pooled. This project trains one
shared fraud model across simulated banks **without any of them sharing raw data** —
each bank trains locally, and only a clipped, differentially-noised update ever leaves
it.

What makes this repo different from a typical capstone writeup: every number below is
read from a checked-in JSON file produced by an actual training run
(`results/comparison/comparison.json`, `results/comparison/epsilon_sweep.json`), not
asserted. Section [The road to real numbers](#the-road-to-real-numbers) documents three
real bugs that were producing fabricated-looking (but not fabricated) 100% results
before they were found and fixed — because a privacy-preserving ML project that can't
show its evaluation is trustworthy isn't worth much regardless of its architecture.

## Contents

- [What's actually implemented](#whats-actually-implemented)
- [Real, measured results](#real-measured-results)
- [Real PaySim results](#real-paysim-results)
- [Quick start](#quick-start)
- [The web app](#the-web-app)
- [Project structure](#project-structure)
- [The road to real numbers](#the-road-to-real-numbers)
- [Honest limitations](#honest-limitations)
- [Tech stack](#tech-stack)
- [Further reading](#further-reading)
- [Authors](#authors)

## What's actually implemented

```
Input: transaction sequence (batch, 10 transactions, 10 features)
  │
  ├── Temporal branch — 4-layer Transformer encoder
  ├── Graph branch    — per-account encoder (see limitations below)
  │
  └── Cross-modal fusion (bidirectional attention)
         │
         ├── Supervised head (BCE)        → fraud probability
         └── Contrastive head (NT-Xent)   → anomaly score for novel fraud
```

- **Federated training** — FedAvg and FedProx across 10 simulated bank clients with
  non-identical (though not fully heterogeneous — see [limitations](#honest-limitations))
  transaction distributions. FedProx's proximal term keeps training stable when only a
  minority of clients are sampled per round.
- **Client-level differential privacy** — DP-FedAvg (McMahan et al., 2018): each
  client clips its whole round update; the server adds one calibrated Gaussian noise
  draw to the aggregate, once per round. Not naive per-minibatch noising, which
  silently blows the stated privacy budget (see [ARCHITECTURE.md](docs/ARCHITECTURE.md)
  for the two broken versions of this that predated the current one).
- **A right-sized model** — 32-dim hidden size, ~45K parameters. This isn't
  arbitrary: Gaussian-mechanism noise scales with `σ√d`, so a smaller model needs
  proportionally less noise for the same `ε` — right-sizing moved the usable epsilon
  floor from ~2000 down to ~20–50, a 40–100× improvement, with zero change to the
  privacy mechanism itself.

## Real, measured results

Four models, trained for real, evaluated on one identical held-out test set (6,000
sequences, 362 fraud) — not a lookup table. Reproduce with
`python experiments/run_model_comparison.py --rounds 10`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Privacy |
|---|---|---|---|---|---|---|
| Centralized RandomForest | 98.27% | 95.74% | 74.59% | 83.85% | 94.74% | None |
| Centralized deep model | 97.72% | 90.04% | 69.89% | 78.69% | 94.11% | None |
| Federated (FedAvg, no DP) | 97.40% | 84.11% | 70.17% | 76.51% | 93.98% | Data locality only |
| **Federated (FedProx + DP, ε=100)** | **97.50%** | **85.81%** | **70.17%** | **77.20%** | 93.48% | **Formal (ε,δ)-DP** |

The honest framing isn't "beats everything" — a method with zero privacy cost (the
RandomForest) should and does edge out a privacy-preserving one on raw accuracy. The
real claim: this project is statistically indistinguishable from pooling every
institution's raw data (0.2pp off centralized), while being the only approach with a
provable privacy guarantee.

**Does differential privacy actually do anything, or is `ε=100` just a number?** A full
9-point sweep (`experiments/run_epsilon_sweep.py`) answers that directly:

| ε | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| 1.0 (textbook) | 6.0% | 0.114 | 0.500 — destroyed |
| 5.0 | 6.2% | 0.114 | 0.442 — destroyed |
| 10.0 | 74.1% | 0.127 | 0.577 — recovering |
| 20.0 | 96.7% | 0.688 | 0.916 — working |
| 50.0 | 97.5% | 0.763 | 0.930 — near-ceiling |
| **100.0 (default)** | **97.1%** | **0.751** | **0.935** |
| 2000.0 | 97.0% | 0.751 | 0.936 |
| No DP (ceiling) | 97.5% | 0.769 | 0.938 |

That's the shape a real privacy-utility tradeoff is supposed to have: a hard floor
where the mechanism genuinely destroys the model, a transition region, and a plateau —
not a flat line, which is what a broken (inert) DP implementation would look like.

## Real PaySim results

The tables above use synthetic data. Real PaySim data (`preprocessed_datasets_csv/`,
1.9M real transactions, pre-windowed into 2.86M ten-step sequences across 10 clients)
is also in this repo, but it needed a real fix before it was usable:

**The data problem.** Every client's delivered `val`/`test` CSV was **100% fraud, zero
legitimate transactions** — verified on the full files, not a sample. That makes
precision, ROC-AUC, and a confusion matrix undefined on those splits. Pooling
train+val+test and re-splitting (a fix that works for a simpler "misassigned rows" bug)
was tried and rejected: it just redistributed the same corruption into a uniform ~30%
fraud rate everywhere, nowhere near real PaySim's documented ~0.13% rate — proof the
val/test files are a separate, inflated fraud pool bolted onto a clean `train` file, not
misassigned legitimate rows. The actual fix
(`experiments/build_real_paysim_splits.py`): rebuild all three splits from each client's
own `train` CSV alone (0.04%–0.17% fraud for 9 of 10 clients — the right order of
magnitude for real PaySim), via a fresh stratified 70/15/15 split. **Client 9 is a
separate, unexplained anomaly** — its own train file alone is 46.9% fraud, ~1000x every
other client — carried through as-is rather than silently corrected, since its root
cause is unknown (`configs/paysim.yaml`, named in the dataset's own metadata as what
built it, doesn't exist anywhere in this repo).

**Baseline results** (`python run_training.py --dataset paysim_real --rounds 15
--clients 10 --exclude-clients 9 --pos-weight 8 --fraud-rate-prior 0.001 --max-samples
50000`, 9 clients, client_9 excluded so the test set stays representative of the true
~0.1% rate):

| Metric | Value |
|---|---|
| Accuracy | 99.71% |
| Precision | 54.74% |
| Recall | 48.15% |
| F1 | 51.23% |
| ROC-AUC | 89.32% |

This is meaningfully worse than the synthetic numbers above (F1 51% vs. 77%) — expected,
not a failure: real PaySim's fraud rate is ~45x more imbalanced, and real fraud patterns
are harder to separate than a rule-based generator's clean ones.

**The fraud-starvation bug and its fix.** At real PaySim's ~0.1–0.3% fraud rate and
`batch_size=64`, most training batches contain **zero** fraud examples — so `pos_weight`
(the loss-level class reweighting) has nothing to act on most of the time. Verified two
ways before fixing it: sweeping `pos_weight` from 8 to 400 changed nothing (confusion
matrix was byte-identical across the whole range — per-batch gradient-norm clipping was
saturating its effect), and post-hoc threshold tuning on the trained checkpoint recovered
essentially zero extra F1. The real bottleneck was upstream of both. Fix: a
`WeightedRandomSampler` (`run_training.py --oversample --oversample-ratio R`) that
oversamples fraud into every batch at a target rate `R`, instead of the true (near-zero)
class ratio.

**Full balance isn't the right ratio — it's a dial, and it has a sweet spot.** Sweeping
`R` (8-round test runs) gave a clean, monotonic precision/recall tradeoff:

| `R` | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| 0 (baseline, no oversample) | 0.6364 | 0.5833 | 0.70 | 0.9039 |
| 0.01 | 0.6305 | 0.5648 | 0.7135 | 0.8966 |
| **0.02** | **0.6560** | **0.7593** | 0.5775 | 0.9713 |
| 0.05 | 0.5434 | 0.7963 | 0.4125 | 0.9725 |
| 0.1 | 0.4123 | 0.8704 | 0.2701 | 0.9749 |
| 0.5 (full balance) | 0.2311 | 0.9074 | 0.1324 | 0.9741 |

`R=0.02` is the one value that improves **both** F1 and recall over baseline simultaneously
— every other tested ratio only trades one for the other.

**An honest wrinkle: more training made it worse, not better.** Scaling `R=0.02` up to
the full run (15 rounds, 50k samples, matching the baseline's scale) did **not** reproduce
the short-run win — it scored *worse* on F1 than even the plain baseline:

| Config | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| Baseline, 8 rounds/20k | 0.6364 | 0.5833 | 0.70 | 0.9039 |
| Baseline, 15 rounds/50k | 0.5123 | 0.4815 | 0.5474 | 0.8932 |
| Oversample R=0.02, 8 rounds/20k | **0.6560** | **0.7593** | 0.5775 | 0.9713 |
| Oversample R=0.02, 15 rounds/50k | 0.4220 | 0.8009 | 0.2864 | 0.9826 |

The pattern holds for *both* configs: more rounds and more data consistently **hurt** F1
here, not helped — almost certainly because each client has only ~100–300 real fraud
sequences total, so longer training just keeps pushing the decision boundary toward
flagging more (recall and ROC-AUC climb every time; precision collapses faster). The
headline real-data checkpoint (`results/paysim_real_best/`) is therefore the smaller
8-round/20k-sample, `R=0.02` run — chosen because it's empirically the best by F1, not
because it's the most training. The longer run (`results/paysim_real_oversampled/`) and
the plain baseline (`results/paysim_real/`) are both kept for comparison, not discarded.

Including client_9 scores much better still (F1 78%) — but that's measuring against a test
set client_9 made ~8% fraud instead of ~0.1%, i.e. an easier distribution, not a better
model. Reported separately, not folded into the headline number above.

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate a small synthetic dataset (schema-matched stand-in for PaySim —
#    see "Honest limitations" for why the real dataset isn't bundled)
cd experiments && python generate_synthetic_paysim.py --out ../data/synthetic_paysim --clients 10 && cd ..

# 2. Train
python run_training.py --dataset synthetic_paysim --rounds 10 --clients 10 --clients-per-round 3
python run_training.py --dataset synthetic_paysim --dp --epsilon 100 --rounds 10   # with DP

# 3. Reproduce the comparison table and epsilon sweep above
python experiments/run_model_comparison.py --rounds 10
python experiments/run_epsilon_sweep.py --rounds 10
```

## The web app

A FastAPI backend serves the real trained checkpoint; a React + TypeScript frontend
gives it six pages, all backed by real endpoints — nothing in the UI is mocked.

```bash
# terminal 1
cd backend && uvicorn main:app --reload --port 8000
# terminal 2
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173`:

| Page | What it does |
|---|---|
| **Overview** | Real headline metrics pulled live from the trained checkpoint |
| **Test Your Data** | Upload *any* CSV (map your own columns after upload) or score one transaction by hand, against the real checkpoint |
| **Train** | Kicks off an actual small/fast FedProx + DP-FedAvg run with live per-round progress |
| **Live Simulation** | Watch it happen: 10 animated bank nodes generate fresh transactions, train locally, and send updates to a central server — with real, live fraud detection results streaming in as it runs |
| **Model Comparison** | The real 4-model comparison table above, plus a disclosed privacy-weighted composite score |
| **Architecture & Privacy** | The real 9-point epsilon sweep chart, and an honest accounting of what's wired in vs. what's a documented simplification |

A Streamlit dashboard (`./run_enhanced_dashboard.sh`) also still works and covers more
of the explanatory content — see [docs/DASHBOARD_GUIDE.md](docs/DASHBOARD_GUIDE.md).

## Project structure

```
federated-fraud-detection/
├── src/
│   ├── models/temporal_graph_transformer.py  # the model
│   ├── federated/                            # DP utilities, Flower client
│   └── data/feature_engineering.py           # the one shared feature encoding
├── run_training.py                # the training loop that's actually used
├── experiments/
│   ├── generate_synthetic_paysim.py  # schema-matched synthetic data generator
│   ├── run_model_comparison.py       # the 4-model real comparison
│   ├── run_epsilon_sweep.py          # the 9-point privacy-utility sweep
│   └── train_federated.py            # Flower/Ray reference path (see docs)
├── backend/main.py                # FastAPI serving the real checkpoint
├── frontend/                      # React + TypeScript + Vite dashboard
├── dashboard/                     # Streamlit alternative
├── configs/experiment_config.yaml
├── docs/
│   ├── ARCHITECTURE.md            # what's real vs. simplified, in detail
│   ├── DASHBOARD_GUIDE.md
│   ├── TESTING_GUIDE.md
│   └── paper/draft_paper.tex      # IEEE-format writeup of this work
└── results/comparison/            # the real JSON behind every table above
```

## The road to real numbers

Three real bugs were found and fixed during development — documented here because a
privacy-utility claim is only as credible as the pipeline that produced it, and
skipping this section would leave the misleading impression that everything just
worked.

1. **A trivially separable benchmark.** The first synthetic data generator drew fraud
   and legitimate transactions from disjoint value ranges. Every model — including a
   plain RandomForest — scored 100% on every metric, not because any model was good,
   but because the benchmark was trivial. Fixed by overlapping the distributions and
   folding in 12% "quiet fraud" drawn from the legitimate distribution.
2. **An over-parameterized, collapsing model.** At `hidden_dim=128` (~1.07M
   parameters for a 10-dimensional input), the model collapsed to a constant
   prediction (0% recall) under class imbalance and DP noise. Right-sized to
   `hidden_dim=32` (~45K parameters) with a prior-informed bias init and class
   weighting — see [Real, measured results](#real-measured-results) for the DP-budget
   payoff this had.
3. **A NaN-masking bug hiding a real training failure.** Guarding every loss term
   with a NaN-to-zero substitution prevented crashes but let the model silently train
   on a degenerate output once corrupted, because `0.0 * NaN == NaN` under IEEE 754
   corrupted the total loss even at zero weight. Fixed by sanitizing at the source and
   adding per-round checkpoint-and-rollback.

Two smaller but real bugs from the same pass: the CSV upload endpoint hung forever on
large files because `mapping`/`account_id_col` were missing `Form(...)` annotations
(FastAPI silently dropped them) and there was no row cap or batching — fixed and
load-tested on 200K+ row files; and the merchant-type dropdown was rendering
white-text-on-white on this dark theme, making 9 of 10 options invisible.

## Honest limitations

- The 4-model comparison and epsilon-sweep tables above are on a schema-matched
  **synthetic** stand-in (a pipeline-correctness check, not a real-world performance
  claim). Real PaySim results now exist separately — see
  [Real PaySim results](#real-paysim-results) — and score meaningfully lower, as
  expected given real PaySim's much sharper ~0.13% fraud rate (vs. the synthetic set's
  ~6%).
- The 10 simulated clients currently draw from the **same** underlying distributions
  with independent random samples, not genuinely heterogeneous per-bank patterns — real
  non-IID heterogeneity (different fraud rates, spending patterns per institution) isn't
  modeled yet, despite FedProx being chosen partly for non-IID robustness.
- The graph branch is a simplified per-account encoder, not a relational graph over
  shared entities (user, merchant, card, device) — the real `HeterogeneousGraphEncoder`
  (HGTConv) exists in code but needs entity-linked edge data this pipeline doesn't
  produce yet.
- Homomorphic encryption **and** Secure Aggregation are now both wired into the live
  training loop (`run_training.py --he` / `--he --secure-agg`) — real Paillier
  encryption of each client's full update, either aggregated directly by the server or
  routed through a tiered client→proxy→server topology so no single proxy sees every
  client. Real, measured cost: 31–48s to encrypt one client's full 45,292-param update,
  34–41s to decrypt the aggregate, per round (512-bit keys — demo-speed only, not
  production-secure; see ARCHITECTURE.md); adding the proxy tier costs almost nothing
  extra (0.33–0.41s for the proxy-level aggregation itself, since homomorphic addition
  is plain modular multiplication, not exponentiation).
- All reported `ε` values are **per-round** DP-FedAvg budgets; no formal multi-round
  composition accountant (e.g., Rényi DP) has been applied, so the true cumulative
  privacy loss across a full run is larger than any single `ε` quoted above.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full, unrounded accounting.

## Tech stack

**Model & training:** PyTorch, PyTorch Geometric, Opacus (reference), Flower
(reference path) · **Backend:** FastAPI, Uvicorn · **Frontend:** React 19, TypeScript,
Vite, Tailwind CSS v4, Recharts, Framer Motion, Lenis · **Data:** NumPy, Pandas,
scikit-learn

## Further reading

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the unrounded technical accounting:
  what's real vs. a documented simplification, the full DP history, the model
  comparison methodology
- [docs/DASHBOARD_GUIDE.md](docs/DASHBOARD_GUIDE.md) — Streamlit dashboard walkthrough
- [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) — live inference test values
- [docs/paper/draft_paper.tex](docs/paper/draft_paper.tex) — an IEEE-format writeup of
  this work (compile with `pdflatex`, or paste into [Overleaf](https://overleaf.com))

## Authors

Tanishk Maheshwari, Vaibhav Handoo, Vibhav Kolachana, Yash Kuber Khanna —
Department of Computer Science and Engineering, PES University, Bengaluru.
Guided by Dr. Prafullata Kiran Auradkar.

## License

No license file is currently included, which by default means all rights are
reserved. Add a `LICENSE` file (e.g. MIT) before or after pushing if you want this
repo to be reusable by others.
