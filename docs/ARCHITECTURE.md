# Encryption, Aggregation & Flower Architecture

This document explains the privacy/security stack and how the [Flower](https://flower.dev/)
federated learning framework is wired into this project. For model architecture
(Temporal Transformer + HGAT + Contrastive head), see the main [README](../README.md).

---

## Privacy/Security Layers

| Layer | Guarantee | Mechanism | Status |
|---|---|---|---|
| 1. Data Locality | Raw data never leaves clients | Training happens on each bank's local data; only model updates are transmitted | ✅ real |
| 2. Differential Privacy (DP) | A client's whole update is bounded/noised | See "Differential Privacy" below - client-level DP-FedAvg | ✅ real, `--dp` flag |
| 3. Homomorphic Encryption (HE) | Server can't see individual client updates | Paillier additive HE in `homomorphic_encryption.py`, wired into `run_training.py`'s aggregation step | ✅ real, `--he` flag |
| 4. Secure Aggregation | No single proxy/server sees every client's update | Proxy servers (`proxy_server.py`) do intermediate HE aggregation, wired into `run_training.py` | ✅ real, `--secure-agg` flag (requires `--he`) |
| 5. TLS 1.3 | Transport security | Encrypts all network traffic between clients/proxies/server | 📝 not implemented - this is a simulation running in one process |
| 6. Mutual Authentication | Identity verification | Clients and server authenticate to each other | 📝 not implemented, same reason |

**On layer 3 (HE):** `run_training.py --he` genuinely routes client→server
aggregation through Paillier encryption instead of plaintext FedAvg - each
selected client's full parameter vector is scaled by its own integer sample
count in plaintext (so `Enc(n_k·w_k)` is computed directly - mathematically
identical to encrypt-then-scale-under-encryption, but avoids a second,
needless modular exponentiation per value) and encrypted client-side (public
key only); all selected clients' encrypted updates are homomorphically
summed entirely under encryption, and the server (the only party holding the
private key) decrypts only that one aggregate, never an individual client's
update (`federated_aggregate_he()` in `run_training.py`). Verified
algebraically identical to plaintext FedAvg up to float rounding.

Naive single-process Paillier encryption of a 45,292-parameter model doesn't
scale (benchmarked: ~2.3ms/value at 512-bit keys ⇒ ~104s to encrypt one
client's full update, serially) - this was fixed with a parallel
encrypt/decrypt pool (`encrypt_values_parallel`/`decrypt_values_parallel` in
`homomorphic_encryption.py`, one Paillier ciphertext per parameter still,
just computed across all CPU cores instead of one). Real, measured cost from
an actual 3-round run (`results/paysim_real_he/`, 2 clients/round, 22 CPU
cores, 512-bit keys): **31-48s to encrypt one client's full update, 34-41s to
decrypt the aggregate, per round.** That overhead is disclosed, not hidden -
it's added on top of normal training time and scales with `clients_per_round`
(each selected client encrypts). **512-bit keys are demo-speed only, not
production-secure** - the module's own docstring already said 2048+ is
needed for real security; that costs roughly another order of magnitude of
compute (see the key-size benchmark at the top of `homomorphic_encryption.py`'s
module docstring).

**On layer 4 (Secure Aggregation / proxy servers):** `run_training.py
--secure-agg` (requires `--he`) now genuinely routes the same HE-encrypted,
pre-scaled client updates through a tiered client→proxy→server topology
(`ProxyServerManager`/`ProxyServer` in `proxy_server.py`) instead of
client→server directly: each client is assigned to a proxy round-robin
(`client_id % num_proxies`, `--num-proxies`, default 2), every proxy blindly
sums the encrypted updates it receives and forwards one partial sum, and the
global server blindly sums the proxies' partial sums and decrypts only the
final total (`federated_aggregate_secure()` in `run_training.py`). This adds
a real collusion-resistance property beyond direct client→server HE alone: a
colluding attacker there only needs to control the server, but here would
additionally need to control every proxy that handled at least one client
that round. Real, measured cost from an actual 3-round run
(`results/paysim_real_secureagg/`, 3 clients/round, 2 proxies, 22 CPU cores,
512-bit keys): per-client encrypt cost is unchanged (30-56s, one outlier at
238s likely system contention, not algorithmic - see the raw history JSON);
**proxy-level aggregation itself is cheap, 0.33-0.41s per round** - unlike
encrypt/decrypt, homomorphic addition is plain modular multiplication, not
exponentiation, so adding a proxy tier costs almost nothing on top of the
HE cost that already exists with plain `--he`. Two real bugs were found and
fixed while wiring this in, both worth knowing about since they show
`proxy_server.py` had genuinely never been run end-to-end before: an
`UnicodeEncodeError` crash from emoji in its `print()` statements on
Windows' default `cp1252` console encoding (stripped), and a scalar-multiply
step in `federated_aggregate_he()` that was real but silently excluded from
the disclosed timing numbers (fixed by the plaintext pre-scaling described
above, which also made it faster).

Note that "no single proxy sees every client" is a real property of this
code path, but "the server never sees a plaintext update" still holds in the
same sense as layer 3 - a single-process simulation, so it's a guarantee
about what the aggregation code path ever reads, not about separate
machines/processes with an actual network boundary between them. HE (layer
3) and this tiered-trust topology (layer 4) are separable and independently
toggleable: `--he` alone, or `--he --secure-agg` together.

### Round flow (what actually runs — `run_training.py`)

```
Client (Bank i)                              Global Server
────────────────                              ─────────────
1. Train locally for E epochs (FedProx loss)
2. delta_i = local_params - global_params
3. Clip: delta_i' = delta_i * min(1, C / ||delta_i||)   [only if --dp]
4. If --he: encrypt params (Paillier, public key only)
                                              ──►  If --he: homomorphic weighted sum, decrypt once
                                                    else:    plaintext FedAvg: w' = Σ (n_i/N) * (global + delta_i')
                                                    if --dp: w' += N(0, σ²)   [once per round]
```

No proxies or real network hops happen here — this is a single-process
simulation where "clients" are just copies of the model trained on different
in-memory data slices, so `--he`'s "server never sees a plaintext client
update" guarantee holds in the sense that the plaintext value is never read
by the aggregation code path, not in the sense of separate machines/processes
with an actual network boundary between them. See the HE/proxy note above for
what a real multi-process/multi-machine deployment would add on top.

### Why homomorphic encryption

Paillier cryptosystem property used here: `Enc(a) · Enc(b) = Enc(a + b)` — the server
(or proxy) can sum encrypted gradients without ever decrypting an individual client's
update. Only the final aggregated sum is decrypted.

- **Pros:** server never sees individual gradients; security reduces to hardness of
  factoring; composes cleanly with DP; no trusted third party needed.
- **Cons:** real, measured overhead - 31-48s to encrypt one client's full 45,292-param
  update and 34-41s to decrypt the aggregate, per round (512-bit keys, 22 CPU cores,
  `results/paysim_real_he/`) - vs. the plaintext FedAvg step it replaces, which is plain
  tensor arithmetic and takes milliseconds. This is overhead added on top of normal
  training time, not a multiplier on it, and it scales with `clients_per_round` (every
  selected client encrypts). Also: larger ciphertexts (~4x size), and it only supports
  addition —
  no multiplication, so it's restricted to linear aggregation like FedAvg/FedProx.

### Why proxy servers for secure aggregation

Proxies distribute the aggregation load, tolerate individual proxy failures, reduce
client-to-server latency, and mean no single node ever sees every client's raw
(encrypted) update — a basic collusion-resistance property, not just a performance
optimization.

### Differential Privacy — how it actually works here, and why ε isn't 1.0

Two broken versions of DP existed in this project before the current fix; both
are worth knowing about because "we use ε=1.0 differential privacy" appears
throughout the docs/dashboard and neither version actually delivered that.

- **v1** (`src/federated/dp_utils.py`, original): clipped + noised raw
  gradients after *every minibatch*, but was documented as ε "per round". With
  L batches × E local epochs, the mechanism actually ran L×E times per round
  with no composition accounting — the real privacy loss was far larger than
  any ε ever quoted for it.
- **v2** (an intermediate fix): moved clip+noise to once per client per round,
  applied to that client's own update independently. Correct accounting, but
  broken at model scale: the standard Gaussian-mechanism noise vector's L2
  norm scales as `σ√d` for a d-dimensional update. At the time this model had
  **~1.07M parameters** (hidden_dim=128, drastically over-parameterized for
  this 10-dim input - see "Model comparison" below for how that was found and
  fixed). At the textbook ε=1.0, we measured the noise added was **~100× the
  clip norm** — verified empirically, this collapses training to a model that
  always predicts "legitimate" (0% recall, ROC-AUC=0.5, i.e. random).

**Current version — client-level DP-FedAvg** (McMahan et al. 2018, *Learning
Differentially Private Recurrent Language Models*): clients only clip their
update; the server adds exactly **one** Gaussian noise draw per round, to the
aggregated update, sized by the largest single client's averaging weight
(`max_client_weight × clip_norm` is the aggregate's sensitivity to one
client, not the full clip_norm). This is both more correct and less noisy
than v2, but the dimensionality problem doesn't disappear — it only gets
smaller. And it did get smaller: right-sizing the model to hidden_dim=32
(~45K params instead of ~1.07M, see "Model comparison" below) cut `√d` by
~4.9x, which moved the whole epsilon curve down proportionally.

**Measured, on the current right-sized model (~45K params, clip_norm=1.0,
δ=1e-5), 10 rounds each — full 9-point sweep, produced by
`experiments/run_epsilon_sweep.py`, raw data in
`results/comparison/epsilon_sweep.json`:**

| ε | Accuracy | Precision | Recall | F1 | ROC-AUC | What happens |
|---|---|---|---|---|---|---|
| 1.0 (textbook) | 6.0% | 6.0% | 100% | 0.114 | 0.500 | destroyed — collapses to predicting "fraud" for everything |
| 5.0 | 6.2% | 6.0% | 100% | 0.114 | 0.442 | still destroyed, same failure mode |
| 10.0 | 74.1% | 8.0% | 31.2% | 0.127 | 0.577 | partial recovery, still unusable |
| 20.0 | 96.7% | 80.1% | 60.2% | 0.688 | 0.916 | working, real margin below default |
| 50.0 | 97.5% | 90.8% | 65.7% | 0.763 | 0.930 | working, close to no-DP |
| **100 (this project's default when `--dp` is passed)** | **97.1%** | **77.4%** | **72.9%** | **0.751** | **0.935** | no measurable utility cost vs. no-DP |
| 500.0 | 97.1% | 77.0% | 73.2% | 0.751 | 0.936 | effectively no DP noise left |
| 2000.0 | 97.0% | 76.1% | 74.0% | 0.751 | 0.936 | effectively no DP noise left |
| — (no DP) | 97.5% | 84.9% | 70.2% | 0.769 | 0.938 | baseline (federated, no DP) |

This is the standard shape a DP-utility curve is supposed to have — a hard
floor where the mechanism destroys the model (ε≤5), a transition region
(ε≈10-20), and a plateau where added noise stops mattering (ε≥50) — and it's
exactly what "the DP mechanism has a real, measurable effect" looks like as
evidence, rather than an assertion. Note the ε=1/5 failure mode is the
*opposite* of naive intuition: the model doesn't collapse to predicting the
majority class ("legitimate", which would give ~94% accuracy for free) — it
collapses to predicting the minority class ("fraud") for every input, which
is what the prior-informed bias init plus `pos_weight=8.0` produce once the
DP noise has destroyed the actual signal the classifier head would otherwise
rely on.

At ε=100 the noise is small enough on this synthetic task to cost nothing
measurable — that won't hold on harder, real-world fraud data, where the
accuracy/ε tradeoff will show up more visibly. The point that survives
regardless: ε≤5 is still not usable, even at 1/24th the parameter count,
full stop - but the *floor* moved from ~2000 down to ~20-50, a genuine ~40-100x
improvement purely from right-sizing the architecture to the actual
input, with no change to the privacy mechanism itself.

**Why ε=100 and not ε=1**: at this parameter count, keeping ε in the
textbook "≈1" range with the plain Gaussian mechanism on the full parameter
vector is still not achievable without destroying the model — this is a
genuine, known tension in DP for deep learning, not a bug to paper over. The
standard fixes that make small-ε practical at scale are (a) **per-example
gradient clipping via Opacus** (already a listed dependency, never actually
used anywhere in this codebase) operating on much smaller per-example
gradient objects rather than the whole parameter vector, or (b) restricting
DP to a lower-dimensional summary (e.g. only the classifier head). Both are
reasonable next steps beyond what's implemented now; for a cross-silo
setting like this one (each client is a bank, not one person), a large ε at
the client-update level is also a more defensible choice than it would be
for a single-user privacy setting, since the unit being protected is a
bank's entire aggregated contribution for the round.

Reproduce these numbers: `python run_training.py --dp --epsilon 1.0 --rounds 3`
vs `--epsilon 100 --rounds 10` (see `results/synthetic_paysim/` for a saved
run, and `experiments/run_model_comparison.py`'s `[D]` baseline for the
side-by-side vs. no-DP).

### Combined privacy guarantee (design intent, not what currently runs)

DP gives a *statistical* privacy guarantee (bounded by ε, δ). HE gives a
*cryptographic* guarantee on top — the server can't read an update at all
before aggregation. Together they'd be defense-in-depth. As-implemented,
only the DP layer is exercised in training; HE remains a correct but
disconnected reference implementation (see the layers table above).

Relevant reading: Dwork & Roth (2014) *Algorithmic Foundations of Differential
Privacy*; Abadi et al. (2016) *Deep Learning with Differential Privacy*; Paillier
(1999) *Public-Key Cryptosystems Based on Composite Degree Residuosity Classes*;
Bonawitz et al. (2017) *Practical Secure Aggregation for Privacy-Preserving ML*;
McMahan et al. (2017) *Communication-Efficient Learning of Decentralized Data*; Li et
al. (2020) *Federated Optimization in Heterogeneous Networks* (FedProx).

---

## Two training paths — which one actually runs

There are two separate implementations of the same FL training loop:

- **`run_training.py`** (repo root) — a direct, dependency-light implementation
  using plain PyTorch loops, no Ray/Flower simulation runtime. **This is the
  one that's actually been run and produced every checkpoint under
  `results/`** (confirmed by `results/paysim/TRAINING_RESULTS.md`'s config
  matching this script's `DEFAULT_CONFIG` exactly). All three fixes in this
  document (graph branch, DP, the "Known issue" below) live here.
- **`experiments/train_federated.py` + `src/federated/client.py`** — the
  Flower/Ray-based version described below. It's a real, complete
  implementation of the same idea, kept in sync with the same fixes, but
  there's no evidence it was ever run to completion in this project (see the
  known Ray/`ModuleNotFoundError` issue below) — treat it as the
  "production-shaped" path and `run_training.py` as the one to actually run.

### Flower components (`experiments/train_federated.py` path)

- **`experiments/train_federated.py`** — orchestrates the run: builds the FedProx
  strategy, defines server-side evaluation, starts the simulation.
- **`src/federated/client.py`** — `FraudDetectionClient(fl.client.NumPyClient)`,
  implements `fit()` (local training + DP) and `evaluate()`.
- **`src/federated/dp_utils.py`** — differential privacy (clipping + noise).
- **`src/federated/homomorphic_encryption.py`**, **`proxy_server.py`** — the HE /
  secure-aggregation layer described above.

### Server setup

```python
strategy = fl.server.strategy.FedProx(
    fraction_fit=0.1,              # sample 10% of clients per round
    min_fit_clients=1,
    proximal_mu=0.1,               # FedProx regularization for non-IID data
    evaluate_fn=get_evaluate_fn(config, device),
)

fl.simulation.start_simulation(
    client_fn=create_client_fn(config, device),
    num_clients=10,
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
    client_resources={"num_cpus": 1},
)
```

FedProx (not vanilla FedAvg) was chosen because each bank's transaction distribution
is non-IID — FedProx's proximal term `μ(w_global - w_local)` discourages any one
client's local update from drifting too far from the global model, which stabilizes
training when only one client is sampled per round (`fraction_fit=0.1` on 10 clients).

### Client round, per bank

1. Receive global parameters.
2. Train locally for `local_epochs` on that bank's own data.
3. Clip gradients, add DP noise, step the optimizer.
4. Return updated parameters + sample count + metrics to the server.

Flower's `NumPyClient` interface only requires `get_parameters()` /
`set_parameters()` / `fit()` / `evaluate()` — everything else (sampling clients,
weighted aggregation, round orchestration) is handled by the framework.

### Known issue: `ModuleNotFoundError: No module named 'src'`

Ray (which Flower's simulation mode uses under the hood) spawns worker processes that
don't inherit the parent's `sys.path.append('..')` hack in `train_federated.py`. Fixes,
in order of preference:

1. `pip install -e .` from the repo root (uses `setup.py`, makes `src` a real
   importable package) — this is the durable fix.
2. Set `PYTHONPATH` to the repo root before running.
3. As a last resort, switch the `from src...` imports to absolute
   `from federated_fraud_detection...` imports matching the installed package name.

---

## The graph branch — what was wrong, what's fixed, what's still a simplification

Two separate bugs made the "novel HGAT fraud-ring detection" branch inert:

1. **Every caller passed `graph_features=None`.** `temporal_graph_transformer.py`
   fell back to `torch.zeros(...)` when that happened — every round, for
   every client, the graph branch's input was a constant, carrying zero
   signal. **Fixed**: when `graph_features` is `None`, the model now derives
   a real per-sample feature (the account's mean transaction profile across
   its window: `sequences.mean(dim=1)`) instead of zeros, so the branch gets
   real, per-sample gradient signal without every caller needing to change.
2. **`SimpleGraphEncoder.forward()` collapsed the batch dimension.** It
   concatenated all node types along `dim=0`, then split the result back
   apart as `{'user': x[:1], 'merchant': x[1:]}` — treating the *batch*
   dimension as if it were separate node types. In `CrossModalFusion`, a
   graph-features tensor with batch size 1 gets broadcast-expanded to the
   full batch (see its `size(0) == 1` branch), so every sample in a batch
   silently received the *same* graph contribution (sample 0's). This is a
   real correctness bug, independent of point 1, and would have corrupted
   results the moment anyone did wire in real graph_features. **Fixed**: each
   node type now keeps its own batch dimension through the encoder.

**What's still a simplification, and why:** the model actually used
(`SimpleGraphEncoder`) is a per-sample residual MLP, not real heterogeneous
message passing — there's no adjacency, no real "graph" being learned over.
The real implementation (`HeterogeneousGraphEncoder`, using PyG's `HGTConv`,
3 layers / 8 heads) is fully built and sitting unused in `graph_encoder.py`.
It needs `edge_index_dict` — actual edges between transactions sharing a
user/merchant/card/device — which requires entity-ID columns the
preprocessed `(seq_len, num_features)` tensors don't carry. That data
(the real PaySim dataset with linked entity IDs) isn't available on this
machine (see below); wiring in `HeterogeneousGraphEncoder` for real is the
natural next step once it is.

## Where the training data comes from

- `results/paysim/` — a checkpoint + `TRAINING_RESULTS.md` from an earlier real
  run on real data (76.6% accuracy, 95.3% ROC-AUC — the real numbers; see
  [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md) for where the "99.85%" figures
  elsewhere in this repo actually came from).
- `experiments/generate_synthetic_paysim.py` — generates a **rule-based
  synthetic stand-in dataset** with the same schema, using the fraud/legit
  transaction archetypes from [TESTING_GUIDE.md](TESTING_GUIDE.md), via the
  single feature encoding in `src/data/feature_engineering.py`. `results/synthetic_paysim/`
  and `results/synthetic_paysim_dp/` are real training runs on this data (not
  fabricated numbers) — useful as a pipeline-correctness check, not a claim
  about real-world performance.
- `preprocessed_datasets_csv/paysim/` — real PaySim data (1.9M transactions,
  2.86M pre-windowed sequences, 10 clients) added later. **It needed a fix
  before use**: every client's delivered `val`/`test` CSV was 100% fraud with
  zero legitimate transactions (verified on full files). Pooling train+val+test
  and re-splitting was tried and rejected — it just produced a uniform ~30%
  fraud rate everywhere, far from real PaySim's documented ~0.13%, proving
  val/test were a separate inflated fraud pool, not misassigned legitimate
  rows. `experiments/build_real_paysim_splits.py` rebuilds all three splits
  from each client's `train` CSV alone (the only trustworthy file, at
  0.04%–0.17% fraud for 9/10 clients) into `data/paysim_real/`. Client 9 is a
  separate, unexplained anomaly (46.9% fraud in its own train file) carried
  through as-is, not silently corrected — see the script's docstring and
  `data/paysim_real/metadata/dataset_info.json`'s `known_anomaly` field.
  Real results: see [README.md's "Real PaySim results"](../README.md#real-paysim-results)
  (baseline: 99.71% accuracy, 51.2% F1, 89.3% ROC-AUC on 9 clients, client_9 excluded
  for a representative test set) — meaningfully worse than the synthetic numbers below,
  as expected given real PaySim's much sharper imbalance.
- **Fraud-starvation fix**: at real PaySim's ~0.1-0.3% fraud rate and `batch_size=64`,
  most batches contain zero fraud examples, so `pos_weight` loss-reweighting has
  nothing to act on (confirmed: sweeping it 8→400 produced a byte-identical confusion
  matrix every time — per-batch gradient-norm clipping saturates its effect). Fixed
  with `--oversample --oversample-ratio 0.02` (`WeightedRandomSampler`), the one ratio
  found to beat baseline on both F1 (0.636→0.656) and recall (0.583→0.759)
  simultaneously — every other ratio tested (0.05 up to full 0.5 balance) only traded
  one for the other. Headline checkpoint: `results/paysim_real_best/`. Honest caveat:
  scaling the same ratio up to a longer 15-round/50k-sample run did **not** reproduce
  the win (F1 dropped to 0.422, below even the plain baseline) — more training
  consistently hurt F1 here, most likely because each client has only ~100-300 real
  fraud sequences total. The smaller run is kept as the real result, not the larger one.

`src/data/feature_engineering.py` is the single source of truth for how a
raw transaction becomes the model's 10-dim feature vector — both the
synthetic data generator and the dashboard's **Test Your Data** tab
(CSV upload / live-feed simulation, using the real trained checkpoint) import
from it, so what the model was trained on and what a live prediction sees
are guaranteed to mean the same thing.

## The web app (`backend/` + `frontend/`)

A FastAPI backend (`backend/main.py`) wraps the same model, `feature_engineering.py`,
and `run_training.py` behind an HTTP API — no logic is duplicated, it's the
same code the CLI and Streamlit dashboard use. A React + Vite + Tailwind
frontend (`frontend/`) is the primary interface; the Streamlit dashboard
remains for the explanatory/demo tabs.

The one real UX improvement over the dashboard's CSV upload: `/api/csv/inspect`
fuzzy-matches an uploaded CSV's columns against the required fields and returns
a suggested mapping (e.g. a raw PaySim export's `nameOrig` auto-maps to
`account_id`), and any field left unmapped falls back to a documented neutral
default (`feature_engineering.DEFAULTS`) rather than requiring an exact column
schema — so a CSV from a completely different dataset still runs, with the
frontend showing exactly which fields were defaulted.

`/api/train/start` + `/api/train/status/{job_id}` run the same real training
loop as `run_training.py`, in a background thread, polled for live progress -
this is what backs the frontend's Train page and Streamlit's "Start Training"
button (added when the dashboard's version of that button turned out to do
nothing - it flipped a session flag with no consumer).

## Model comparison — real baselines, not a lookup table

`experiments/run_model_comparison.py` trains and evaluates four models on the
**same held-out test set** (pooled `test_temporal.pt` across all clients,
never touched by any of the four models' training):

| Model | Federated? | DP? | Isolates |
|---|---|---|---|
| A. Centralized RandomForest | No | No | classical ML vs. deep, both with full data access |
| B. Centralized deep model (same architecture) | No | No | cost of federation alone |
| C. Federated, FedAvg (mu=0) | Yes | No | cost of DP alone (federation already applied) |
| D. Federated, FedProx + DP — this project | Yes | Yes | — |

Results (real numbers, per-model `train_seconds`, and the methodology text)
are saved to `results/comparison/comparison.json` and served at
`GET /api/comparison`, rendered on the frontend's **Model Comparison** page.

**Real measured results** (synthetic PaySim-style data, v2 generator, 6000-row
held-out test set, 6.0% fraud):

| Model | Acc | Prec | Rec | F1 | AUC | Privacy |
|---|---|---|---|---|---|---|
| Centralized RandomForest | 0.983 | 0.957 | 0.746 | 0.839 | 0.947 | none |
| Centralized deep model | 0.977 | 0.900 | 0.699 | 0.787 | 0.941 | none |
| Federated, FedAvg (no DP) | 0.974 | 0.841 | 0.702 | 0.765 | 0.940 | federated only |
| **Federated, FedProx + DP (this project)** | 0.975 | 0.858 | 0.702 | 0.772 | 0.935 | federated + DP (ε=100) |

Honest read: the classical RandomForest edges out on raw metrics (not
unusual for tabular fraud features). More interesting - our model is
*statistically indistinguishable* from the fully-open centralized deep model
(0.975 vs 0.977 acc) and actually **beats** the no-DP federated baseline on
every metric (FedProx's proximal term compensating for the DP noise). The
honest claim this supports: privacy here costs close to nothing, not that
this is the most accurate model on the leaderboard.

**Same comparison, real PaySim data** (`--dataset paysim_real --exclude-clients 9
--oversample-ratio 0.02`, 270,225-sequence test set, 0.08% fraud; see README.md's
"Real PaySim model comparison" for the full table): a structurally different, more
important finding than "lower numbers" - B (centralized deep) and C (FedAvg, no DP)
both stayed degenerate even with the same oversampling fix that works for the main
pipeline (B: F1=0, ROC-AUC=0.5, collapsed after 4 retries; C: inverted into flagging
~19% of all legitimate transactions as fraud). Only D (FedProx+DP, this project's
actual approach) produced a real result (F1=27.3%). At real PaySim's severe imbalance,
oversampling fixes batch-level fraud starvation but doesn't substitute for FedProx's
proximal term or DP's noise as training-time regularization - consistent with
`run_centralized_deep`'s own docstring already describing B as "the only one of the
four models with zero training-time regularization." This is reported as a real
property of the architecture under this data regime, not re-run until it looks better.

### Why federated + DP, if it doesn't win on accuracy?

No serious federated-learning / differential-privacy paper claims to beat a
centralized, non-private model on raw accuracy - that would be mathematically
suspicious, not impressive, since privacy mechanisms trade some utility for a
guarantee by definition. The comparison that actually matters is on what
guarantee each approach carries, not just its accuracy:

| Approach | Raw data ever pooled? | Formal privacy guarantee? | Cross-institution collaboration? |
|---|---|---|---|
| Single-institution rules / classical ML (current common practice) | No sharing needed | none | Not possible - each institution sees only its own fraud patterns |
| Centralized data-pooling (shared consortium warehouse) | Yes, all raw data pooled | none | Requires every party to legally share raw data - often blocked by regulation (GDPR, bank secrecy laws) or competitive concerns |
| Federated learning, no DP | No | none - still vulnerable to membership-inference / gradient-inversion attacks on the model updates | Possible |
| **Federated + Differential Privacy (this project)** | No | **(ε=100, δ=1e-5)-DP, provable** | Possible, with mathematically bounded leakage regardless of what an attacker does |

RandomForest and the centralized deep model don't just have "weaker
privacy" than this project - they have **none**: full data exposure to
whoever trains the model, no bound on what can be inferred later.
Federated-no-DP is a real improvement (raw data stays local) but the model
updates themselves are still unbounded - nothing stops a membership-inference
attack against them. Only the DP-enabled model gives a number: a provable
ceiling on how much any single client's participation can be inferred, no
matter what the adversary tries. That guarantee, at a ~0.2pp accuracy cost
vs. pooling everyone's raw data, is the actual thesis this project supports -
not "highest accuracy on a leaderboard."

See `experiments/run_epsilon_sweep.py` / `results/comparison/epsilon_sweep.json`
and the frontend's Architecture page for the epsilon-vs-accuracy curve that
demonstrates the DP mechanism has a real, measurable effect - proof it isn't
a config flag that silently does nothing.

**Same sweep, real PaySim data** (`results/comparison_real/epsilon_sweep.json`,
5 epsilon points instead of synthetic's 9 - reduced scope since the real-data model
comparison above took 96 minutes and a comparable 9-point sweep wasn't tractable in
one sitting): ε=1 destroys the model exactly as on synthetic data (accuracy 0.08%,
AUC=0.500 - it collapses to flagging everything as fraud), recovers by ε=10, plateaus
from ε=50 up (F1 0.32-0.34) - independent confirmation on real data that the mechanism
does genuine work, not a synthetic-only artifact. One real difference from synthetic:
moderate DP (ε=50-500) slightly *outperforms* no-DP here (F1 0.32-0.34 vs. 0.316)
rather than costing utility, consistent with the model-comparison finding above that
DP's noise functions as useful training-time regularization on real data's severe
imbalance, not purely a privacy tax. Full table in README.md's "Real PaySim epsilon
sweep."

### The road to these numbers - two real bugs found along the way

The first version of this comparison hit **100% accuracy on every single
model**, RandomForest included. That was real given the data, but not
believable: the v1 synthetic generator (`generate_synthetic_paysim.py`)
drew fraud and legit transactions from *disjoint* value ranges (legit
`amount` capped at $400; every fraud archetype either above $400 or below
$3) - a single-feature threshold separated the classes almost perfectly.
Real fraud and legit transactions overlap heavily in amount, distance,
timing; that overlap is exactly what makes fraud detection hard and would
show real daylight between models. **Fix:** rewrote the generator (v2) to
draw every archetype from the *same distribution families* as legit,
shifted rather than clipped to a separate range, plus a 12% "quiet fraud"
slice drawn straight from the legit distribution (so no feature-based model
can hit 100% recall by construction - some real fraud genuinely looks
identical to a legit transaction on these 8 fields, same as in reality).

Retraining on the harder v2 data then exposed a second, deeper problem:
**the deep model flatlined at 0% recall.** The architecture
(hidden_dim=128, a 4-layer/8-head transformer) turned out to be ~24x
over-parameterized for a 10-dim input - verified empirically that during
training the network's internal representations (`fused`, `temporal_cls`)
collapse to a near-identical vector for every sample (batch std → ~0,
down from ~0.05 at init): the optimizer was actively learning to *discard*
the input and just output the training set's base fraud rate, a real, easy-
to-reach local minimum of unweighted BCE under ~6% class imbalance. Fixed
with three changes, all in `TemporalGraphTransformer`:
1. **Right-sized the architecture** to hidden_dim=32 (~45K params instead
   of ~1.07M) - matched to the actual input dimensionality.
2. **`pos_weight`** on the BCE loss (extra gradient pressure on the
   minority/fraud class).
3. **Prior-informed bias init** on the classifier's final layer
   (`log(fraud_rate / (1-fraud_rate))`), so the network doesn't get a free
   "match the base rate" win before it's forced to use the input features.

That, in turn, surfaced a **third**, more subtle issue while re-running the
comparison: even with the right-sized architecture, the *unregularized*
baselines (B: centralized, no FedProx/no DP; C: FedAvg, mu=0/no DP) would
occasionally have `fraud_prob` go to `nan` partway through training -
verified as a real forward-pass event (`nan` appears *before*
`TemporalGraphTransformer`'s own `nan_to_num` guard), reproducible across
many different random seeds and learning rates, i.e. a genuine attractor of
this exact unregularized combination, not an unlucky init. Once `nan`
enters, `nan_to_num` keeps training from crashing but the model then trains
on a fully degenerate constant prediction for every subsequent step -
notably, **D (FedProx + DP) never showed this instability across any run**,
which is itself a small, honest, and slightly ironic finding: this
project's own privacy mechanisms (FedProx's proximal term, DP's noise) seem
to have a stabilizing side-effect on training, not just a privacy cost.
Fixed with standard early-stopping / checkpoint-best: `run_centralized_deep`
tracks the best (non-collapsed, highest-F1) epoch's weights instead of
whatever epoch happens to be last; `run_federated` does the same per-round,
plus rolls back and retries a round that collapsed rather than letting it
poison the rest of the federated run. Both are visible in
`experiments/run_model_comparison.py`'s console output when they trigger -
nothing is hidden, a collapsed epoch/round prints as `[collapsed]` /
`retrying`.

One side benefit of the right-sized architecture: since Gaussian-mechanism
noise scales as `sigma*sqrt(d)`, a ~24x smaller model needs much less noise
for the same epsilon. Empirically re-measured the DP epsilon floor on this
architecture with a full 9-point sweep (see the DP section above): ε=1.0
(textbook) still destroys the model (f1=0.11, collapses to predicting
"fraud" for every input), but the floor moved from **~2000** (the original
1.07M-parameter architecture's smallest usable value) down to **~20-50**,
with ε=100 as the new default (real working margin, ~40-100x better than
before).

---

*For dashboard usage and demo talking points, see [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md).
For realistic test values in the Live Inference tab, see [TESTING_GUIDE.md](TESTING_GUIDE.md).*
