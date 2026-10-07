"""
Multi-round Rényi Differential Privacy (RDP) composition accounting for
DP-FedAvg (run_training.py's DifferentialPrivacy class).

Why this exists: every (epsilon, delta) this project reports elsewhere is a
PER-ROUND budget for the server's one Gaussian noise draw that round
(run_training.py's DifferentialPrivacy.noise_scale()). Across R rounds, the
TRUE cumulative privacy loss is larger than any single round's epsilon - but
naively summing R * epsilon_per_round (the loosest possible bound, "basic
composition") substantially overstates how much privacy is actually lost,
because independent applications of a noisy mechanism compose sub-linearly,
not linearly. This module implements the standard, tight approach for
composing Gaussian mechanisms (Mironov, "Rényi Differential Privacy", 2017):

1. Convert each round's Gaussian mechanism (sensitivity, sigma) into its
   Rényi divergence at a range of orders alpha - gaussian_rdp().
2. RDP composes by simple addition across rounds, at each fixed alpha -
   compose_rdp().
3. Convert the summed RDP back to a single (epsilon, delta)-DP guarantee by
   picking whichever alpha gives the tightest result - rdp_to_dp().

This gives three numbers worth comparing after R rounds:
  - epsilon_per_round: what's already reported everywhere else
  - epsilon_naive = R * epsilon_per_round: the loose "basic composition" bound
  - epsilon_rdp: the tight bound this module computes - always <= naive,
    because RDP composition is provably at least as tight as basic
    composition for the Gaussian mechanism.

No optimistic rounding: if epsilon_rdp still comes out large (this project's
per-round budgets are already large - ~50-100 - because of Gaussian noise's
sigma*sqrt(d) scaling against a ~45K-parameter model, see run_training.py's
DifferentialPrivacy docstring), that's reported as-is, not hidden.
"""
import math
from typing import List, Tuple

# A log-spaced range of Renyi orders to search over - standard practice
# (e.g. Opacus, TensorFlow Privacy use a similar grid). Order 1 is excluded
# (the RDP-to-DP conversion is undefined there - alpha must be > 1).
DEFAULT_ALPHAS = (
    [1 + x / 10.0 for x in range(1, 100)] +
    [10 + x for x in range(0, 90)] +
    [100 + x * 10 for x in range(0, 40)]
)


def gaussian_rdp(alpha: float, sigma: float, sensitivity: float) -> float:
    """
    Renyi divergence of order `alpha` for one application of the Gaussian
    mechanism with the given L2 sensitivity and noise std `sigma`.

    Standard closed form for the Gaussian mechanism (Mironov 2017, Prop. 7):
        RDP(alpha) = alpha * sensitivity^2 / (2 * sigma^2)
    """
    if sigma <= 0:
        return float("inf")
    return alpha * (sensitivity ** 2) / (2.0 * sigma ** 2)


def compose_rdp(rounds: List[Tuple[float, float]], alphas=DEFAULT_ALPHAS) -> dict:
    """
    rounds: list of (sigma, sensitivity) tuples, one per round actually run.
    Returns {alpha: total_rdp} - RDP composes by simple addition at each
    fixed order, independent of whether sigma/sensitivity varied per round
    (e.g. because max_client_weight differed round to round with different
    client subsets selected).
    """
    totals = {alpha: 0.0 for alpha in alphas}
    for sigma, sensitivity in rounds:
        for alpha in alphas:
            totals[alpha] += gaussian_rdp(alpha, sigma, sensitivity)
    return totals


def rdp_to_dp(rdp_per_alpha: dict, delta: float) -> Tuple[float, float]:
    """
    Convert a total-RDP-per-order dict to a single (epsilon, best_alpha)
    (epsilon, delta)-DP guarantee, via the standard conversion (Mironov
    2017, Prop. 3): for any alpha > 1,
        epsilon(alpha) = rdp(alpha) + ln(1/delta) / (alpha - 1)
    and the tightest valid epsilon is the minimum over alpha searched.
    """
    best_eps, best_alpha = float("inf"), None
    log_inv_delta = math.log(1.0 / delta)
    for alpha, rdp in rdp_per_alpha.items():
        if alpha <= 1:
            continue
        eps = rdp + log_inv_delta / (alpha - 1)
        if eps < best_eps:
            best_eps, best_alpha = eps, alpha
    return best_eps, best_alpha


def gaussian_rdp_single_round_eps(sigma: float, sensitivity: float, delta: float) -> float:
    """The RDP-tight (epsilon, delta)-DP guarantee for ONE Gaussian mechanism
    application with this sigma/sensitivity - i.e. what epsilon this round
    actually, rigorously achieves, as opposed to whatever nominal epsilon was
    fed into the classical calibration formula that produced this sigma.
    These two values match closely when the classical formula's calibration
    target was epsilon<=1 (where it's proven tight) and diverge substantially
    above that (verified: at the project's actual epsilon=100 default, the
    classical formula's "epsilon=100" label corresponds to an RDP-tight
    epsilon of ~313 - see module docstring)."""
    rdp = compose_rdp([(sigma, sensitivity)])
    eps, _ = rdp_to_dp(rdp, delta)
    return eps


class DPFedAvgAccountant:
    """
    Tracks every round's actual (sigma, sensitivity) as DP-FedAvg training
    runs, and reports the real cumulative privacy cost at any point - both
    the tight RDP-composed epsilon and the loose naive-sum epsilon, so the
    gap between them (RDP composition's whole value proposition) is visible
    rather than asserted.
    """

    def __init__(self, delta: float):
        self.delta = delta
        self.rounds: List[Tuple[float, float, float]] = []  # (sigma, sensitivity, per_round_epsilon)

    def add_round(self, sigma: float, sensitivity: float, per_round_epsilon: float):
        self.rounds.append((sigma, sensitivity, per_round_epsilon))

    def cumulative(self) -> dict:
        """Returns the real cumulative privacy accounting after every round
        added so far. Three numbers that must not be conflated:

        - epsilon_naive_total_nominal: R * the CONFIGURED per-round epsilon
          label (e.g. R * 100). This is the number someone would wrongly
          assume is the total if they just multiplied the label everyone
          else in this project quotes by the round count. It is NOT a valid
          bound - see the next field.
        - epsilon_true_single_round: the RDP-tight (epsilon,delta)-DP
          guarantee ONE round's mechanism actually achieves - this project's
          classical calibration formula is only proven tight for
          epsilon<=1, so above that this is materially larger than the
          nominal label (e.g. ~3.13x at the project's epsilon=100 default -
          see gaussian_rdp_single_round_eps's docstring).
        - epsilon_naive_total_true: R * epsilon_true_single_round - the
          correct "naive/basic composition" baseline, i.e. what you'd get
          composing the TRUE per-round cost without RDP's tighter method.
        - epsilon_rdp_total: the actual tight multi-round total via RDP
          composition. epsilon_rdp_total < epsilon_naive_total_true
          demonstrates RDP composition is doing real work, even though
          epsilon_rdp_total > epsilon_naive_total_nominal (comparing against
          the wrong, too-small nominal baseline would be misleading).
        """
        if not self.rounds:
            return {
                "num_rounds": 0, "epsilon_per_round_nominal_avg": 0.0,
                "epsilon_true_single_round": 0.0,
                "epsilon_naive_total_nominal": 0.0, "epsilon_naive_total_true": 0.0,
                "epsilon_rdp_total": 0.0, "best_alpha": None,
            }
        sigma_sensitivity_pairs = [(s, sen) for s, sen, _ in self.rounds]
        nominal_epsilons = [e for _, _, e in self.rounds]

        rdp_totals = compose_rdp(sigma_sensitivity_pairs)
        eps_rdp, best_alpha = rdp_to_dp(rdp_totals, self.delta)

        true_single_round_epsilons = [
            rdp_to_dp(compose_rdp([(s, sen)]), self.delta)[0] for s, sen in sigma_sensitivity_pairs
        ]
        avg_true_single_round = sum(true_single_round_epsilons) / len(true_single_round_epsilons)

        return {
            "num_rounds": len(self.rounds),
            "epsilon_per_round_nominal_avg": sum(nominal_epsilons) / len(nominal_epsilons),
            "epsilon_true_single_round": avg_true_single_round,
            "epsilon_naive_total_nominal": sum(nominal_epsilons),
            "epsilon_naive_total_true": avg_true_single_round * len(self.rounds),
            "epsilon_rdp_total": eps_rdp,
            "best_alpha": best_alpha,
            "delta": self.delta,
        }
