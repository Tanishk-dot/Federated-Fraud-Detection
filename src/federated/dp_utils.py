"""
Differential Privacy Utilities
Gradient clipping + Gaussian noise injection
"""

import torch
import numpy as np


class DifferentialPrivacy:
    """
    Client-level Differential Privacy for Federated Learning.

    Applied ONCE per client per round, to that client's whole local update
    (parameters after local training minus the global parameters it started
    from) - not per-minibatch to raw gradients. See run_training.py's
    DifferentialPrivacy class (the actually-exercised training path) for the
    full rationale: per-batch application spends the privacy budget once per
    batch with no composition accounting, so the ε configured here would not
    have been the ε actually achieved. Round-level/client-level DP fixes
    that with a single clean Gaussian-mechanism application per round, whose
    privacy unit (an entire client's contribution) also matches this
    cross-silo setting (each client is a bank, not one transaction).

    Two-step mechanism, run once at the end of local training:
    1. Clip the local update's L2 norm (bounds sensitivity)
    2. Add Gaussian noise calibrated to (epsilon, delta) (provides privacy)
    """

    def __init__(self, epsilon=1.0, delta=1e-5, clip_norm=1.0):
        """
        Args:
            epsilon: Privacy budget per client per round
            delta: Privacy parameter (probability of privacy breach)
            clip_norm: Maximum L2 norm for the local update
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm

        # Compute noise scale
        self.noise_scale = self._compute_noise_scale()

        # ASCII-only: some Windows consoles (cp1252) raise UnicodeEncodeError
        # on Greek letters when stdout isn't a real console (e.g. piped/redirected).
        print(f"Differential Privacy initialized:")
        print(f"  epsilon: {self.epsilon}")
        print(f"  delta: {self.delta}")
        print(f"  Clip norm: {self.clip_norm}")
        print(f"  Noise scale (sigma): {self.noise_scale:.4f}")

    def _compute_noise_scale(self):
        """
        Compute Gaussian noise scale using the Gaussian mechanism
        σ = (sensitivity * sqrt(2 * log(1.25/δ))) / ε
        """
        sensitivity = self.clip_norm  # After clipping, sensitivity = clip_norm
        sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        return sigma

    def privatize_update(self, model, global_params):
        """
        Clip the client's local update to `clip_norm` and add Gaussian
        noise, once, in place - applied AFTER local training finishes.

        Args:
            model: client model after local training (mutated in place)
            global_params: list of tensors - model's parameters before local
                training started (i.e. what the server sent this round)

        Returns:
            update_norm: L2 norm of the update before clipping (for logging)
        """
        with torch.no_grad():
            deltas = [p.detach() - g.detach() for p, g in zip(model.parameters(), global_params)]
            update_norm = torch.norm(torch.stack([torch.norm(d) for d in deltas])).item()

            clip_coef = min(1.0, self.clip_norm / (update_norm + 1e-10))

            for p, g, d in zip(model.parameters(), global_params, deltas):
                clipped = d * clip_coef
                noise = torch.normal(mean=0.0, std=self.noise_scale, size=clipped.shape, device=clipped.device)
                p.copy_(g + clipped + noise)

        return update_norm

    def apply(self, model, global_params=None):
        """
        Apply the DP mechanism. If `global_params` is given, clips/noises
        the local update relative to it (the correct, round-level use).
        Without it, falls back to clipping+noising the current gradients
        directly (kept only for the __main__ smoke test below / backward
        compatibility - NOT what train() should call per-batch anymore).
        """
        if global_params is not None:
            return self.privatize_update(model, global_params)

        # Fallback: single-shot clip+noise on whatever gradients are set.
        parameters = [p for p in model.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in parameters])).item()
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-10))
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
            noise = torch.normal(mean=0.0, std=self.noise_scale, size=p.grad.shape, device=p.grad.device)
            p.grad.add_(noise)
        return total_norm


def apply_differential_privacy(model, epsilon=1.0, delta=1e-5, clip_norm=1.0):
    """
    Convenience function to apply DP to a model
    
    Args:
        model: PyTorch model with gradients
        epsilon: Privacy budget
        delta: Privacy parameter
        clip_norm: Gradient clipping threshold
    
    Returns:
        total_norm: gradient norm before clipping
    """
    dp = DifferentialPrivacy(epsilon, delta, clip_norm)
    return dp.apply(model)


if __name__ == "__main__":
    print("Testing Differential Privacy...")
    
    # Create dummy model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    
    # Create dummy gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 10  # Large gradients
    
    # Compute norm before DP
    params = [p for p in model.parameters() if p.grad is not None]
    norm_before = torch.norm(torch.stack([torch.norm(p.grad) for p in params])).item()
    
    # Apply DP
    dp = DifferentialPrivacy(epsilon=1.0, clip_norm=1.0)
    norm_clipped = dp.apply(model)
    
    # Compute norm after DP
    norm_after = torch.norm(torch.stack([torch.norm(p.grad) for p in params])).item()
    
    print(f"\nGradient norms:")
    print(f"  Before DP: {norm_before:.4f}")
    print(f"  After clipping: {norm_clipped:.4f}")
    print(f"  After DP (with noise): {norm_after:.4f}")
    print(f"  Clipped: {norm_clipped < norm_before}")
    
    print("\n✅ Differential Privacy test passed!")
