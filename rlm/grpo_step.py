"""One GRPO update step, written by hand. This is the part you must understand, not just run.

``GRPOTrainer`` hides the algorithm behind a ``.train()`` call. Here you reimplement its
core on plain tensors, following the slides step by step:

    Step 2: sample a group of G outputs for the same question   (done by the caller)
    Step 3: advantages   A_i = (r_i - mean(r)) / (std(r) + eps)
    Step 4: policy ratio ρ_{i,t} = π_θ(o_{i,t}) / π_{θ_old}(o_{i,t})
            clipped objective  min(ρ·A, clip(ρ, 1-ε, 1+ε)·A)
            optional KL penalty  D_KL(π_θ || π_ref) ≈ exp(logp_ref - logp) - (logp_ref - logp) - 1
            loss = -( mean over tokens of clipped objective  -  β · KL )

Everything works on *log-probabilities per token*, shaped ``(G, T)``, with a mask that is
1 on completion tokens and 0 on padding. Do not touch the model here: the caller computes
the log-probs; you compute the loss.

Run the tests to check the implementation::

    uv run pytest tests/test_grpo_step.py -v
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4, scale: bool = True) -> torch.Tensor:
    """Step 3 of the slides: normalise rewards within the group.

    Args:
        rewards: shape ``(G,)``, one scalar reward per sampled output.
        eps: numerical guard for the standard deviation.
        scale: if False, return ``r_i - mean(r)`` without dividing by the std.

    Returns:
        Advantages with shape ``(G,)``. A positive advantage means "better than the group".
    """
    centered = rewards - rewards.mean()
    if not scale:
        return centered
    # Population std (unbiased=False) so that a group of two gets advantages of exactly ±1.
    # When every output gets the same reward the numerator is 0: no signal, as it should be.
    return centered / (rewards.std(unbiased=False) + eps)


def policy_ratio(logp_new: torch.Tensor, logp_old: torch.Tensor) -> torch.Tensor:
    """ρ_{i,t} = π_θ / π_{θ_old}, computed in log space for numerical stability.

    Both inputs have shape ``(G, T)``. Return a tensor of the same shape.
    """
    return torch.exp(logp_new - logp_old)


def clipped_objective(
    ratio: torch.Tensor, advantages: torch.Tensor, epsilon: float = 0.2
) -> torch.Tensor:
    """Per-token GRPO surrogate with clipping (variation 1 in the slides).

    Args:
        ratio: shape ``(G, T)``.
        advantages: shape ``(G,)``; broadcast over the token dimension.
        epsilon: clipping range.

    Returns:
        Per-token objective, shape ``(G, T)``, *before* masking and averaging.
    """
    adv = advantages.unsqueeze(-1)  # (G, 1): the same advantage for every token of output i
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
    # The min makes the objective pessimistic: moving the ratio outside [1-ε, 1+ε] can never
    # increase it, so the update has no incentive to push the policy far from θ_old.
    return torch.minimum(unclipped, clipped)


def kl_penalty(logp_new: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Per-token estimate of D_KL(π_θ || π_ref) used by DeepSeekMath (variation 2).

    exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1. Always >= 0. Shape ``(G, T)``.
    """
    log_ratio = logp_ref - logp_new
    return torch.exp(log_ratio) - log_ratio - 1.0


def grpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 0.2,
    beta: float = 0.0,
    logp_ref: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Put the pieces together and return the scalar loss to minimise.

    Average the per-token objective over the valid tokens of each output (``1/|o_i|`` in the
    formula), then over the group (``1/G``). Subtract β times the masked-mean KL when
    ``beta > 0``. Return ``(-objective, stats)`` where ``stats`` holds at least
    ``mean_advantage``, ``clip_fraction`` (share of tokens where clipping was active) and
    ``kl`` for logging.
    """
    mask = mask.to(logp_new.dtype)
    tokens_per_output = mask.sum(dim=-1).clamp(min=1.0)

    def masked_mean(per_token: torch.Tensor) -> torch.Tensor:
        """1/|o_i| Σ_t over valid tokens, then 1/G Σ_i."""
        return ((per_token * mask).sum(dim=-1) / tokens_per_output).mean()

    advantages = group_advantages(rewards).detach()
    ratio = policy_ratio(logp_new, logp_old.detach())
    objective = masked_mean(clipped_objective(ratio, advantages, epsilon))

    kl = torch.zeros((), dtype=logp_new.dtype)
    if beta > 0:
        if logp_ref is None:
            raise ValueError("beta > 0 needs the reference log-probs (logp_ref)")
        kl = masked_mean(kl_penalty(logp_new, logp_ref.detach()))
        objective = objective - beta * kl

    with torch.no_grad():
        # Clipping is active where the ratio left [1-ε, 1+ε] in the direction the advantage
        # pushes: there the gradient of that token is zero.
        adv = advantages.unsqueeze(-1)
        clipped = ((ratio > 1 + epsilon) & (adv > 0)) | ((ratio < 1 - epsilon) & (adv < 0))
        clip_fraction = (clipped.to(mask.dtype) * mask).sum() / mask.sum().clamp(min=1.0)

    stats = {
        "mean_advantage": advantages.mean().item(),
        "reward_mean": rewards.float().mean().item(),
        "reward_std": rewards.float().std(unbiased=False).item(),
        "clip_fraction": clip_fraction.item(),
        "kl": kl.item(),
        "mean_ratio": ((ratio.detach() * mask).sum() / mask.sum().clamp(min=1.0)).item(),
    }
    return -objective, stats
