"""Batch collation with masked-gene (MLM-style) token masking.

Ref: Sec. 4.6 — `L_recon` "applied to masked genes under an MLM paradigm". Paper does not state the
masking rate; see `docs/deviations.md` for the default (BERT-style 15 %).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor


def collate_trajectory_batch(batch: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Stack variable-length trajectory windows with right-padding + attention mask.

    Expected per-item keys: ``x`` ∈ R^{T × G}, ``pseudo_time`` ∈ R^{T} (both mandatory);
    ``perturbation_genes`` ∈ Z^{k}, ``doses`` ∈ R^{dose_dim}, ``types`` ∈ Z (optional).

    Returns a dict with right-padded tensors and a ``valid_mask`` of shape (B, T_max) that is
    True on real time steps and False on padding; `pseudo_time` padding is filled with the last
    real value so the transformer's temporal decay bias remains well defined.
    """
    if len(batch) == 0:
        raise ValueError("cannot collate an empty batch")
    for key in ("x", "pseudo_time"):
        if key not in batch[0]:
            raise KeyError(f"each batch item must contain '{key}'")

    t_max = max(int(item["x"].shape[0]) for item in batch)
    n_genes = int(batch[0]["x"].shape[-1])
    batch_size = len(batch)

    x = torch.zeros(batch_size, t_max, n_genes, dtype=batch[0]["x"].dtype)
    pt = torch.zeros(batch_size, t_max, dtype=batch[0]["pseudo_time"].dtype)
    valid = torch.zeros(batch_size, t_max, dtype=torch.bool)

    for i, item in enumerate(batch):
        t = int(item["x"].shape[0])
        x[i, :t] = item["x"]
        pt[i, :t] = item["pseudo_time"]
        if t < t_max:
            pt[i, t:] = item["pseudo_time"][-1]  # clamp to last real time (bias stays in-range)
        valid[i, :t] = True

    out: dict[str, Tensor] = {"x": x, "pseudo_time": pt, "valid_mask": valid}

    if "perturbation_genes" in batch[0]:
        k_max = max(int(item["perturbation_genes"].shape[0]) for item in batch)
        pg = torch.full((batch_size, k_max), -1, dtype=torch.long)
        for i, item in enumerate(batch):
            k = int(item["perturbation_genes"].shape[0])
            pg[i, :k] = item["perturbation_genes"]
        out["perturbation_genes"] = pg
        out["doses"] = torch.stack([item["doses"] for item in batch])
        out["types"] = torch.stack([item["types"] for item in batch]).long()

    return out


def apply_gene_mlm_mask(
    x: Tensor, rate: float = 0.15, *, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Return (masked_x, mask_bool). BERT-style: zero-out masked positions at `rate`.

    `x` has shape (..., G). `mask` is a bool tensor of the same shape; True = position was masked.
    The paper is silent on `rate`; the 15 % default is logged in `docs/deviations.md`.
    """
    if not (0.0 <= rate <= 1.0):
        raise ValueError(f"rate must be in [0, 1]; got {rate}")
    if rate == 0.0:
        return x.clone(), torch.zeros_like(x, dtype=torch.bool)
    probs = torch.full_like(x, rate, dtype=torch.float32)
    mask = torch.bernoulli(probs, generator=generator).to(torch.bool)
    masked = x.masked_fill(mask, 0.0)
    return masked, mask
