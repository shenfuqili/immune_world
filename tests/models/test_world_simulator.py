"""End-to-end forward test for `ImmuneWorld` (Sec. 4.1, Eq. 1)."""

from __future__ import annotations

import torch

from immune_world.models.world_simulator import ImmuneWorld


def test_world_simulator_trajectory_mode_output_shapes() -> None:
    model = ImmuneWorld(
        n_genes=32, d_model=16, n_layers=2, n_heads=2, n_cancer_types=3, n_cell_types=4
    )
    x = torch.rand(2, 5, 32)
    t = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1)
    out = model(x, t, perturbation=None)

    assert out["hidden"].shape == (2, 5, 16)
    assert out["perturbation_pred"].shape == (2, 5, 32)
    assert out["trajectory_pred"].shape == (2, 5, 32)
    assert out["shared_repr"].shape[0] == 2
    assert out["cancer_logits"].shape == (2, 3)


def test_world_simulator_counterfactual_branch_changes_output() -> None:
    torch.manual_seed(0)
    model = ImmuneWorld(
        n_genes=16, d_model=16, n_layers=2, n_heads=2, n_cancer_types=3, n_cell_types=4
    )
    x = torch.rand(1, 4, 16)
    t = torch.linspace(0, 1, 4).unsqueeze(0)
    no_pert = model(x, t, perturbation=None)

    pert_genes = torch.tensor([[2, 5, -1]])
    doses = torch.tensor([[1.0]])
    types = torch.tensor([0])
    with_pert = model(x, t, perturbation=(pert_genes, doses, types))

    # Counterfactual should shift the hidden state.
    assert not torch.allclose(no_pert["hidden"], with_pert["hidden"], atol=1e-5)


def test_world_simulator_parameter_count_is_within_paper_order() -> None:
    # Sanity check: tiny config (16×2×2) should be well under the paper's 68 M. This guards
    # against accidental over-parameterisation; full-config parity is validated separately.
    model = ImmuneWorld(
        n_genes=16, d_model=16, n_layers=2, n_heads=2, n_cancer_types=3, n_cell_types=4
    )
    total = sum(p.numel() for p in model.parameters())
    assert total < 1_000_000, f"tiny ImmuneWorld unexpectedly large: {total} params"
