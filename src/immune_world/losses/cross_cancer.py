"""Cross-cancer adversarial (GRL) loss.

Ref: Sec. 4.4, Eq. (6) — `L_transfer = L_task − λ_adv · L_disc`, λ_adv = 0.1.
The gradient reversal happens upstream inside `CrossCancerHead` so this loss is a plain CE on the
cancer-type logits produced after reversal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class CrossCancerAdversarialLoss(nn.Module):
    """Cross-entropy on cancer-type prediction — gradient reversed in the shared encoder."""

    def __init__(self, lambda_adv: float = 0.1) -> None:
        super().__init__()
        self.lambda_adv = lambda_adv
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, cancer_logits: Tensor, cancer_labels: Tensor) -> Tensor:
        loss: Tensor = self.criterion(cancer_logits, cancer_labels)
        return loss
