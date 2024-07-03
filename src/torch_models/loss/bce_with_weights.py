from typing import Callable, Optional
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

def binary_cross_entropy_with_logits_and_weights(
    logits, target, weights):
    """
    Calculate binary cross-entropy with elementwise weights

    Parameters
    ----------
    logits: torch.Tensor
    target: torch.Tensor
    weights: torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    return (F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction='none'
    ) * weights).mean()

class BCEWithLogitsLossAndWeights(_Loss):
    """
    Binary Cross Entropy Loss with logits and elementwise weights
    """
    def __init__(self, weight: Optional[Tensor] = None, size_average=None,
                 reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None) -> None:
        super(BCEWithLogitsLossAndWeights, self).__init__(size_average,reduce,reduction)

    def forward(self, input: Tensor, target: Tensor, elementwise_weights: Tensor) -> Tensor:
        return binary_cross_entropy_with_logits_and_weights(
            input, target, elementwise_weights)