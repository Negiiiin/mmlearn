"""Learnable logit scaling layer."""

import numpy as np
import torch
from hydra_zen import store


# modified from: https://github.com/facebookresearch/ImageBind/blob/main/imagebind/models/helpers.py
@store(group="modules/layers", provider="mmlearn")
class LearnableLogitScaling(torch.nn.Module):
    """Logit scaling layer.

    Parameters
    ----------
    logit_scale_init : float, optional, default=1/0.07
        Initial value of the logit scale.
    learnable : bool, optional, default=True
        If True, the logit scale is learnable. Otherwise, it is fixed.
    max_logit_scale : float, optional, default=100
        Maximum value of the logit scale.
    """

    def __init__(
        self,
        logit_scale_init: float = 1 / 0.07,
        learnable: bool = True,
        max_logit_scale: float = 100,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale_init = logit_scale_init
        self.learnable = learnable
        log_logit_scale = torch.ones([]) * np.log(self.logit_scale_init)
        if learnable:
            self.log_logit_scale = torch.nn.Parameter(log_logit_scale)
        else:
            self.register_buffer("log_logit_scale", log_logit_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaling to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_sz, seq_len, dim).
        """
        return torch.clip(self.log_logit_scale.exp(), max=self.max_logit_scale) * x

    def extra_repr(self) -> str:
        """Return the string representation of the layer."""
        return (
            f"logit_scale_init={self.logit_scale_init},learnable={self.learnable},"
            f" max_logit_scale={self.max_logit_scale}"
        )