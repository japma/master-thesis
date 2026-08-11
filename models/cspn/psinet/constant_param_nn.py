from __future__ import annotations

import torch
import torch.nn as nn

from models.cspn.psinet.param_shapes import derive_layer_param_shapes


class ConstantParamNN(nn.Module):
    def __init__(self, einet_layers: list) -> None:
        super().__init__()
        shapes = derive_layer_param_shapes(einet_layers)
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(*shape) * 0.1) for shape in shapes]
        )

    def forward(
        self, y: torch.Tensor, x_unused: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        batch_size: int = y.shape[0]
        return [p.unsqueeze(0).expand(batch_size, *p.shape) for p in self.params]
