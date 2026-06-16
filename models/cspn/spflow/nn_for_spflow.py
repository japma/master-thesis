import torch
from torch import nn


class NeuralNetworkForSPFlow(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_leaves: int,
        num_repetitions: int,
        num_layers: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_leaves = num_leaves
        self.num_repetitions = num_repetitions
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(num_classes, hidden_dim)

        layers = []
        for _ in range(self.num_layers):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)

        out_dim = self.num_leaves * self.num_repetitions
        self.loc_head = nn.Linear(self.hidden_dim, out_dim)
        self.scale_head = nn.Linear(self.hidden_dim, out_dim)

    def forward(self, evidence: torch.Tensor) -> dict[str, torch.Tensor]:
        labels = evidence.long().squeeze(-1)
        one_hot = nn.functional.one_hot(labels, num_classes=self.num_classes).float()
        h = self.backbone(self.input_proj(one_hot))

        loc = self.loc_head(h).view(-1, 1, self.num_leaves, self.num_repetitions)
        scale = (
            self.scale_head(h)
            .view(-1, 1, self.num_leaves, self.num_repetitions)
            .exp()
            .clamp(min=1e-4)
        )

        return {"loc": loc, "scale": scale}
