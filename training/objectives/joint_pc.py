from pathlib import Path

import torch

from training.objectives.base import Batch, StepOutput
from training.objectives.cspn import CSPNObjective
from utils.checkpoints import save_joint_pc


class JointPCObjective(CSPNObjective):
    """The joint PC trains on the CSPN's signal -- negative log-likelihood of the
    encoded batch -- plus, optionally, the exact `log p(y | z)` the circuit can compute
    by marginalizing its own label variables. The joint term alone spends nearly all of
    its gradient on the 16 continuous dims; the conditional term is what makes the
    labels worth attending to.
    """

    def _loss(
        self, latent: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        joint = self.model(latent, labels)
        total = -joint.mean()
        metrics = {"joint_nll": total.detach()}

        weight = self.model.config.conditional_weight
        if weight:
            conditional = -self.model.conditional_log_prob(latent, labels).mean()
            total = total + weight * conditional
            metrics["conditional_nll"] = conditional.detach()
        metrics["total"] = total.detach()
        return total, metrics

    def train_step(self, batch: Batch) -> StepOutput:
        if batch.images is None or batch.labels is None:
            raise ValueError("Images and labels must be provided for JointPC training")
        images, labels = batch.images, batch.labels

        self.model.train()
        with torch.no_grad():
            latent = self.autoencoder.encode(images)

        loss, metrics = self._loss(latent, labels.long())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return StepOutput(metrics=metrics, batch_size=images.size(0))

    @torch.no_grad()
    def val_step(self, batch: Batch) -> StepOutput:
        if batch.images is None or batch.labels is None:
            raise ValueError("Images and labels must be provided for JointPC training")
        images, labels = batch.images, batch.labels

        self.model.eval()
        latent = self.autoencoder.encode(images)
        _, metrics = self._loss(latent, labels.long())
        return StepOutput(metrics=metrics, batch_size=images.size(0))

    def save_checkpoint(self, path: Path) -> None:
        save_joint_pc(self.model, path, source_artifact=self.source_artifact)
