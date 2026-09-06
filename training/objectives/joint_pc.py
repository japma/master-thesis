from pathlib import Path

from training.objectives.cspn import CSPNObjective
from utils.checkpoints import save_joint_pc


class JointPCObjective(CSPNObjective):
    """The joint PC trains on exactly the CSPN's signal -- negative log-likelihood of
    the encoded batch -- so only the checkpoint format differs."""

    def save_checkpoint(self, path: Path) -> None:
        save_joint_pc(self.model, path)
