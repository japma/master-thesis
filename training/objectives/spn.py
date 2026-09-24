from pathlib import Path

from training.objectives.cspn import CSPNObjective
from utils.checkpoints import save_spn


class SPNObjective(CSPNObjective):
    """The CSPN's NLL objective; the SPN ignores the labels it is handed."""

    def save_checkpoint(self, path: Path) -> None:
        save_spn(self.model, path, source_artifact=self.source_artifact)
