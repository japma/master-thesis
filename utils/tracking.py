import wandb


class WandbTracker:
    def __init__(self, wandb_cfg: dict[str, str]):
        self.dataset = wandb_cfg.get("dataset")
        # TODO use actual values
        self.model = wandb_cfg.get("model", "Unknown Model")
        self.name = f"{self.dataset}_{self.model}"

        # TODO values to add
        # - seed
        # - device ???
        # - paths
        #   - for loading
        #   - for saving
        self.config = {
            "dataset": self.dataset,
            "model": self.model,
            "epochs": wandb_cfg.get("epochs"),
            "seed": wandb_cfg.get("seed"),
        }

        self.run = wandb.init(
            entity="jmartini-tu-darmstadt",
            project="master-thesis",
            name=self.name,
            config=self.config,
        )

    def log(self, metrics: dict, step: int = None):
        self.run.log(metrics, step=step)

    def finish(self):
        self.run.alert(
            title=f"{self.name} finished",
            text=f"Model: {self.model}, Dataset: {self.dataset}",
        )
        self.run.finish()
