from pathlib import Path

import wandb


# TODO unify checkpoint loading and naming
def load_from_wandb(
    ckpt_name: str, download_path: str, file_name: str | None = None
) -> Path:
    name = f"jmartini-tu-darmstadt/master-thesis/{ckpt_name}:latest"
    api = wandb.Api()
    artifact = api.artifact(name)
    artifact_dir = artifact.download(download_path)
    if file_name is not None:
        return Path(artifact_dir) / Path(file_name)
    else:
        raise NotImplementedError
