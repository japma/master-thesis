"""Record the VAE kind on autoencoder versions logged before artifacts carried it.

Models trained on a VAE are named after the kind its artifact records; without one it
falls back to the model config, which cannot tell a TCVAE or a beta from the plain VAE.

    uv run python -m scripts.set_vae_kind            # show what would change
    uv run python -m scripts.set_vae_kind --apply    # write it

An already recorded kind is never overwritten.
"""

import argparse

import wandb
from utils.naming import VAE_KIND_KEY
from utils.wandb_utils import _qualified

KINDS = {
    "variational_celeba:v2": "variational_beta4",
    "variational_celeba:v3": "variational_tcvae",
    "variational_celeba:v4": "variational_beta0p5",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    api = wandb.Api()
    for ref, kind in KINDS.items():
        artifact = api.artifact(_qualified(ref, "latest"))
        run = artifact.logged_by()
        training = (run.config.get("training") or {}) if run is not None else {}
        print(
            f"{ref} aliases={artifact.aliases} run={run.name if run else None} "
            f"vae_type={training.get('vae_type')} beta={training.get('beta')}"
        )

        recorded = artifact.metadata.get(VAE_KIND_KEY)
        if recorded is not None:
            note = "" if recorded == kind else f" -- NOT {kind}, left alone"
            print(f"  already records {recorded}{note}")
            continue

        print(f"  {VAE_KIND_KEY} -> {kind}")
        if args.apply:
            artifact.metadata[VAE_KIND_KEY] = kind
            artifact.save()
            print("  saved")

    if not args.apply:
        print("\nDry run; pass --apply to write.")


if __name__ == "__main__":
    main()
