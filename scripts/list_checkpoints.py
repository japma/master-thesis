"""List the checkpoints on Weights & Biases, newest first, with what references them.

Two recurring failures motivated this: a config naming an artifact that no longer exists
(or never did), and an artifact whose architecture has drifted from the current code so it
downloads fine but fails to load. The first is caught by the cross-reference column, the
second by --check.

    uv run list_checkpoints                    # collections, newest version, references
    uv run list_checkpoints --check            # also try loading each one (downloads)
    uv run list_checkpoints --type autoencoder # one artifact type
    uv run list_checkpoints --versions 5       # more history per collection
    uv run list_checkpoints --intermediate     # include intermediate_* collections

`--check` answers "does this deserialize under the current code", not "is this still
meaningful". A CSPN trained against a retired autoencoder's latent space loads perfectly
and is worthless; only the checkpoint's provenance tells you that.
"""

import argparse
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

import wandb
from utils.wandb_utils import ENTITY, PROJECT

CONFIG_ROOT = Path("configs")

# wandb auto-creates a `run-<id>-history` artifact per run under its own types; those are
# telemetry, not checkpoints, and would bury the listing.
_NOISE_TYPE_PREFIX = "wandb-"
_NOISE_NAME_PREFIX = "run-"

# Collections shown per type before the listing is truncated (--all shows everything).
_DEFAULT_LIMIT = 12

# Artifact type -> loader for --check. Types not listed here are only listed, not loaded.
_LOADERS = {
    "autoencoder": "load_ae_from_path",
    "cspn": "load_cspn_from_path",
    "joint_pc": "load_joint_pc_from_path",
    "label_pc": "load_label_pc_from_path",
    "nn_baseline": "load_nn_baseline_from_path",
}


def _config_references() -> dict[str, list[str]]:
    """Map wandb artifact name -> config files naming it as their autoencoder.

    Skips `external: true`, where `name` is a HuggingFace repo rather than an artifact.
    """
    references: dict[str, list[str]] = {}
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(raw, dict):
            continue
        autoencoder = raw.get("autoencoder")
        if not isinstance(autoencoder, dict) or autoencoder.get("external"):
            continue
        name = autoencoder.get("name")
        if isinstance(name, str):
            references.setdefault(name, []).append(str(path))
    return references


def _collections(api: wandb.Api, only_type: str | None) -> Iterator[tuple[str, Any]]:
    for artifact_type in api.artifact_types(f"{ENTITY}/{PROJECT}"):
        type_name: str = artifact_type.name
        if only_type is not None and type_name != only_type:
            continue
        if only_type is None and type_name.startswith(_NOISE_TYPE_PREFIX):
            continue
        for collection in artifact_type.collections():
            if collection.name.startswith(_NOISE_NAME_PREFIX):
                continue
            yield type_name, collection


def _try_load(artifact_type: str, name: str, version: str) -> str:
    """Download an artifact and run it through the real loader. Returns a status string."""
    loader_name = _LOADERS.get(artifact_type)
    if loader_name is None:
        return ""

    from utils import checkpoints
    from utils.wandb_utils import load_from_wandb

    loader = getattr(checkpoints, loader_name)
    try:
        path = load_from_wandb(ckpt_name=name, tag=version)
        loader(path, device="cpu")
    except Exception as error:
        first_line = str(error).splitlines()[0] if str(error) else type(error).__name__
        return f"FAILS: {first_line[:70]}"
    return "loads"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--type", help="only this artifact type (e.g. autoencoder, cspn)"
    )
    parser.add_argument(
        "--versions", type=int, default=1, help="versions to show per collection"
    )
    parser.add_argument(
        "--intermediate",
        action="store_true",
        help="include intermediate_* collections (hidden by default)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"show every collection (default: the {_DEFAULT_LIMIT} newest per type)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="download each newest version and try loading it with the current code",
    )
    args = parser.parse_args()

    api = wandb.Api()
    references = _config_references()
    seen_names: set[str] = set()

    grouped: dict[str, list[tuple[datetime, str, list[Any]]]] = {}
    for type_name, collection in _collections(api, args.type):
        name = collection.name
        if not args.intermediate and name.startswith("intermediate_"):
            continue
        versions = list(collection.artifacts())[: max(1, args.versions)]
        if not versions:
            continue
        newest = datetime.fromisoformat(versions[0].created_at.replace("Z", "+00:00"))
        grouped.setdefault(type_name, []).append((newest, name, versions))

    for type_name in sorted(grouped):
        entries = sorted(grouped[type_name], key=lambda row: row[0], reverse=True)
        shown = entries if args.all else entries[:_DEFAULT_LIMIT]
        hidden = len(entries) - len(shown)
        print(f"\n{type_name}")
        for _, name, versions in shown:
            seen_names.add(name)
            for i, artifact in enumerate(versions):
                created = artifact.created_at.replace("T", " ").replace("Z", "")
                aliases = ",".join(a for a in artifact.aliases) or "-"
                if i == 0:
                    used_by = references.get(name, [])
                    suffix = f"  <- {', '.join(used_by)}" if used_by else ""
                    status = (
                        f"  [{_try_load(type_name, name, artifact.version)}]"
                        if args.check
                        else ""
                    )
                    print(
                        f"  {name:40s} {artifact.version:>4s}  {created} UTC  "
                        f"{aliases:12s}{status}{suffix}"
                    )
                else:
                    print(
                        f"  {'':40s} {artifact.version:>4s}  {created} UTC  {aliases}"
                    )
        if hidden:
            print(f"  ... {hidden} older collection(s) hidden, use --all")

    dangling = {n: c for n, c in references.items() if n not in seen_names}
    if dangling:
        print("\nconfigs naming an artifact that does not exist:")
        for name, configs in sorted(dangling.items()):
            print(f"  {name:40s} <- {', '.join(configs)}")
    elif not args.type:
        print("\nevery config's autoencoder name resolves to an existing artifact.")


if __name__ == "__main__":
    main()
