"""Where is a joint PC failing: `p(y)`, or the way a factor steers the latents?

Three questions per label factor, because they have different fixes.

  1. Does the circuit know the factor's marginal? Its exact `p(y_f)` against the
     training labels. Wrong here means the easy part was not learned -- a training
     problem, not a structural one.
  2. Does conditioning on that factor move the latents at all? Sample
     `p(z | factor = v)` with every other factor marginalized, for each value v, and
     measure how far apart those means are. Near zero means the factor is inert.
  3. Is that movement the *right* size? The same spread measured on real latents is
     the target. The ratio is the headline: 1.0 means the factor steers the latents as
     much as it does in the data, 0.0 means the label is being ignored.

    uv run diagnose_joint_pc configs/evaluation/colour_mnist_skewed_joint_pc.yaml
"""

import pandas as pd
import torch

from dataset_loaders import build_data_loaders
from dataset_loaders.colour_mnist import FACTOR_NAMES
from evaluation.conditionals import CARDINALITIES, training_labels
from evaluation.evaluate import RUN_KEYS, write_metric
from evaluation.generate import (
    encode_and_decode,
    load_generative_model,
    load_vae,
    resolve_autoencoder,
)
from models.cspn.joint_pc import JointPC
from utils.config import EvaluationRunConfig, load_config
from utils.progress import start_rtpt
from utils.reproducibility import resolve_device

FILENAME = "factor_steering.csv"
SAMPLES_PER_VALUE = 256


def spread_of_means(means: torch.Tensor) -> float:
    """How far a factor's per-value latent means sit from their common centre."""
    return float((means - means.mean(dim=0)).norm(dim=1).mean())


def label_distribution(model: JointPC, device: torch.device) -> torch.Tensor:
    """The circuit's `p(y)` over every (digit, fg, bg) cell, as a normalized table."""
    cells = torch.tensor(
        [
            [digit, fg, bg]
            for digit in range(CARDINALITIES[0])
            for fg in range(CARDINALITIES[1])
            for bg in range(CARDINALITIES[2])
        ],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        joint = model.label_log_marginal(cells).exp().cpu()
    joint = joint / joint.sum()
    return joint.reshape(*CARDINALITIES)


def factor_report(
    model: JointPC,
    real_latents: torch.Tensor,
    real_labels: torch.Tensor,
    device: torch.device,
    labels: torch.Tensor,
) -> pd.DataFrame:
    """One row per label factor: does it steer the latents, and by how much."""
    # p(y_f) needs the *other* label factors summed out, not pinned to zero.
    joint = label_distribution(model, device)

    rtpt = start_rtpt(f"diagnose_{len(FACTOR_NAMES)}factors", len(FACTOR_NAMES))
    rows = []
    for factor, name in enumerate(FACTOR_NAMES):
        rtpt.step(subtitle=name)
        others = tuple(i for i in range(len(CARDINALITIES)) if i != factor)
        modelled = joint.sum(dim=others)
        counts = torch.bincount(
            labels[:, factor], minlength=CARDINALITIES[factor]
        ).float()
        empirical = counts / counts.sum()

        sampled_means, real_means = [], []
        for value in range(CARDINALITIES[factor]):
            with torch.no_grad():
                latents, _ = model.sample_partial_labels(
                    {factor: value}, SAMPLES_PER_VALUE, device=device
                )
            sampled_means.append(latents.cpu().mean(dim=0))

            rows_for_value = real_labels[:, factor] == value
            if int(rows_for_value.sum()) >= 8:
                real_means.append(real_latents[rows_for_value].mean(dim=0))

        model_spread = spread_of_means(torch.stack(sampled_means))
        data_spread = spread_of_means(torch.stack(real_means))
        rows.append(
            {
                "factor": name,
                "label_tv": float(0.5 * (modelled - empirical).abs().sum()),
                "model_spread": model_spread,
                "data_spread": data_spread,
                "ratio": model_spread / data_spread if data_spread else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg, _, _ = load_config()
    assert isinstance(cfg, EvaluationRunConfig)
    device = resolve_device()

    model, resolved, model_path = load_generative_model(
        cfg.model.model_type, cfg.model.name, device, cfg.model.tag
    )
    if not isinstance(model, JointPC):
        raise SystemExit(f"{resolved} is a {type(model).__name__}, not a JointPC")
    name, tag, external = resolve_autoencoder(cfg.autoencoder, model_path, resolved)
    vae, resolved_vae = load_vae(name, tag, external, cfg.dataset, device)

    labels = training_labels(cfg.dataset.name)
    _, val_loader = build_data_loaders(cfg.dataset, batch_size=256, drop_last=False)
    real_latents, _, _, real_labels = encode_and_decode(vae, val_loader, device)

    report = factor_report(model, real_latents, real_labels, device, labels)

    # Written out as well as printed: the ratio is the headline of this diagnosis and
    # belongs in a figure, not only in a terminal.
    columns = {
        "model": cfg.model.model_type,
        "dataset": cfg.dataset.name,
        "checkpoint": resolved,
        "seed": cfg.generation.seed,
        "std_correction": cfg.generation.std_correction,
        "vae": resolved_vae,
    }
    write_metric(
        cfg.evaluation.results_root / FILENAME,
        report.assign(**columns)[[*columns, *report.columns]],
        {key: columns[key] for key in RUN_KEYS},
    )

    print(f"\nmodel    {resolved}    decoder {resolved_vae}")
    print("\nper factor:")
    print(report.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "\n  label_tv      circuit's p(y_f) vs the training labels (0 is perfect)\n"
        "  model_spread  how far apart p(z | factor=v) means are, over v\n"
        "  data_spread   the same on real latents -- the target\n"
        "  ratio         near 1.0: the factor steers the latents like the data does\n"
        "                near 0.0: the label is inert, the circuit ignores it"
    )


if __name__ == "__main__":
    main()
