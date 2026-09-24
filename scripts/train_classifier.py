"""Entry point for training the colour-MNIST judge (digit, fg and bg heads)."""

import torch
from rtpt import RTPT
from torchinfo import summary

import wandb
from dataset_loaders import build_data_loaders
from models.classifier import DigitClassifier
from training.loop import CheckpointSpec, run_training_loop
from training.metrics import PerClassAccuracy
from training.objectives.classifier import ClassifierObjective
from utils.checkpoints import (
    final_checkpoint_path,
    intermediate_checkpoint_path,
    load_classifier_from_path,
)
from utils.compilation import maybe_compile, uncompiled
from utils.config import ClassifierRunConfig, load_config
from utils.naming import ModelFamily, artifact_name
from utils.reproducibility import resolve_device, seed_everything
from utils.wandb_utils import init_run, log_summary

# Below this the judge understates every model it scores, and a metric computed with it
# cannot be told apart from a genuine failure of the model under test.
MIN_JUDGE_ACCURACY = 0.95


def main() -> None:
    cfg, cfg_seed, resume = load_config()
    assert isinstance(cfg, ClassifierRunConfig)
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    training_cfg = cfg.training

    seed = seed_everything(cfg_seed)
    device = resolve_device()
    dataset_name = dataset_cfg.artifact_name

    run_name = f"digit_classifier_{dataset_name}"
    init_run(cfg.wandb, run_name, cfg.model_dump())

    print(
        f"Training digit classifier on {dataset_name} | device={device} | seed={seed}"
    )

    name = artifact_name(ModelFamily.DIGIT_CLASSIFIER, dataset_cfg)
    ckpt_path = intermediate_checkpoint_path(name)
    if resume and ckpt_path.exists():
        model = load_classifier_from_path(ckpt_path, device=device).to(device)
        print(f"Resumed model weights from {ckpt_path}")
    else:
        if resume:
            print(
                f"--resume given but no checkpoint found at {ckpt_path}; "
                "starting from scratch"
            )
        model = DigitClassifier(config=model_cfg).to(device)

    print("Digit classifier architecture:")
    summary(model)

    train_loader, test_loader = build_data_loaders(
        dataset_cfg, batch_size=training_cfg.batch_size, drop_last=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_cfg.epochs
    )

    model = maybe_compile(model, training_cfg.compile, training_cfg.compile_mode)

    objective = ClassifierObjective(
        model=model, optimizer=optimizer, lr_scheduler=scheduler
    )

    rtpt = RTPT(
        name_initials="JM",
        experiment_name=run_name,
        max_iterations=max(training_cfg.epochs, 1),
    )
    rtpt.start()

    checkpoint = CheckpointSpec(
        intermediate_path=ckpt_path,
        final_path=final_checkpoint_path(name),
        artifact_type="classifier",
    )

    run_training_loop(
        objective=objective,
        device=device,
        epochs=training_cfg.epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        rtpt=rtpt,
        checkpoint=checkpoint,
        resume=resume,
    )

    _report_judge_quality(model, test_loader, device)

    wandb.finish()


@torch.no_grad()
def _report_judge_quality(
    model: DigitClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """Scores the saved weights over the whole validation set, per label factor."""
    model.eval()
    config = uncompiled(model).config
    accuracies = [PerClassAccuracy(c) for c in config.cardinalities]
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        for i, logits in enumerate(model(images)):
            accuracies[i].update(logits.argmax(dim=1), labels[:, i])

    summary_metrics = {}
    for name, accuracy in zip(config.names, accuracies, strict=True):
        per_class = accuracy.per_class
        print(f"\nFinal judge {name} accuracy: {accuracy.overall:.4f}")
        for c, value in enumerate(per_class):
            print(f"  {name} {c}: {value:.4f}")

        summary_metrics[f"judge/{name}_accuracy"] = accuracy.overall
        summary_metrics |= {
            f"judge/{name}_accuracy/{c}": float(value)
            for c, value in enumerate(per_class)
        }

        if accuracy.overall < MIN_JUDGE_ACCURACY:
            print(
                f"WARNING: {name} accuracy {accuracy.overall:.3f} is below "
                f"{MIN_JUDGE_ACCURACY} -- {name} metrics computed with this judge will "
                "understate every model."
            )
        worst = int(per_class.nan_to_num(1.0).argmin())
        if float(per_class[worst]) < MIN_JUDGE_ACCURACY:
            print(
                f"WARNING: {name} {worst} is at {float(per_class[worst]):.3f}. A judge "
                f"blind to one class scores every model unfairly on that {name} alone."
            )
    log_summary(summary_metrics)


if __name__ == "__main__":
    main()
