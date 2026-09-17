"""Entry point for digit classifier training."""

import torch
from rtpt import RTPT
from torchinfo import summary

import wandb
from dataset_loaders import build_data_loaders
from models.classifier import DigitClassifier
from training.early_stopping import EarlyStopping
from training.loop import CheckpointSpec, run_training_loop
from training.metrics import PerClassAccuracy
from training.objectives.classifier import DIGIT_FACTOR, ClassifierObjective
from utils.checkpoints import (
    final_checkpoint_path,
    intermediate_checkpoint_path,
    load_classifier_from_path,
)
from utils.compilation import maybe_compile
from utils.config import ClassifierRunConfig, load_config
from utils.reproducibility import resolve_device, seed_everything
from utils.wandb_utils import init_run, log_summary

# Below this the judge understates every model it scores, and a digit metric computed
# with it cannot be told apart from a genuine failure of the model under test.
MIN_JUDGE_ACCURACY = 0.95


def main() -> None:
    cfg, cfg_seed, resume = load_config()
    assert isinstance(cfg, ClassifierRunConfig)
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    training_cfg = cfg.training

    seed = seed_everything(cfg_seed)
    device = resolve_device()
    dataset_name = dataset_cfg.name

    run_name = f"digit_classifier_{dataset_name}"
    init_run(cfg.wandb, run_name, cfg.model_dump())

    print(
        f"Training digit classifier on {dataset_name} | device={device} | seed={seed}"
    )

    ckpt_path = intermediate_checkpoint_path("digit_classifier", dataset_name)
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

    early_stopping = EarlyStopping(
        patience=training_cfg.early_stopping_patience,
        min_delta=training_cfg.early_stopping_min_delta,
    )
    print(
        f"Early stopping on val error_rate: patience "
        f"{training_cfg.early_stopping_patience}, min_delta "
        f"{training_cfg.early_stopping_min_delta}"
    )

    checkpoint = CheckpointSpec(
        intermediate_path=ckpt_path,
        final_path=final_checkpoint_path("digit_classifier", dataset_name),
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
        early_stopping=early_stopping,
        early_stopping_metric="error_rate",
    )

    _report_judge_quality(model, test_loader, device, model_cfg.num_classes)

    wandb.finish()


@torch.no_grad()
def _report_judge_quality(
    model: DigitClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> None:
    """Scores the weights actually saved -- early stopping restores an earlier epoch's,
    so the last epoch's validation numbers are not necessarily the judge's."""
    model.eval()
    accuracy = PerClassAccuracy(num_classes)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        digits = labels[:, DIGIT_FACTOR].to(device, non_blocking=True).long()
        accuracy.update(model(images).argmax(dim=1), digits)

    per_digit = accuracy.per_class
    print(f"\nFinal judge accuracy: {accuracy.overall:.4f}")
    for digit, value in enumerate(per_digit):
        print(f"  digit {digit}: {value:.4f}")

    log_summary(
        {"judge/accuracy": accuracy.overall}
        | {
            f"judge/digit_accuracy/{digit}": float(value)
            for digit, value in enumerate(per_digit)
        }
    )

    if accuracy.overall < MIN_JUDGE_ACCURACY:
        print(
            f"WARNING: accuracy {accuracy.overall:.3f} is below {MIN_JUDGE_ACCURACY} "
            "-- digit metrics computed with this judge will understate every model."
        )
    worst = int(per_digit.nan_to_num(1.0).argmin())
    if float(per_digit[worst]) < MIN_JUDGE_ACCURACY:
        print(
            f"WARNING: digit {worst} is at {float(per_digit[worst]):.3f}. A judge blind "
            "to one digit scores every model unfairly on that digit alone."
        )


if __name__ == "__main__":
    main()
