"""One module per metric.

Each exports `FILENAME`, the CSV it writes, and
`compute(images, predictions, labels, num_classes) -> pd.DataFrame`, its own table.
`images` are float in [0, 1]; every metric takes the same arguments and uses what it
needs. Adding a metric is a new module plus one entry in `METRICS`; `evaluate.py` loops
over the list and needs no other change.
"""

from types import ModuleType

from evaluation.metrics import (
    colour_accuracy,
    colour_accuracy_by_combination,
    colour_contrast,
    colour_drift,
    confusion_digit,
    digit_accuracy,
    digit_accuracy_by_combination,
)

METRICS = [
    digit_accuracy,
    digit_accuracy_by_combination,
    confusion_digit,
    colour_accuracy,
    colour_accuracy_by_combination,
    colour_drift,
    colour_contrast,
]

METRICS_BY_NAME = {module.__name__.rsplit(".", 1)[-1]: module for module in METRICS}


def selected(names: list[str] | None) -> list[ModuleType]:
    """The metrics `names` asks for, in `METRICS` order; all of them when `None`."""
    if names is None:
        return METRICS
    return [METRICS_BY_NAME[name] for name in names]


__all__ = [
    "METRICS",
    "METRICS_BY_NAME",
    "colour_accuracy",
    "colour_accuracy_by_combination",
    "colour_contrast",
    "colour_drift",
    "confusion_digit",
    "digit_accuracy",
    "digit_accuracy_by_combination",
    "selected",
]
