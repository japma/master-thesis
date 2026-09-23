import pandas as pd
import pytest
from PIL import Image

from dataset_loaders import build_dataset
from dataset_loaders.colour_mnist import (
    LABELS_FILENAME,
    ColourMNIST,
    label_columns,
    variant_root,
)
from utils.config import DatasetConfig


@pytest.fixture
def root(tmp_path):
    split_dir = variant_root(tmp_path, "tiny") / "train"
    (split_dir / "images").mkdir(parents=True)
    Image.new("RGB", (28, 28)).save(split_dir / "images" / "000000.png")
    pd.DataFrame(
        {
            "label": [7],
            "fg_colour": ["blue"],
            "bg_colour": ["grey"],
            "filename": ["000000.png"],
        }
    ).to_csv(split_dir / LABELS_FILENAME, index=False)
    return tmp_path


def test_default_returns_every_factor(root):
    _, target = ColourMNIST(root, variant="tiny")[0]
    assert target.tolist() == [7, 2, 2]


@pytest.mark.parametrize(
    ("labels", "expected"),
    [(["digit"], [7]), (["fg"], [2]), (["digit", "bg"], [7, 2])],
)
def test_selection_keeps_named_columns_and_full_targets(root, labels, expected):
    data = ColourMNIST(root, variant="tiny", labels=labels)
    _, target = data[0]
    assert target.tolist() == expected
    assert data.targets.shape == (1, 3)


@pytest.mark.parametrize(
    "labels", [[], ["colour"], ["fg", "digit"], ["digit", "digit"]]
)
def test_invalid_selection_is_rejected(labels):
    with pytest.raises(ValueError):
        label_columns(labels)


def test_selection_changes_artifact_name():
    base = {
        "name": "colour_mnist_uniform",
        "channels": 3,
        "height": 28,
        "width": 28,
        "num_classes": 10,
    }
    assert DatasetConfig(**base).artifact_name == "colour_mnist_uniform"
    assert (
        DatasetConfig(**base, labels=("digit", "fg")).artifact_name
        == "colour_mnist_uniform_labels-digit-fg"
    )


def test_selection_on_other_datasets_is_rejected():
    with pytest.raises(ValueError, match="no label factors"):
        build_dataset("mnist", labels=["digit"])
