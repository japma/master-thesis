"""COCO dataset with primary category label for CSPN conditioning."""

import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    """COCO 2017 dataset returning (image, primary_category_idx).

    The label is the index (0–79) of the most-annotated category in the image,
    suitable as a simple conditioning signal for the CSPN. Images with no
    annotations fall back to label 0.

    Args:
        root:      Path to the image directory (train/ or val/).
        ann_file:  Path to the instances annotation JSON.
        transform: torchvision transform applied to the PIL image.
    """

    def __init__(self, root, ann_file, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.coco = COCO(ann_file)

        # Build a stable sorted list of 80 category ids → 0-based index
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(self.cat_ids)}

        # Only keep images that actually exist on disk
        self.img_ids = [
            img_id
            for img_id in self.coco.getImgIds()
            if (self.root / self.coco.loadImgs(img_id)[0]["file_name"]).exists()
        ]

    @property
    def num_classes(self):
        return len(self.cat_ids)

    def _primary_label(self, img_id: int) -> int:
        """Return the index of the most frequent category in the image."""
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        if not anns:
            return 0
        counts = Counter(a["category_id"] for a in anns)
        most_common_cat_id = counts.most_common(1)[0][0]
        return self.cat_id_to_idx[most_common_cat_id]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_meta = self.coco.loadImgs(img_id)[0]
        image = Image.open(self.root / img_meta["file_name"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self._primary_label(img_id)
        return image, label


class CocoCachedDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        meta_file = self.root / f"meta_{split}.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        with meta_file.open() as f:
            meta = json.load(f)

        self.samples = [
            (self.root / split / v["file"], v["label"]) for v in meta.values()
        ]

    @property
    def num_classes(self):
        return 80

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        tensor = torch.load(path, weights_only=True)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label
