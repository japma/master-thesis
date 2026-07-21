import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, root: Path, split: str, transform: Callable | None = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.coco = COCO(ann_file)

        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(self.cat_ids)}

        self.img_ids = [
            img_id
            for img_id in self.coco.getImgIds()
            if (self.root / self.coco.loadImgs(img_id)[0]["file_name"]).exists()
        ]

    @property
    def num_classes(self) -> int:
        return len(self.cat_ids)

    def _primary_label(self, img_id: int) -> int:
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        if not anns:
            return 0
        counts = Counter(a["category_id"] for a in anns)
        most_common_cat_id = counts.most_common(1)[0][0]
        return self.cat_id_to_idx[most_common_cat_id]

    def __len__(self) -> int:
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
    def __init__(self, root, split: str="train", transform=None) -> None:
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
    def num_classes(self) -> int:
        return 80

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        tensor = torch.load(path, weights_only=True)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label
