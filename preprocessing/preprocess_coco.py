"""
Preprocess COCO dataset to cached PyTorch tensors.

Reads all COCO images from ./data/coco-2017/, resizes to 128x128,
and saves as float32 tensors with category labels.
"""

import json
import os
from collections import Counter
from multiprocessing import Pool, cpu_count

import torch
import torchvision.transforms.functional as F
from PIL import Image
from pycocotools.coco import COCO
from rtpt import RTPT
from tqdm import tqdm


DATA_DIR = "./data/coco-2017"
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train")
VAL_IMAGES_DIR = os.path.join(DATA_DIR, "val")
TRAIN_ANNOTATIONS = os.path.join(DATA_DIR, "annotations/instances_train2017.json")
VAL_ANNOTATIONS = os.path.join(DATA_DIR, "annotations/instances_val2017.json")
OUTPUT_DIR = "./data/coco-cached"
TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "val")
TRAIN_META_JSON_PATH = os.path.join(OUTPUT_DIR, "meta_train.json")
VAL_META_JSON_PATH = os.path.join(OUTPUT_DIR, "meta_val.json")

TARGET_SIZE = 128
NUM_WORKERS = max(1, cpu_count() - 2)


def _process_image(args):
    image_id, image_path, output_dir = args

    output_path = os.path.join(output_dir, f"{image_id}.pt")
    if os.path.exists(output_path):
        return image_id

    try:
        img = Image.open(image_path).convert("RGB")
        tensor = F.to_tensor(img)
        tensor_resized = F.resize(tensor, [TARGET_SIZE, TARGET_SIZE])
        torch.save(tensor_resized.float(), output_path)
        return image_id
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        return None


def load_annotations_and_build_label_map():
    print("Loading COCO annotations...")

    image_to_label = {}
    all_category_ids = set()
    image_annotations = {}

    if os.path.exists(TRAIN_ANNOTATIONS):
        coco_train = COCO(TRAIN_ANNOTATIONS)
        for img_id in coco_train.getImgIds():
            image_annotations[img_id] = []

        for ann in coco_train.dataset["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            all_category_ids.add(category_id)
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(category_id)

    if os.path.exists(VAL_ANNOTATIONS):
        coco_val = COCO(VAL_ANNOTATIONS)
        for img_id in coco_val.getImgIds():
            if img_id not in image_annotations:
                image_annotations[img_id] = []

        for ann in coco_val.dataset["annotations"]:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            all_category_ids.add(category_id)
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(category_id)

    sorted_category_ids = sorted(list(all_category_ids))
    category_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted_category_ids)}

    for image_id, cat_ids in image_annotations.items():
        if cat_ids:
            counter = Counter(cat_ids)
            primary_cat_id = counter.most_common(1)[0][0]
            image_to_label[image_id] = category_to_idx[primary_cat_id]
        else:
            image_to_label[image_id] = 0

    print(
        f"Loaded {len(image_to_label)} images with {len(sorted_category_ids)} categories"
    )

    return image_to_label


def get_all_images():
    images = []

    # training images
    if os.path.exists(TRAIN_IMAGES_DIR):
        for filename in os.listdir(TRAIN_IMAGES_DIR):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                try:
                    image_id = int(filename.split(".")[0])
                    image_path = os.path.join(TRAIN_IMAGES_DIR, filename)
                    images.append((image_id, image_path, "train"))
                except ValueError:
                    continue

    # validation images
    if os.path.exists(VAL_IMAGES_DIR):
        for filename in os.listdir(VAL_IMAGES_DIR):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                try:
                    image_id = int(filename.split(".")[0])
                    image_path = os.path.join(VAL_IMAGES_DIR, filename)
                    images.append((image_id, image_path, "val"))
                except ValueError:
                    continue

    return images


def main():
    rtpt = RTPT(
        name_initials="JM",
        experiment_name="COCO Preprocessing",
        max_iterations=1,
    )
    rtpt.start()

    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(VAL_OUTPUT_DIR, exist_ok=True)

    image_to_label = load_annotations_and_build_label_map()

    images = get_all_images()
    print(f"Found {len(images)} images total")

    already_cached = sum(
        1
        for image_id, _, split in images
        if os.path.exists(
            os.path.join(
                TRAIN_OUTPUT_DIR if split == "train" else VAL_OUTPUT_DIR,
                f"{image_id}.pt",
            )
        )
    )
    if already_cached > 0:
        print(f"{already_cached} images already cached, skipping...")

    tasks = [
        (image_id, image_path, TRAIN_OUTPUT_DIR if split == "train" else VAL_OUTPUT_DIR)
        for image_id, image_path, split in images
    ]

    print(f"Processing images with {NUM_WORKERS} workers...")

    rtpt.max_iterations = len(tasks)

    with Pool(NUM_WORKERS) as pool:
        results = []
        for idx, result in enumerate(
            tqdm(
                pool.imap_unordered(_process_image, tasks),
                total=len(tasks),
                desc="Processing images",
            )
        ):
            results.append(result)
            # Update progress every 100 images or at the end
            if (idx + 1) % 100 == 0 or (idx + 1) == len(tasks):
                rtpt.step(subtitle=f"Processed {idx + 1}/{len(tasks)} images")

    processed = sum(1 for r in results if r is not None)
    print(f"\nSuccessfully processed {processed} images")

    train_meta = {}
    val_meta = {}

    for image_id, _, split in images:
        output_dir = TRAIN_OUTPUT_DIR if split == "train" else VAL_OUTPUT_DIR
        tensor_path = os.path.join(output_dir, f"{image_id}.pt")
        if os.path.exists(tensor_path):
            label = image_to_label.get(image_id, 0)
            metadata_entry = {"file": f"{image_id}.pt", "label": label}

            if split == "train":
                train_meta[str(image_id)] = metadata_entry
            else:
                val_meta[str(image_id)] = metadata_entry

    with open(TRAIN_META_JSON_PATH, "w") as f:
        json.dump(train_meta, f, indent=2)

    with open(VAL_META_JSON_PATH, "w") as f:
        json.dump(val_meta, f, indent=2)

    print(f"Train images saved: {len(train_meta)}")
    print(f"Val images saved: {len(val_meta)}")
    print(f"Train metadata saved to: {TRAIN_META_JSON_PATH}")
    print(f"Val metadata saved to: {VAL_META_JSON_PATH}")


if __name__ == "__main__":
    main()
