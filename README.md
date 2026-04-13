# Master Thesis

## Prerequisites
- Install `uv`

## Getting started

```bash
uv sync
```

## Usage
```bash
uv run main.py
```

Training outputs are written under `results/<dataset>_<timestamp>/` with separate
`checkpoints/` and `images/` folders.

Changing the dataset to e.g. FashionMNIST can be done with
```bash
uv run main.py data=fashionmnist
```
