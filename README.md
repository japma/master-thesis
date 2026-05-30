# Master Thesis

## Setup

`uv` needs to be installed. Then run

```bash
uv sync
source .venv/bin/activate
```

## Usage

### Training Autoencoder

```bash
python train_ae.py
```

To train on different datasets or configurations, edit `configs/config.yaml` or create a new config file.

### Training CSPN

```bash
# Requires a pretrained autoencoder checkpoint
python train_cspn.py
```

### Running Inference & Visualization

```bash
python visualize.py
```

## Configuration

All configuration is handled through YAML files located in `configs/`:
- `configs/config.yaml` – main config
- `configs/dataset/*.yaml` – dataset configurations
- `configs/training/*.yaml` – training parameter sets
- `configs/autoencoder/*.yaml` – AE architecture presets
- `configs/cspn/*.yaml` – CSPN architecture presets

### Modifying Configs

Edit the `defaults` section in `config.yaml` to compose different configurations:

```yaml
defaults:
  - dataset: mnist
  - training: default
  - _self_

wandb_mode: "offline"
seed: 42
```

To use a different dataset, training setup, or model architecture, modify the corresponding YAML file or the defaults list.

## Podman

Run

```bash
./run.sh
```

## Legacy Hydra Usage (Deprecated)

The old `main.py` with Hydra is deprecated. Use the new split entrypoints instead.

