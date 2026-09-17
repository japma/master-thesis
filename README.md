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

### Evaluation

Two stages that communicate through `eval_runs/`. Stage 1 writes latents, labels and a
manifest per run; stage 2 decodes them and upserts `results/metrics.csv`, one row per
`(model, dataset, checkpoint, seed, metric_name, value)`. New metrics never need new samples.

```bash
# 1. the reference: real test images encoded through the VAE
uv run generate_run real --variant uniform --split test \
    --vae variational_colour_mnist_uniform:v1

# 2. a model, sampled for exactly the reference's labels
uv run generate_run model --model-type cspn --checkpoint psinet_colour_mnist_uniform:v1 \
    --model-name cspn_std0.6 --std-correction 0.6 --seed 0 \
    --vae variational_colour_mnist_uniform:v1 \
    --reference eval_runs/colour_mnist_uniform_test__real__seed0

# 3. metrics
uv run evaluate_run --run eval_runs/colour_mnist_uniform_test__cspn_std0.6__seed0 \
    --reference eval_runs/colour_mnist_uniform_test__real__seed0 --metrics fid
```

Checkpoints are always wandb artifacts (`name` or `name:version`); manifests record the
exact version they resolved to. FID uses torchmetrics 1.9.0 with
torch-fidelity 0.4.0's `inception-v3-compat` weights
(`weights-inception-2015-12-05-6726825d.pth`), downloaded to `$TORCH_HOME` on first use.

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

