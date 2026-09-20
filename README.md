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

Two stages that communicate through a sample pool directory, both driven by one config
file. Stage 1 samples p(z | y), decodes through the VAE and caches the images; stage 2
scores them with a frozen classifier and writes one CSV per metric. No metric is computed
in stage 1, and stage 2 never loads the VAE.

```bash
uv run generate_samples configs/evaluation/colour_mnist_uniform.yaml
uv run evaluate_samples configs/evaluation/colour_mnist_uniform.yaml
```

Both stages read the same config, and the pool location is derived from it
(`<output_root>/<dataset>__<model>__seed<N>`), so no path is pasted between them. `--seed`
overrides `generation.seed` for both, which moves the pool together.

```yaml
type: evaluation
dataset:      { name: colour_mnist_uniform }
model:        { name: psinet_colour_mnist_uniform, model_type: cspn, tag: latest }
classifier:   { name: digit_classifier_colour_mnist_uniform, tag: latest }
generation:   { n_per_cell: 100, seed: 0, std_correction: 1.0, output_root: results/samples }
evaluation:   { batch_size: 512, results_root: results }
```

`model_type` is one of `cspn`, `joint_pc`, `nn_baseline` -- they all expose the same
`sample(labels, std_correction)`. Unknown keys are rejected, so a typo fails at load
rather than silently taking a default.

There is no `autoencoder:` entry by default: the decoder is whichever autoencoder the
model checkpoint recorded being trained against (`read_source_artifact`, falling back to
wandb lineage), so the latents always fit the decoder. Add
`autoencoder: { name: ..., external: false, tag: latest }` only to override that. Either
way the sampled latent width is checked against the decoder before anything is decoded,
so a mismatched pair fails naming both artifacts.

Stage 1 also writes a nested `reference/` pool -- the validation split encoded and
decoded through the same VAE -- so stage 2 scores three sets of images: the generated
samples, the real validation images, and their VAE round trip. The last two are the
ceilings; generated accuracy is not readable without them.

Stage 2 writes **one CSV per metric** under `results_root`, accumulating across runs, so
a thesis figure is one `read_csv`:

```
results/digit_accuracy.csv                 <run>,source,value,n
results/digit_accuracy_by_combination.csv  <run>,source,digit,fg,bg,value,n
results/confusion_digit.csv                <run>,source,truth,predicted,n
```

`<run>` is the same identity prefix everywhere -- model, dataset, checkpoint, seed,
std_correction, vae, classifier -- so every row is self-describing, and re-evaluating a
run replaces its rows instead of duplicating them. `source` is `generated`, `real` or
`reconstruction`, so the ceilings are rows rather than suffixes on metric names. Only the
digit is judged for now; adding fg/bg means more metric functions and more CSVs, not a
change to what exists.

Checkpoints are always wandb artifacts (`name` or `name:version`); manifests record the
exact version they resolved to.

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

