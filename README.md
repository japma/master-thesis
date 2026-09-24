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

Two stages around one **pool per dataset**, both driven by one config file per dataset
(`configs/pools/<dataset>.yaml`). Stage 1 samples every listed model, decodes through
the autoencoder its checkpoint was trained against, and caches the images; stage 2
scores them and writes one CSV per metric. No metric is computed in stage 1, and stage 2
never loads a VAE.

```bash
uv run generate_pools   configs/pools/colour_mnist_skewed.yaml   # stage 1
uv run evaluate_samples configs/pools/colour_mnist_skewed.yaml   # judged metrics (colour-MNIST)
uv run evaluate_sets    configs/pools/colour_mnist_skewed.yaml   # FID, KID, precision/recall, CMMD
```

```
results/pools/<dataset>/
  real/                                 the val split: images + labels
  stratified_<n>/                       conditioning labels (colour-MNIST)
  vaes/<vae>/<vN>/                      real/ round-tripped through that VAE
  seed<s>/<type>/<model>/<vN>/std<x>/   the model's samples
```

`real/` and `vaes/` hold nothing random, so they are shared by every model and seed.
Models are conditioned either on the stratified schedule (every colour-MNIST cell,
held-out ones included) or on `real/`'s own labels (CelebA: sample i on image i).

```yaml
type: pools
dataset: { name: colour_mnist_skewed }
models:
  - { type: cspn, name: psinet_colour_mnist_skewed }                 # latest, resolved to vN
  - { type: cspn, name: psinet_colour_mnist_skewed_anchored, version: v3 }  # pinned
  - { type: joint_pc, name: joint_pc_colour_mnist_skewed_labels-digit, labels: [digit] }
classifier: { name: digit_classifier_colour_mnist_uniform, tag: latest }
generation: { labels: stratified, n_per_cell: 100, seeds: [0] }
evaluation: { batch_size: 512, results_root: results }
```

`type` is one of `cspn`, `joint_pc`, `nn_baseline` -- they all expose the same
`sample(labels, std_correction)`. `latest` is always resolved to the exact `vN` before
anything is written, and every directory is written under a temporary name and renamed
when complete. Re-running skips everything already generated, samples a newly trained
version beside the old one, and ends with a summary; a listed model wandb does not have
is reported as missing, never fatal. Stage 2 scores each model's newest generated
version, or the one it pins.

The decoder is always the autoencoder the model checkpoint recorded being trained
against (`read_source_artifact`, falling back to wandb lineage), so the latents always
fit it. The sampled latent width is checked against the decoder before anything is
decoded, so a mismatched pair fails naming both artifacts.

Stage 2 judges every set of images in the pool once: the real validation images, their
round trip through each VAE a listed model uses, and each model's generated samples. The
real and reconstruction rows are the ceilings; generated accuracy is not readable
without them. A model conditioned on a subset of the factors is scored on that subset.

Stage 2 writes **one CSV per metric** under `results_root`, accumulating across runs, so
a thesis figure is one `read_csv`:

```
results/accuracy.csv                  <set>,factor,value,n
results/accuracy_by_combination.csv   <set>,factor,digit,fg,bg,value,n
results/confusion.csv                 <set>,factor,truth,predicted,n
```

`<set>` is the same identity prefix everywhere -- model, dataset, checkpoint, seed,
std_correction, vae, classifier, source -- so every row is self-describing, and
re-scoring a set replaces its rows instead of duplicating them. `source` is `generated`,
`real` or `reconstruction`; a real row leaves the model's columns and `vae` empty, a
reconstruction row the model's columns. The metrics are plain functions in
`evaluation/metrics.py`, named in its `METRICS`.

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

## Podman

Run

```bash
./run.sh
```
