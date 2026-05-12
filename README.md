# Master Thesis

## uv version
`uv` needs to be installed. Then run
```bash
uv sync
uv run train_autoencoder.py
```

To train the CSPN, run `uv run train_cspn.py`. For inference, run `uv run inference.py`.

## Docker
Run
```bash
docker-compose up -d
```

## Outputs
Hydra creates a unique run directory for each execution. Training checkpoints
are written to `<hydra run dir>/checkpoints/` and generated visualizations are
written to `<hydra run dir>/images/`.

## Configuration
Configuration is done via `hydra`.
It is possible to e.g. change the dataset by passing `data=mnist` to the task entrypoint directly, or by passing it to Docker via `HYDRA_ARGS="data=mnist"`

