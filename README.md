# Master Thesis

## uv version
`uv` needs to be installed. Then run 
```bash
uv sync
uv run main.py
```


## Docker
Run 
```bash
docker-compose up -d
```

## Outputs
Training outputs are written under `results/<dataset>_<timestamp>/` with separate
`checkpoints/` and `images/` folders.

## Configuration
Configuration is done via `hydra`.
It is possible to e.g. change the dataset by passing `data=mnist` either to `main.py` directly, or by passing them to Docker via `HYDRA_ARGS="data=mnist"`
