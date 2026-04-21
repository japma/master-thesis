FROM docker.io/nvidia/cuda:12.8.1-base-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_LINK_MODE=copy \
	UV_COMPILE_BYTECODE=1 \
	UV_PYTHON_DOWNLOADS=automatic

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project
RUN uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 --upgrade torch torchvision

COPY . .

RUN mkdir -p /app/data /app/results /app/outputs
VOLUME ["/app/data", "/app/results", "/app/outputs"]

CMD [".venv/bin/python", "main.py"]
