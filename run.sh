#!/bin/bash
podman run --rm --env-file .env --shm-size=8g --device nvidia.com/gpu=all -v ./:/app localhost/master-thesis-jm:latest "$@"
