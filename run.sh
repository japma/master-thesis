#!/bin/bash
podman run --rm --env-file .env --device nvidia.com/gpu=all -v ./:/app localhost/master-thesis-jm:latest "$@"
