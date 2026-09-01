"""torch.compile wiring."""

import torch
from torch import nn


def maybe_compile(
    model: nn.Module, enabled: bool, mode: str = "default"
) -> nn.Module:
    """Compile `model` when enabled, otherwise hand it back untouched."""
    if not enabled:
        return model
    print(f"Compiling {type(model).__name__} with torch.compile(mode={mode!r})")
    return torch.compile(model, mode=mode)


def uncompiled(model: nn.Module) -> nn.Module:
    """The original module behind a `torch.compile` wrapper, or `model` unchanged."""
    return getattr(model, "_orig_mod", model)
