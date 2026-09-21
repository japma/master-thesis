"""torch.compile wiring."""

import torch
from torch import nn


def maybe_compile[M: nn.Module](model: M, enabled: bool, mode: str = "default") -> M:
    """Compile `model` when enabled, otherwise hand it back untouched.

    Typed as returning the model it was given: at runtime this is a `torch.compile`
    wrapper that proxies to the original, so every attribute and method still resolves.
    `isinstance` does not -- use `uncompiled` for that.
    """
    if not enabled:
        return model
    print(f"Compiling {type(model).__name__} with torch.compile(mode={mode!r})")
    return torch.compile(model, mode=mode)  # type: ignore[return-value]


def uncompiled[M: nn.Module](model: M) -> M:
    """The original module behind a `torch.compile` wrapper, or `model` unchanged.

    The one thing the wrapper does not proxy is its own type, so anything doing an
    `isinstance` check has to come through here first.
    """
    return getattr(model, "_orig_mod", model)
