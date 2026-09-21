"""A torch.compile wrapper is not an instance of what it wraps.

`train_ae` asserts the autoencoder's type *after* compiling it, so this is the seam
where `--compile` silently changes behaviour -- and nothing else in the suite runs the
trainers with it on.
"""

import torch
from torch import nn

from utils.compilation import maybe_compile, uncompiled


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_compiling_is_a_no_op_when_disabled() -> None:
    model = Tiny()
    assert maybe_compile(model, enabled=False) is model


def test_a_compiled_model_fails_a_naive_isinstance_check() -> None:
    """The reason `uncompiled` has to exist; if this ever stops holding, the guards
    around it are dead weight."""
    compiled = maybe_compile(Tiny(), enabled=True)
    assert not isinstance(compiled, Tiny)


def test_uncompiled_recovers_the_original_type() -> None:
    model = Tiny()
    compiled = maybe_compile(model, enabled=True)

    assert isinstance(uncompiled(compiled), Tiny)
    assert uncompiled(compiled) is model
    # Safe on something that was never compiled.
    assert uncompiled(model) is model


def test_the_wrapper_still_proxies_attributes() -> None:
    """Which is why the objective can be handed the wrapper rather than the original."""
    compiled = maybe_compile(Tiny(), enabled=True)
    assert isinstance(compiled.linear, nn.Linear)
