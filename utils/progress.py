"""RTPT wiring for the evaluation entrypoints.

The GPU box is shared, so anything that runs for a while advertises whose it is and how
much longer it needs. The trainers build their own RTPT from an epoch count; evaluation
only learns its work after loading a pool, so the helper is called from inside the run
functions rather than from the thin scripts.
"""

import math

from rtpt import RTPT

NAME_INITIALS = "JM"


def start_rtpt(experiment_name: str, max_iterations: int) -> RTPT:
    """A started RTPT titled `experiment_name`, stepping towards `max_iterations`."""
    rtpt = RTPT(
        name_initials=NAME_INITIALS,
        experiment_name=experiment_name,
        max_iterations=max(max_iterations, 1),
    )
    rtpt.start()
    return rtpt


def batch_count(n: int, batch_size: int) -> int:
    """Batches `n` items make, for an RTPT budget that matches the loops below."""
    return math.ceil(n / batch_size) if batch_size > 0 else 1
