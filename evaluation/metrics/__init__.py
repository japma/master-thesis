"""The metrics the harness fans batches out to.

Grouped by the claim each supports, not by the model that produced the evidence:

  conditioning   does it obey the label
  distribution   is the spread right
  density        what does it say about real data
  colour_mnist   this dataset's own probes, kept separate on purpose

Each is independent of every other and of any particular model -- they see an
`EvalBatch`, not a circuit -- so a new one is a class here and a line in a suite.
"""

from evaluation.metrics.colour_mnist import ColourFidelity
from evaluation.metrics.conditioning import DigitAccuracy, LabelDiscrimination
from evaluation.metrics.density import NegativeLogLikelihood
from evaluation.metrics.distribution import LatentPlausibility, SampleDiversity

__all__ = [
    "ColourFidelity",
    "DigitAccuracy",
    "LabelDiscrimination",
    "LatentPlausibility",
    "NegativeLogLikelihood",
    "SampleDiversity",
]
