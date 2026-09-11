"""Autoencoder model, training, and run configs."""

from enum import StrEnum
from itertools import accumulate, pairwise
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from utils.config.common import (
    BaseTrainingConfig,
    DatasetConfig,
    WandbConfig,
)


class AutoencoderType(StrEnum):
    VARIATIONAL = "variational"
    SUPERVISED = "supervised"
    ANCHORED = "anchored"
    OTHER = "other"


class AnchorScheme(StrEnum):
    """Where a label factor's block is pinned to in latent space.

    `none` leaves the block to a classification head, which fixes the factor only up
    to an invertible linear map of the block. The others name a fixed table of target
    coordinates, one row per class, which is what makes a *dimension* -- and so a
    circuit's per-dimension leaf -- mean something.
    """

    NONE = "none"
    ONEHOT = "onehot"
    COLOUR_MNIST_FG = "colour_mnist_fg"
    COLOUR_MNIST_BG = "colour_mnist_bg"


class VAETrainingType(StrEnum):
    VANILLA = "vanilla"
    BETA = "beta"
    FACTOR = "factor"
    TCVAE = "tcvae"


class SupervisionConfig(BaseModel):
    """Which block of the latent each label factor is forced into.

    `dims`, `cardinalities` and `names` are parallel lists in target-vector order --
    colour-MNIST targets are [digit, fg, bg], so cardinalities [10, 6, 3]. Blocks are
    laid out contiguously from dimension 0; whatever `latent_dim` has left over stays
    free for everything the labels do not name.
    """

    model_config = ConfigDict(extra="forbid")

    dims: list[int]
    cardinalities: list[int]
    names: list[str] | None = None

    # Per factor, in the same order. Factors left at `none` keep a classification head;
    # the rest are pinned to their scheme's anchor table instead. Default: all `none`,
    # which is exactly the supervised VAE.
    anchors: list[AnchorScheme] | None = None
    # Width of the conditional prior around each anchor. Small enough that the classes
    # stay far apart, large enough that the decoder sees a band rather than a point --
    # this is the knob that decides whether values *between* anchors decode to anything.
    anchor_std: float = 0.1

    @model_validator(mode="after")
    def validate_blocks(self) -> Self:
        if not self.dims:
            raise ValueError("supervision needs at least one label factor")
        if len(self.dims) != len(self.cardinalities):
            raise ValueError(
                f"dims ({len(self.dims)}) and cardinalities "
                f"({len(self.cardinalities)}) must name the same label factors"
            )
        if self.names is not None and len(self.names) != len(self.dims):
            raise ValueError(
                f"names ({len(self.names)}) must have one entry per label factor "
                f"({len(self.dims)})"
            )
        if any(dim <= 0 for dim in self.dims):
            raise ValueError(f"every block needs at least one dimension: {self.dims}")
        if any(cardinality <= 1 for cardinality in self.cardinalities):
            raise ValueError(
                f"a label factor needs at least two classes: {self.cardinalities}"
            )
        if self.anchors is not None and len(self.anchors) != len(self.dims):
            raise ValueError(
                f"anchors ({len(self.anchors)}) must have one entry per label factor "
                f"({len(self.dims)})"
            )
        if self.anchor_std <= 0:
            raise ValueError(f"anchor_std must be positive: {self.anchor_std}")
        return self

    @property
    def num_supervised_dims(self) -> int:
        return sum(self.dims)

    @property
    def factor_names(self) -> list[str]:
        return self.names or [f"factor{i}" for i in range(len(self.dims))]

    @property
    def anchor_schemes(self) -> list[AnchorScheme]:
        return self.anchors or [AnchorScheme.NONE] * len(self.dims)

    @property
    def anchored_factors(self) -> list[int]:
        """Indices of the factors pinned to an anchor table."""
        return [
            i
            for i, scheme in enumerate(self.anchor_schemes)
            if scheme is not AnchorScheme.NONE
        ]

    @property
    def head_factors(self) -> list[int]:
        """Indices of the factors still supervised by a classification head."""
        return [
            i
            for i, scheme in enumerate(self.anchor_schemes)
            if scheme is AnchorScheme.NONE
        ]

    def slices(self) -> list[slice]:
        """One latent slice per label factor, in target-vector order."""
        bounds = list(accumulate(self.dims, initial=0))
        return [slice(start, stop) for start, stop in pairwise(bounds)]

    def anchored_dims(self) -> list[int]:
        """Latent dimensions covered by an anchor table, factors in target-vector order.

        Concatenating the anchor rows in this order lines them up with `mu[:, dims]`.
        """
        blocks = self.slices()
        return [
            dim
            for i in self.anchored_factors
            for dim in range(blocks[i].start, blocks[i].stop)
        ]


class AutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: AutoencoderType
    latent_dim: int
    num_blocks: int
    base_channels: int
    image_size: int = 0
    channels: int = 3 # rgb as default
    num_encoder_resblocks: int = 1
    num_decoder_resblocks: int = 1
    # Required by (and only read by) model_type=supervised/anchored.
    supervision: SupervisionConfig | None = None

    @model_validator(mode="after")
    def validate_supervision(self) -> Self:
        is_supervised = self.model_type in (
            AutoencoderType.SUPERVISED,
            AutoencoderType.ANCHORED,
        )
        if is_supervised and self.supervision is None:
            raise ValueError(
                f"model_type={self.model_type} needs a `supervision` block naming the "
                "latent dimensions each label factor is forced into"
            )
        if not is_supervised and self.supervision is not None:
            raise ValueError(
                f"`supervision` is only read by model_type=supervised/anchored, but "
                f"model_type is {self.model_type}; nothing would shape the latent"
            )
        if (
            self.model_type is AutoencoderType.SUPERVISED
            and self.supervision is not None
            and self.supervision.anchored_factors
        ):
            raise ValueError(
                "model_type=supervised has classification heads only; use "
                "model_type=anchored to pin a factor to an anchor table"
            )
        if (
            self.model_type is AutoencoderType.ANCHORED
            and self.supervision is not None
            and not self.supervision.anchored_factors
        ):
            raise ValueError(
                "model_type=anchored needs at least one factor with an anchor scheme; "
                "with all of them `none` this is just model_type=supervised"
            )
        if self.supervision is not None:
            used = self.supervision.num_supervised_dims
            if used > self.latent_dim:
                raise ValueError(
                    f"supervised blocks need {used} latent dimensions but latent_dim "
                    f"is {self.latent_dim}"
                )
        return self


class AutoencoderTrainingConfig(BaseTrainingConfig):
    beta: float
    beta_start: float
    beta_end: float
    kl_warmup_epochs: int
    vae_type: VAETrainingType

    # Weight on the latent supervision loss -- classification for model_type=supervised,
    # classification plus anchor KL for model_type=anchored. Ramped gamma_start ->
    # gamma_end over classifier_warmup_epochs *after* the KL warmup, so supervision only
    # starts pulling once the latent means something.
    gamma_start: float = 0.0
    gamma_end: float = 0.0
    classifier_warmup_epochs: int = 0
    # Relative weight of the anchor KL inside that gamma; only read by
    # model_type=anchored. The two supervision terms start orders of magnitude apart --
    # an anchor KL opens at (anchor spacing / anchor_std)^2 / 2 nats per dimension while
    # a cross-entropy opens at log(cardinality) -- so a model with both needs a knob
    # for their ratio and not just for their sum.
    anchor_weight: float = 1.0

    # only required (and used) when vae_type == tcvae; see validate_tcvae_params
    tcvae_alpha: float | None = None
    tcvae_beta: float | None = None
    tcvae_gamma: float | None = None

    # TODO move into config files if needed
    free_bits: float = 0.5
    lambda_perceptual: float = 1.0
    lambda_adversarial: float = 0.1
    adversarial_warmup_steps: int = 1000

    @model_validator(mode="after")
    def validate_beta(self) -> Self:
        assert self.beta >= 0
        assert self.beta_start <= self.beta_end
        if self.vae_type == VAETrainingType.BETA:
            assert self.beta == self.beta_end
        return self

    @model_validator(mode="after")
    def validate_gamma(self) -> Self:
        if self.gamma_start < 0 or self.gamma_end < 0:
            raise ValueError("gamma must be non-negative")
        if self.gamma_start > self.gamma_end:
            raise ValueError(
                f"gamma_start ({self.gamma_start}) must not exceed gamma_end "
                f"({self.gamma_end})"
            )
        if self.classifier_warmup_epochs < 0:
            raise ValueError("classifier_warmup_epochs must be non-negative")
        if self.anchor_weight < 0:
            raise ValueError("anchor_weight must be non-negative")
        return self

    @model_validator(mode="after")
    def validate_tcvae_params(self) -> Self:
        if self.vae_type == VAETrainingType.TCVAE:
            missing = [
                name
                for name, val in (
                    ("tcvae_alpha", self.tcvae_alpha),
                    ("tcvae_beta", self.tcvae_beta),
                    ("tcvae_gamma", self.tcvae_gamma),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"vae_type=tcvae requires {', '.join(missing)} to be set in training config"
                )
            assert self.tcvae_beta is not None
            assert self.beta_start <= self.tcvae_beta
        return self


class AERunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ae"]
    dataset: DatasetConfig
    model: AutoencoderConfig
    training: AutoencoderTrainingConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @model_validator(mode="after")
    def inject_image_size(self) -> Self:
        self.model.image_size = self.dataset.height
        self.model.channels = self.dataset.channels
        return self

    @model_validator(mode="after")
    def supervision_matches_gamma(self) -> Self:
        """A supervised model with gamma 0 trains heads nothing reads, and a positive
        gamma without supervised blocks has nothing to weight -- either way the run
        would silently not be the experiment it looks like."""
        supervised = self.model.supervision is not None
        if supervised and self.training.gamma_end <= 0:
            raise ValueError(
                "model.supervision is set but training.gamma_end is 0, so the "
                "supervision would never influence the latent"
            )
        if not supervised and self.training.gamma_end > 0:
            raise ValueError(
                f"training.gamma_end is {self.training.gamma_end} but the model has no "
                "`supervision` block to apply it to"
            )
        return self
