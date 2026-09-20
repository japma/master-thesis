"""Run configuration schemas.

Split by run type; re-exported here so `from utils.config import X` keeps working
regardless of which module `X` lives in.
"""

from utils.config.autoencoder import (
    AERunConfig,
    AnchorScheme,
    AutoencoderConfig,
    AutoencoderTrainingConfig,
    AutoencoderType,
    SupervisionConfig,
    VAETrainingType,
)
from utils.config.classifier import (
    ClassifierConfig,
    ClassifierRunConfig,
)
from utils.config.common import (
    WANDB_ENTITY,
    WANDB_PROJECT,
    BaseTrainingConfig,
    DatasetConfig,
    PretrainedAutoencoderConfig,
    WandbConfig,
)
from utils.config.cspn import (
    ConditioningType,
    CSPNConfig,
    CSPNEncoderConfig,
    CSPNEncoderType,
    CSPNRunConfig,
    CSPNTrainingConfig,
    CSPNType,
    PretrainedLabelPCConfig,
)
from utils.config.evaluation import (
    CheckpointConfig,
    EvaluationConfig,
    EvaluationRunConfig,
    GeneratedModelConfig,
    GenerationConfig,
    MarginalConfig,
)
from utils.config.joint_pc import (
    JointPCConfig,
    JointPCRunConfig,
)
from utils.config.label_pc import (
    LabelPCConfig,
    LabelPCRunConfig,
)
from utils.config.loading import load_config, load_dataset_config
from utils.config.neural_baseline import (
    NeuralBaselineConfig,
    NeuralBaselineRunConfig,
    NeuralBaselineType,
)

for _enum in (
    AutoencoderType,
    VAETrainingType,
    CSPNType,
    CSPNEncoderType,
    ConditioningType,
    NeuralBaselineType,
):
    _enum.__module__ = __name__


__all__ = [
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "AERunConfig",
    "AnchorScheme",
    "AutoencoderConfig",
    "AutoencoderTrainingConfig",
    "AutoencoderType",
    "BaseTrainingConfig",
    "CSPNConfig",
    "CSPNEncoderConfig",
    "CSPNEncoderType",
    "CSPNRunConfig",
    "CSPNTrainingConfig",
    "CSPNType",
    "CheckpointConfig",
    "ClassifierConfig",
    "ClassifierRunConfig",
    "ConditioningType",
    "DatasetConfig",
    "EvaluationConfig",
    "EvaluationRunConfig",
    "GeneratedModelConfig",
    "GenerationConfig",
    "JointPCConfig",
    "JointPCRunConfig",
    "LabelPCConfig",
    "LabelPCRunConfig",
    "MarginalConfig",
    "NeuralBaselineConfig",
    "NeuralBaselineRunConfig",
    "NeuralBaselineType",
    "PretrainedAutoencoderConfig",
    "PretrainedLabelPCConfig",
    "SupervisionConfig",
    "VAETrainingType",
    "WandbConfig",
    "load_config",
    "load_dataset_config",
]
