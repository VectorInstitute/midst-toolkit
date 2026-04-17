from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from midst_toolkit.models.clavaddpm.enumerations import ClusteringMethod
from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import GaussianLossType, SchedulerType
from midst_toolkit.models.clavaddpm.model import ModelType


class GeneralConfig(BaseModel):
    """General configuration settings."""

    data_dir: Path
    test_data_dir: Path
    exp_name: str
    workspace_dir: Path
    sample_prefix: str


class ClavaDDPMClusteringConfig(BaseModel):
    """Configuration for the trainer's clustering model."""

    num_clusters: int | dict[str, int]
    clustering_method: ClusteringMethod
    parent_scale: float


class ClavaDDPMDiffusionConfig(BaseModel):
    """Configuration for the trainer's diffusion model."""

    d_layers: list[int]
    dropout: float
    num_timesteps: int
    model_type: ModelType
    iterations: int  # This will determine the amount of steps of the diffusion model.
    batch_size: int
    lr: float
    gaussian_loss_type: GaussianLossType
    weight_decay: float
    scheduler: SchedulerType
    data_split_ratios: list[float] = [0.7, 0.2, 0.1]

    @model_validator(mode="after")
    def validate_data_split_ratios(self) -> Self:
        """Validate data_split_ratios."""
        assert len(self.data_split_ratios) == 3, "The ratios must be a list of 3 values (train, validation, test). "

        return self


class ClavaDDPMClassifierConfig(BaseModel):
    """Configuration for the trainer's classifier model."""

    d_layers: list[int]
    lr: float
    dim_t: int
    batch_size: int
    iterations: int  # This will determine the amount of steps of the classifier model.
    data_split_ratios: list[float] = [0.7, 0.2, 0.1]

    @model_validator(mode="after")
    def validate_data_split_ratios(self) -> Self:
        """Post-initialization checks and validations."""
        assert len(self.data_split_ratios) == 3, "The ratios must be a list of 3 values (train, validation, test)."

        return self


class ClavaDDPMSamplingConfig(BaseModel):
    """Configuration for the synthesizer's sampling process."""

    batch_size: int
    classifier_scale: float


class ClavaDDPMMatchingConfig(BaseModel):
    """Configuration for the synthesizer's matching process."""

    num_matching_clusters: int
    matching_batch_size: int
    unique_matching: bool
    no_matching: bool


class CTGANModelConfig(BaseModel):
    """Configuration for the CTGAN model."""

    epochs: int
    verbose: bool


class CTGANSynthesizingConfig(BaseModel):
    """Configuration for the CTGAN model."""

    sample_size: int


class TrainingConfig(BaseModel):
    """Base configuration settings for training models."""

    model_config = ConfigDict(extra="forbid")  # disallow extra fields from config files

    general: GeneralConfig
    save_dir: Path | None = None


class ClavaDDPMTrainingConfig(TrainingConfig):
    """All configuration settings for training, synthesizing, and fine tuning TabDDPM models."""

    clustering: ClavaDDPMClusteringConfig
    diffusion: ClavaDDPMDiffusionConfig
    classifier: ClavaDDPMClassifierConfig
    sampling: ClavaDDPMSamplingConfig
    matching: ClavaDDPMMatchingConfig


class CTGANTrainingConfig(TrainingConfig):
    """All configuration settings for training, synthesizing, and fine tuning CTGAN models."""

    training: CTGANModelConfig
    synthesizing: CTGANSynthesizingConfig
