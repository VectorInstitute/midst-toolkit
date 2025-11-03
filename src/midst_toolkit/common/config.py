from pathlib import Path
from typing import Self

from pydantic import BaseModel, model_validator

from midst_toolkit.models.clavaddpm.enumerations import ClusteringMethod
from midst_toolkit.models.clavaddpm.gaussian_multinomial_diffusion import GaussianLossType, SchedulerType
from midst_toolkit.models.clavaddpm.model import ModelType


class GeneralConfig(BaseModel):
    """General configuration settings for training and synthesizing."""

    exp_name: str
    workspace_dir: Path
    sample_prefix: str


class ClusteringConfig(BaseModel):
    """Configuration for the clustering model."""

    num_clusters: int | dict[str, int]
    clustering_method: ClusteringMethod
    parent_scale: float


class DiffusionConfig(BaseModel):
    """Configuration for the diffusion model."""

    d_layers: list[int]
    dropout: float
    num_timesteps: int
    model_type: ModelType
    iterations: int
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


class ClassifierConfig(BaseModel):
    """Configuration for the classifier model."""

    d_layers: list[int]
    lr: float
    dim_t: int
    batch_size: int
    iterations: int
    data_split_ratios: list[float] = [0.7, 0.2, 0.1]

    @model_validator(mode="after")
    def validate_data_split_ratios(self) -> Self:
        """Post-initialization checks and validations."""
        assert len(self.data_split_ratios) == 3, "The ratios must be a list of 3 values (train, validation, test)."

        return self


class SamplingConfig(BaseModel):
    """Configuration for the sampling model."""

    batch_size: int
    classifier_scale: float


class MatchingConfig(BaseModel):
    """Configuration for the matching model."""

    num_matching_clusters: int
    matching_batch_size: int
    unique_matching: bool
    no_matching: bool
