"""Samplers for the ClavaDDPM model."""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import torch
from torch import Tensor


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged. However, subclasses may override sample() to
    change how the resampled terms are reweighted, allowing for actual changes
    in the objective.
    """

    @abstractmethod
    def weights(self) -> Tensor:
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size: int, device: str) -> tuple[Tensor, Tensor]:
        # TODO: what's happening with batch_size? Is is also the number of timesteps?
        # We need to clarify this.
        """
        Importance-sample timesteps for a batch.

        Args:
            batch_size: The number of timesteps.
            device: The torch device to save to.

        Returns:
            A tuple (timesteps, weights):
                - timesteps: a tensor of timestep indices.
                - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights().cpu().numpy()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps: int):
        """
        Initialize the UniformSampler.

        Args:
            num_timesteps: The number of diffusion timesteps.
        """
        self.num_timesteps = num_timesteps
        self._weights = torch.from_numpy(np.ones([num_timesteps]))

    def weights(self) -> Tensor:
        """Return the weights."""
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts: Tensor, local_losses: Tensor) -> None:
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        Args:
            local_ts: An integer Tensor of timesteps.
            local_losses: A 1D Tensor of losses.
        """
        batch_sizes = [
            torch.tensor([0], dtype=torch.int32, device=local_ts.device)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(
            batch_sizes,
            torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        max_bs = max([int(x.item()) for x in batch_sizes])

        timestep_batches = [torch.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [torch.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        torch.distributed.all_gather(timestep_batches, local_ts)
        torch.distributed.all_gather(loss_batches, local_losses)
        timesteps = [x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts: list[int], losses: list[float]) -> None:
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        Args:
            ts: A list of int timesteps.
            losses: A list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(
        self,
        num_timesteps: int,
        history_per_term: int = 10,
        uniform_prob: float = 0.001,
    ):
        """
        Initialize the LossSecondMomentResampler.

        Args:
            num_timesteps: The number of diffusion timesteps.
            history_per_term: The number of losses to keep for each timestep.
            uniform_prob: The probability of sampling a uniform timestep.
        """
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([num_timesteps, history_per_term], dtype=np.float64)
        self._loss_counts = np.zeros([num_timesteps], dtype=np.uint)

    def weights(self):
        """
        Return the weights.

        Warms up the sampler if it's not warmed up.
        """
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts: list[int], losses: list[float]) -> None:
        """
        Update the reweighting using losses from the model.

        Args:
            ts: The timesteps.
            losses: The losses.
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self) -> bool:
        """
        Check if the sampler is warmed up by checking if the loss counts are equal
        to the history per term.

        Returns:
            True if the sampler is warmed up, False otherwise.
        """
        return (self._loss_counts == self.history_per_term).all()


class ScheduleSamplerType(Enum):
    """Possible types of schedule sampler."""

    UNIFORM = "uniform"
    LOSS_SECOND_MOMENT = "loss-second-moment"

    def create_named_schedule_sampler(self, num_timesteps: int) -> ScheduleSampler:
        """
        Create a ScheduleSampler from a library of pre-defined samplers.

        Args:
            num_timesteps: The number of diffusion timesteps.

        Returns:
            The UniformSampler if ScheduleSamplerType.UNIFORM, LossSecondMomentResampler
            if ScheduleSamplerType.LOSS_SECOND_MOMENT.
        """
        if self == ScheduleSamplerType.UNIFORM:
            return UniformSampler(num_timesteps)
        if self == ScheduleSamplerType.LOSS_SECOND_MOMENT:
            return LossSecondMomentResampler(num_timesteps)
        raise NotImplementedError(f"Unsupported schedule sampler: {self.value}")
