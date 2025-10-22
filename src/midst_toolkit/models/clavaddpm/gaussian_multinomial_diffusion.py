"""
Based on the code below.

https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
https://github.com/ehoogeboom/multinomial_diffusion
"""

import math
from collections.abc import Callable
from enum import Enum
from logging import DEBUG, INFO, WARNING
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

from midst_toolkit.common.logger import log
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.clavaddpm.diffusion_utils import (
    FoundNaNsError,
    discretized_gaussian_log_likelihood,
    extract,
    index_to_log_onehot,
    log_1_min_a,
    log_add_exp,
    log_categorical,
    mean_flat,
    normal_kl,
    one_hot_encoding_to_categories,
    sliced_logsumexp,
    sum_except_batch,
)


# Based in part on:
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
eps = 1e-8


class GaussianLossType(Enum):
    """Possible types of Gaussian loss."""

    MSE = "mse"
    KL = "kl"


class SchedulerType(Enum):
    """Possible types of scheduler."""

    COSINE = "cosine"
    LINEAR = "linear"


class GaussianParametrization(Enum):
    """Possible types of Gaussian parametrization."""

    EPS = "eps"
    X0 = "x0"


class Parametrization(Enum):
    """Possible types of parametrization."""

    X0 = "x0"
    DIRECT = "direct"


class SampleMethod(Enum):
    """Possible types of sample method."""

    UNIFORM = "uniform"
    IMPORTANCE = "importance"


class ConditioningFunction(Protocol):
    """The definition of a function used to condition the model output."""

    def __call__(self, features: Tensor, timestep: Tensor, **kwargs: Any) -> Tensor:
        """
        The function call definition.

        Args:
            features: The input features.
            timestep: The timestep.
            **kwargs: Extra keyword arguments passed to the model.

        Returns:
            The tensor result of the conditioning function.
        """
        ...


def get_named_beta_schedule(scheduler_type: SchedulerType, num_diffusion_timesteps: int) -> np.ndarray:
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.

    Args:
        scheduler_type: The scheduler type to use.
        num_diffusion_timesteps: The number of diffusion timesteps.

    Returns:
        The beta schedule.
    """
    if scheduler_type == SchedulerType.LINEAR:
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

    if scheduler_type == SchedulerType.COSINE:
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_type.value}")


def betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar: Callable, max_beta: float = 0.999) -> np.ndarray:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    Args:
        num_diffusion_timesteps: The number of timesteps to produce the betas.
        alpha_bar: A lambda that takes an argument t from 0 to 1 and
                  produces the cumulative product of (1-beta) up to that
                  part of the diffusion process.
        max_beta: The maximum beta to use; use values lower than 1 to
                  prevent singularities.

    Returns:
        The beta schedule.
    """
    if max_beta >= 1:
        log(WARNING, f"max_beta is set to {max_beta}. Use values lower than 1 to prevent singularities.")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

    return np.array(betas)


class GaussianMultinomialDiffusion(torch.nn.Module):
    def __init__(
        self,
        num_classes: np.ndarray,
        num_numerical_features: int,
        # TODO: change the type hint of denoise_fn to include None. See `train.train_classifier`.
        denoise_fn: torch.nn.Module,
        num_timesteps: int = 1000,
        gaussian_loss_type: GaussianLossType = GaussianLossType.MSE,
        gaussian_parametrization: GaussianParametrization = GaussianParametrization.EPS,
        parametrization: Parametrization = Parametrization.X0,
        scheduler_type: SchedulerType = SchedulerType.COSINE,
        device: torch.device = DEVICE,
    ):
        """
        Initialize a GaussianMultinomialDiffusion instance.

        Args:
            num_classes: The number of classes.
            num_numerical_features: The number of numerical features.
            denoise_fn: The denoising function.
            num_timesteps: The number of timesteps. Default is 1000.
            gaussian_loss_type: The type of Gaussian loss. Default is GaussianLossType.MSE.
            gaussian_parametrization: The type of Gaussian parametrization. Default is GaussianParametrization.EPS.
            parametrization: The type of parametrization. Default is Parametrization.X0.
            scheduler_type: The type of scheduler. Default is SchedulerType.COSINE.
            device: The device to use. Default is midst_toolkit.common.variables.DEVICE.
        """
        super().__init__()

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes  # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device)

        self.slices_for_classes = [np.arange(self.num_classes[0])]
        offsets: np.ndarray = np.cumsum(self.num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(np.append([0], offsets)).to(device)

        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler_type = scheduler_type
        self.device = device
        self.alphas: Tensor
        self.alphas_cumprod: Tensor
        self.alphas_cumprod_next: Tensor
        self.alphas_cumprod_prev: Tensor
        self.sqrt_alphas_cumprod: Tensor
        self.sqrt_one_minus_alphas_cumprod: Tensor
        self.log_cumprod_alpha: Tensor
        self.log_alpha: Tensor
        self.log_1_min_alpha: Tensor
        self.log_1_min_cumprod_alpha: Tensor
        self.sqrt_recipm1_alphas_cumprod: Tensor
        self.sqrt_recip_alphas_cumprod: Tensor
        self.lt_history: Tensor
        self.lt_count: Tensor

        buffers = self._calculate_buffer_values()

        # Gaussian diffusion
        betas = 1.0 - buffers["alphas"]
        self.posterior_variance = betas * (1.0 - buffers["alphas_cumprod_prev"]) / (1.0 - buffers["alphas_cumprod"])
        posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_log_variance_clipped = torch.from_numpy(posterior_log_variance_clipped).float().to(self.device)
        posterior_mean_coef1 = betas * torch.sqrt(buffers["alphas_cumprod_prev"]) / (1.0 - buffers["alphas_cumprod"])
        self.posterior_mean_coef1 = posterior_mean_coef1.float().to(self.device)
        coef2_denominator = (1.0 - buffers["alphas_cumprod_prev"]) * torch.sqrt(buffers["alphas"])
        coef2_numerator = 1.0 - buffers["alphas_cumprod"]
        self.posterior_mean_coef2 = (coef2_denominator / coef2_numerator).float().to(self.device)

        assert log_add_exp(buffers["log_alpha"], buffers["log_1_min_alpha"]).abs().sum().item() < 1.0e-5
        assert log_add_exp(buffers["log_cumprod_alpha"], buffers["log_1_min_cumprod_alpha"]).abs().sum().item() < 1e-5
        diff = torch.cumsum(buffers["log_alpha"], dim=0) - buffers["log_cumprod_alpha"]
        assert diff.abs().sum().item() < 1.0e-5

        # Convert to float32 and register buffers.
        for key, value in buffers.items():
            self.register_buffer(key, value.float().to(self.device))

    def _calculate_buffer_values(self) -> dict[str, Tensor]:
        """
        Calculate the values to register in this module's buffer.

        Returns:
            A dictionary of tensors with the values to register in the buffer. Will contain the keys:
            log_alpha, log_cumprod_alpha, log_1_min_alpha, log_1_min_cumprod_alpha, alphas_cumprod,
            alphas_cumprod_prev, alphas_cumprod_next, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod, lt_history, lt_count
        """
        a = 1.0 - get_named_beta_schedule(self.scheduler_type, self.num_timesteps)
        alphas = torch.tensor(a.astype("float64"))

        log_alpha = torch.log(alphas)
        log_cumprod_alpha = torch.cumsum(log_alpha, dim=0)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0))
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        return {
            "alphas": alphas,
            "log_alpha": log_alpha,
            "log_1_min_alpha": log_1_min_alpha,
            "log_1_min_cumprod_alpha": log_1_min_cumprod_alpha,
            "log_cumprod_alpha": log_cumprod_alpha,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "alphas_cumprod_next": alphas_cumprod_next,
            "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
            "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
            "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,
            "lt_history": torch.zeros(self.num_timesteps),
            "lt_count": torch.zeros(self.num_timesteps),
        }

    # Gaussian part
    def gaussian_q_mean_variance(self, x_start: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the mean and variance of the Gaussian posterior distribution.

        Args:
            x_start: The initial, noiseless input.
            timestep: The timestep.

        Returns:
            The mean and variance of the Gaussian posterior distribution.
        """
        mean = extract(self.sqrt_alphas_cumprod, timestep, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, timestep, x_start.shape)
        log_variance = extract(self.log_1_min_cumprod_alpha, timestep, x_start.shape)
        return mean, variance, log_variance

    def gaussian_q_sample(self, x_start: Tensor, timestep: Tensor, noise: Tensor | None = None) -> Tensor:
        """
        Sample from the Gaussian posterior distribution.

        Args:
            x_start: The initial, noiseless input.
            timestep: The timestep.
            noise: The noise. Optional, default is None.

        Returns:
            The sample from the Gaussian posterior distribution.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, timestep, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timestep, x_start.shape) * noise
        )

    def gaussian_q_posterior_mean_variance(
        self,
        features_start: Tensor,
        features_timestep: Tensor,
        timestep: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the mean and variance of the Gaussian posterior distribution.

        Args:
            features_start: The initial, noiseless input.
            features_timestep: The features used to compute the Gaussian parameters at the given timestep.
            timestep: The timestep.

        Returns:
            A tuple with 3 tensors: the mean, the variance, and the log variance of
            the Gaussian posterior distribution.
        """
        assert features_start.shape == features_timestep.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, timestep, features_timestep.shape) * features_start
            + extract(self.posterior_mean_coef2, timestep, features_timestep.shape) * features_timestep
        )
        posterior_variance = extract(self.posterior_variance, timestep, features_timestep.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, timestep, features_timestep.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == features_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(
        self,
        model_output: Tensor,
        features_timestep: Tensor,
        timestep: Tensor,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """
        Calculate the mean and variance of the Gaussian prior distribution.

        Args:
            model_output: The model output.
            features_timestep: The features of the Gaussian distribution at the given timestep.
            timestep: The timestep.
            model_kwargs: The model kwargs. Optional, default is None.

        Returns:
            A dictionary with the followingf keys:
             - "mean": the mean of the Gaussian prior distribution.
             - "variance": the variance of the Gaussian prior distribution.
             - "log_variance": the log variance of the Gaussian prior distribution.
             - "pred_xstart": the predicted xstart of the Gaussian prior distribution.
        """
        if model_kwargs is None:
            model_kwargs = {}

        batch_size, _ = features_timestep.shape[:2]
        assert timestep.shape == (batch_size,)

        model_variance = torch.cat(
            [
                self.posterior_variance[1].unsqueeze(0).to(self.device),
                (1.0 - self.alphas)[1:],
            ],
            dim=0,
        )
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, timestep, features_timestep.shape)
        model_log_variance = extract(model_log_variance, timestep, features_timestep.shape)

        if self.gaussian_parametrization == GaussianParametrization.EPS:
            pred_xstart = self._predict_xstart_from_eps(
                features_timestep=features_timestep, timestep=timestep, eps=model_output
            )

        elif self.gaussian_parametrization == GaussianParametrization.X0:
            pred_xstart = model_output

        else:
            raise ValueError(f"Unsupported Gaussian parametrization: {self.gaussian_parametrization}")

        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            features_start=pred_xstart,
            features_timestep=features_timestep,
            timestep=timestep,
        )

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == features_timestep.shape, (
            "Expected shapes to be equal, but got: ",
            f"model_mean.shape: {model_mean.shape}, ",
            f"model_log_variance.shape: {model_log_variance.shape}, ",
            f"pred_xstart.shape: {pred_xstart.shape}, ",
            f"features.shape: {features_timestep.shape}",
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _vb_terms_bpd(
        self,
        model_output: Tensor,
        features_start: Tensor,
        features_timestep: Tensor,
        timestep: Tensor,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """
        Calculate the VB terms for the Gaussian part.

        Args:
            model_output: The model output.
            features_start: The initial, noiseless input.
            features_timestep: The features used to compute the Gaussian parameters at the given timestep.
            timestep: The timestep.
            model_kwargs: The model kwargs. Optional, default is None.

        Returns:
            A dictionary with the following keys:
            - "output": The output of the VB terms.
            - "pred_xstart": The predicted xstart of the Gaussian prior distribution.
            - "out_mean": The mean of the Gaussian prior distribution.
            - "true_mean": The true mean of the Gaussian prior distribution.
        """
        if model_kwargs is None:
            model_kwargs = {}

        true_mean, _, true_log_variance_clipped = self.gaussian_q_posterior_mean_variance(
            features_start=features_start,
            features_timestep=features_timestep,
            timestep=timestep,
        )
        p_mean_variance = self.gaussian_p_mean_variance(
            model_output, features_timestep, timestep, model_kwargs=model_kwargs
        )
        kl = normal_kl(true_mean, true_log_variance_clipped, p_mean_variance["mean"], p_mean_variance["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            features_start, means=p_mean_variance["mean"], log_scales=0.5 * p_mean_variance["log_variance"]
        )
        assert decoder_nll.shape == features_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((timestep == 0), decoder_nll, kl)
        return {
            "output": output,
            "pred_xstart": p_mean_variance["pred_xstart"],
            "out_mean": p_mean_variance["mean"],
            "true_mean": true_mean,
        }

    def _prior_gaussian(self, x_start: Tensor) -> Tensor:
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        Args:
            x_start: the [N x C x ...] tensor of inputs.

        Returns:
            A batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def _gaussian_loss(
        self,
        model_out: Tensor,
        features_start: Tensor,
        features_timestep: Tensor,
        timestep: Tensor,
        noise: Tensor,
        model_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        """
        Calculate the Gaussian loss.

        Args:
            model_out: The model output.
            features_start: The initial, noiseless input.
            features_timestep: The features used to compute the Gaussian parameters at the given timestep.
            timestep: The timestep.
            noise: The noise.
            model_kwargs: The model kwargs. Optional, default is None.

        Returns:
            The Gaussian loss.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if self.gaussian_loss_type == GaussianLossType.MSE:
            return mean_flat((noise - model_out) ** 2)

        if self.gaussian_loss_type == GaussianLossType.KL:
            return self._vb_terms_bpd(
                model_output=model_out,
                features_start=features_start,
                features_timestep=features_timestep,
                timestep=timestep,
                model_kwargs=model_kwargs,
            )["output"]

        raise ValueError(f"Unsupported Gaussian loss type: {self.gaussian_loss_type}")

    def _predict_xstart_from_eps(self, features_timestep: Tensor, timestep: Tensor, eps: Tensor) -> Tensor:
        """
        Predict the xstart from the eps.

        Args:
            features_timestep: The features at the given timestep.
            timestep: The timestep.
            eps: The eps.

        Returns:
            The predicted xstart.
        """
        assert features_timestep.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, timestep, features_timestep.shape) * features_timestep
            - extract(self.sqrt_recipm1_alphas_cumprod, timestep, features_timestep.shape) * eps
        )

    def _predict_eps_from_xstart(self, features: Tensor, timestep: Tensor, pred_xstart: Tensor) -> Tensor:
        """
        Predict the eps from the xstart.

        Args:
            features: The features used to compute the Gaussian parameters.
            timestep: The timestep.
            pred_xstart: The predicted xstart.

        Returns:
            The predicted eps.
        """
        return (extract(self.sqrt_recip_alphas_cumprod, timestep, features.shape) * features - pred_xstart) / extract(
            self.sqrt_recipm1_alphas_cumprod, timestep, features.shape
        )

    def condition_mean(
        self,
        conditioning_function: ConditioningFunction,
        p_mean_var: dict[str, Tensor],
        features: Tensor,
        timestep: Tensor,
        model_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        """
        Compute the mean for the previous step, given a function ``conditioning_function``
        that computes the gradient of a conditional log probability with respect to
        ``features``. In particular, ``conditioning_function`` computes grad(log(p(y|x))),
        and we want to condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).

        Args:
            conditioning_function: The conditioning function.
            p_mean_var: The mean and variance of the Gaussian prior distribution.
            features: The features used to compute the Gaussian parameters.
            timestep: The timestep.
            model_kwargs: The model kwargs. Optional, default is None.

        Returns:
            The conditioned mean for the previous step.
        """
        if model_kwargs is None:
            model_kwargs = {}

        gradient = conditioning_function(features, timestep, **model_kwargs)
        return p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()

    def condition_score(
        self,
        conditioning_function: ConditioningFunction,
        p_mean_var: dict[str, Tensor],
        features: Tensor,
        timestep: Tensor,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by ``conditioning_function``.

        See condition_mean() for details on ``conditioning_function``.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).

        Args:
            conditioning_function: The conditioning function.
            p_mean_var: The mean and variance of the Gaussian prior distribution.
            features: The features used to compute the Gaussian parameters.
            timestep: The timestep.
            model_kwargs: The model kwargs. Optional, default is None.

        Returns:
            A copy of the ``p_mean_var`` dictionary with the following additional keys:
                - "pred_xstart": the predicted xstart.
                - "mean": the mean of the Gaussian prior distribution.
        """
        if model_kwargs is None:
            model_kwargs = {}

        alpha_bar = extract(self.alphas_cumprod, timestep, features.shape)

        eps = self._predict_eps_from_xstart(features, timestep, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * conditioning_function(features, timestep, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(features, timestep, eps)
        out["mean"], _, _ = self.gaussian_q_posterior_mean_variance(
            features_start=out["pred_xstart"],
            features_timestep=features,
            timestep=timestep,
        )
        return out

    def gaussian_p_sample(
        self,
        model_out: Tensor,
        features: Tensor,
        timestep: Tensor,
        model_kwargs: dict[str, Any] | None = None,
        cond_fn: ConditioningFunction | None = None,
    ) -> dict[str, Tensor]:
        """
        Sample from the Gaussian posterior distribution.

        Args:
            model_out: The model output.
            features: The features used to compute the Gaussian parameters.
            timestep: The timestep.
            model_kwargs: The model kwargs. Optional, default is None.
            cond_fn: The conditioning function. Optional, default is None.

        Returns:
            A dictionary with teo tensors:
             - "sample": the sample from the Gaussian posterior distribution.
             - "pred_xstart": the predicted xstart.
        """
        if model_kwargs is None:
            model_kwargs = {}

        out = self.gaussian_p_mean_variance(
            model_out,
            features,
            timestep,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(features)
        # no noise when t == 0
        nonzero_mask = (timestep != 0).float().view(-1, *([1] * (len(features.shape) - 1)))

        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, features, timestep, model_kwargs=model_kwargs)

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    # Multinomial part

    def multinomial_kl(self, log_prob1: Tensor, log_prob2: Tensor) -> Tensor:
        """
        Calculate the KL divergence between two log probabilities.

        Args:
            log_prob1: The first log probability.
            log_prob2: The second log probability.

        Returns:
            The KL divergence.
        """
        return (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)

    def q_pred_one_timestep(self, log_x_t: Tensor, timestep: Tensor) -> Tensor:
        """
        Calculate the predicted log probability for one timestep.

        Args:
            log_x_t: The log samples of the features at the given timestep.
            timestep: The timestep.

        Returns:
            The predicted log probability.
        """
        log_alpha_t = extract(self.log_alpha, timestep, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, timestep, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        return log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.num_classes_expanded),
        )

    def q_pred(self, log_x_start: Tensor, timestep: Tensor) -> Tensor:
        """
        Compute the predicted log-probability at ``timestep`` given ``log_x_start``.

        Args:
            log_x_start: The log sample of the start.
            timestep: The timestep.

        Returns:
            The predicted log probability.
        """
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, timestep, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, timestep, log_x_start.shape)

        return log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded),
        )

    def predict_start(self, model_out: Tensor, log_x_t: Tensor) -> Tensor:
        """
        Predict the start from the model output.

        Args:
            model_out: The model output.
            log_x_t: The log sample of the features at the given timestep.

        Returns:
            The predicted start.
        """
        assert model_out.size(0) == log_x_t.size(0)
        assert self.num_classes is not None
        assert model_out.size(1) == self.num_classes.sum(), f"{model_out.size()}"

        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = functional.log_softmax(model_out[:, ix], dim=1)
        return log_pred

    def q_posterior(self, log_x_start: Tensor, log_x_t: Tensor, timestep: Tensor) -> Tensor:
        """
        Calculate the posterior probability for one timestep.

        Args:
            log_x_start: The log sample of the initial input.
            log_x_t: The log sample of the features at the given timestep.
            timestep: The timestep.

        Returns:
            The posterior probability.
        """
        t_minus_1 = timestep - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_ev_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = timestep.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_ev_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_ev_qxtmin_x0.to(torch.float32))

        # unnormed_logprobs = log_EV_qxtmin_x0 +
        #                     log q_pred_one_timestep(x_t, t)
        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_ev_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, timestep)

        return unnormed_logprobs - sliced_logsumexp(unnormed_logprobs, self.offsets)

    def p_pred(self, model_out: Tensor, log_x: Tensor, timestep: Tensor) -> Tensor:
        """
        Predict the start from the model output based on the parametrization set in ``self.parametrization``.

        Args:
            model_out: The model output.
            log_x: The log sample of the features.
            timestep: The timestep.

        Returns:
            The predicted start from the model output.
        """
        if self.parametrization == Parametrization.X0:
            log_x_recon = self.predict_start(model_out, log_x)
            log_model_pred = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_x, timestep=timestep)

        elif self.parametrization == Parametrization.DIRECT:
            log_model_pred = self.predict_start(model_out, log_x)

        else:
            raise ValueError(f"Unsupported parametrization: {self.parametrization}")

        return log_model_pred

    @torch.no_grad()
    def p_sample(self, model_out: Tensor, log_x: Tensor, timestep: Tensor) -> Tensor:
        """
        Sample from the model output.

        Args:
            model_out: The model output.
            log_x: The log sample of the features.
            timestep: The timestep.

        Returns:
            The sample from the model output.
        """
        model_log_prob = self.p_pred(model_out, log_x=log_x, timestep=timestep)
        return self.log_sample_categorical(model_log_prob)

    def log_sample_categorical(self, logits: Tensor) -> Tensor:
        """
        Sample from the categorical logits.

        Args:
            logits: The logits.

        Returns:
            The sample from the categorical logits.
        """
        full_sample = []
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))

        full_sample_tensor = torch.cat(full_sample, dim=1)
        return index_to_log_onehot(full_sample_tensor, torch.from_numpy(self.num_classes))

    def q_sample(self, log_x_start: Tensor, timestep: Tensor) -> Tensor:
        """
        Sample from the log of the initial input for one timestep.

        Args:
            log_x_start: The log of the initial input.
            timestep: The timestep.

        Returns:
            The sample from the categorical logits.
        """
        log_ev_qxt_x0 = self.q_pred(log_x_start, timestep)
        return self.log_sample_categorical(log_ev_qxt_x0)

    def kl_prior(self, log_x_start: Tensor) -> Tensor:
        """
        Calculate the KL divergence between the prior and the posterior.

        Args:
            log_x_start: The log sample of the initial input.

        Returns:
            The KL divergence between the prior and the posterior.
        """
        batch_size = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(batch_size, device=device).long()

        log_qxt_prob = self.q_pred(log_x_start, timestep=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes_expanded * torch.ones_like(log_qxt_prob))

        kl_prior = self.multinomial_kl(log_qxt_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_lt(
        self,
        model_out: Tensor,
        log_x_start: Tensor,
        log_x_t: Tensor,
        timestep: Tensor,
        detach_mean: bool = False,
    ) -> Tensor:
        """
        Calculate the KL divergence between the true and the model probability.

        Args:
            model_out: The model output.
            log_x_start: The log sample of the initial input.
            log_x_t: The log samples of the features at the given timestep.
            timestep: The timestep.
            detach_mean: Whether to detach the mean.

        Returns:
            The KL divergence between the true and the model probability.
        """
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_x_t, timestep=timestep)
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, timestep=timestep)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (timestep == torch.zeros_like(timestep)).float()
        return mask * decoder_nll + (1.0 - mask) * kl

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        method: SampleMethod = SampleMethod.UNIFORM,
    ) -> tuple[Tensor, Tensor]:
        """
        Sample the timestep.

        Args:
            batch_size: The batch size.
            device: The device.
            method: The method to sample the timestep.

        Returns:
            The timestep and the probability of the timestep.
        """
        if method == SampleMethod.IMPORTANCE:
            if not (self.lt_count > 10).all():
                return self.sample_time(batch_size, device, method=SampleMethod.UNIFORM)

            lt_sqrt = torch.sqrt(self.lt_history + 1e-10) + 0.0001
            lt_sqrt[0] = lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = (lt_sqrt / lt_sqrt.sum()).to(device)

            timestep = torch.multinomial(pt_all, num_samples=batch_size, replacement=True).to(device)

            p_timestep = pt_all.gather(dim=0, index=timestep)

            return timestep, p_timestep

        if method == SampleMethod.UNIFORM:
            timestep = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

            p_timestep = torch.ones_like(timestep).float() / self.num_timesteps
            return timestep, p_timestep

        raise ValueError(f"Unsupported method: {method}")

    def _multinomial_loss(
        self,
        model_out: Tensor,
        log_x_start: Tensor,
        log_x_t: Tensor,
        timestep: Tensor,
        p_timestep: Tensor,
    ) -> Tensor:
        """
        Calculate the multinomial loss.

        Args:
            model_out: The model output.
            log_x_start: The log samples of the initial input.
            log_x_t: The log samples of the features at the given timestep.
            timestep: The timestep.
            p_timestep: The probability of the timestep.

        Returns:
            The multinomial loss.
        """
        # Here we are calculating the VB_STOCHASTIC loss. In the original implementation, there
        # was a choice between VB_STOCHASTIC and VB_ALL. VB_ALL is deprecated for being too
        # expensive to calculate.
        kl = self.compute_lt(model_out, log_x_start, log_x_t, timestep)
        kl_prior = self.kl_prior(log_x_start)
        # Upweigh loss term of the kl
        return (kl / p_timestep) + kl_prior

    def mixed_loss(self, features: Tensor, out_dict: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """
        Calculate the mixed loss.

        Args:
            features: The input features.
            out_dict: The output dictionary.

        Returns:
            The multinomial loss and the Gaussian loss.
        """
        batch_size = features.shape[0]
        timestep, p_timestep = self.sample_time(batch_size, self.device, method=SampleMethod.UNIFORM)

        numerical_features = features[:, : self.num_numerical_features]
        categorical_features = features[:, self.num_numerical_features :]

        numerical_features_ts = numerical_features
        if numerical_features.shape[1] > 0:
            noise = torch.randn_like(numerical_features)
            numerical_features_ts = self.gaussian_q_sample(numerical_features, timestep, noise=noise)

        log_categorical_features_ts = categorical_features
        if categorical_features.shape[1] > 0:
            log_x_cat = index_to_log_onehot(categorical_features.long(), torch.from_numpy(self.num_classes))
            log_categorical_features_ts = self.q_sample(log_x_start=log_x_cat, timestep=timestep)

        input_features = torch.cat([numerical_features_ts, log_categorical_features_ts], dim=1)

        model_output = self._denoise_fn(input_features, timestep, **out_dict)

        model_numerical_output = model_output[:, : self.num_numerical_features]
        model_categorical_output = model_output[:, self.num_numerical_features :]

        multinomial_loss = torch.zeros((1,)).float()
        gaussian_loss = torch.zeros((1,)).float()
        if categorical_features.shape[1] > 0:
            multinomial_loss = self._multinomial_loss(
                model_categorical_output,
                log_x_cat,
                log_categorical_features_ts,
                timestep,
                p_timestep,
            )
            multinomial_loss = multinomial_loss / len(self.num_classes)

        if numerical_features.shape[1] > 0:
            gaussian_loss = self._gaussian_loss(
                model_numerical_output,
                numerical_features,
                numerical_features_ts,
                timestep,
                noise,
            )

        return multinomial_loss.mean(), gaussian_loss.mean()

    @torch.no_grad()
    def gaussian_ddim_step(
        self,
        model_mumerical_output: Tensor,
        features: Tensor,
        timestep: Tensor,
        eta: float = 0.0,
        model_kwargs: dict[str, Any] | None = None,
        cond_fn: ConditioningFunction | None = None,
    ) -> Tensor:
        """
        Calculate the Gaussian DDIM step.

        Args:
            model_mumerical_output: The numerical features of themodel output.
            features: The features.
            timestep: The timestep.
            eta: The DDIM stochasticity coefficient. Optional, default is 0.0.
            model_kwargs: The model kwargs. Optional, default is None.
            cond_fn: The conditioning function. Optional, default is None.

        Returns:
            The predicted features.
        """
        if model_kwargs is None:
            model_kwargs = {}

        out = self.gaussian_p_mean_variance(
            model_mumerical_output,
            features,
            timestep,
            model_kwargs=None,
        )

        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, features, timestep, model_kwargs=model_kwargs)

        eps = self._predict_eps_from_xstart(features, timestep, out["pred_xstart"])

        alpha_bar = extract(self.alphas_cumprod, timestep, features.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, timestep, features.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)

        noise = torch.randn_like(features)
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        nonzero_mask = (timestep != 0).float().view(-1, *([1] * (len(features.shape) - 1)))  # no noise when t == 0
        return mean_pred + nonzero_mask * sigma * noise

    @torch.no_grad()
    def gaussian_ddim_sample(
        self,
        noise: Tensor,
        num_timesteps: int,
        out_dict: dict[str, Tensor],
        eta: float = 0.0,
        model_kwargs: Any | None = None,
        cond_fn: ConditioningFunction | None = None,
    ) -> Tensor:
        """
        Produce the Gaussian DDIM sample.

        Args:
            noise: The noise.
            num_timesteps: The number of timesteps.
            out_dict: The output dictionary.
            eta: The DDIM stochasticity coefficient. Optional, default is 0.0.
            model_kwargs: The model kwargs. Optional, default is None.
            cond_fn: The conditioning function. Optional, default is None.

        Returns:
            The predicted features.
        """
        features = noise
        batch_size = features.shape[0]
        for t in reversed(range(num_timesteps)):
            log(DEBUG, f"Sample timestep {t:4d}")
            t_array = (torch.ones(batch_size, device=self.device) * t).long()
            out_num = self._denoise_fn(features, t_array, **out_dict)
            features = self.gaussian_ddim_step(
                out_num,
                features,
                t_array,
                eta=eta,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
            )

        return features

    @torch.no_grad()
    def gaussian_ddim_reverse_step(
        self,
        model_out_num: Tensor,
        features: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """
        Calculate the Gaussian DDIM reverse step.

        Args:
            model_out_num: The numerical features of the model output.
            features: The input features.
            timestep: The timestep.

        Returns:
            The predicted features.
        """
        out = self.gaussian_p_mean_variance(model_out_num, features, timestep)

        coefficient = extract(self.sqrt_recip_alphas_cumprod, timestep, features.shape)
        denominator = extract(self.sqrt_recipm1_alphas_cumprod, timestep, features.shape)
        numerator = coefficient * features - out["pred_xstart"]
        eps = numerator / denominator

        alpha_bar_next = extract(self.alphas_cumprod_next, timestep, features.shape)

        return out["pred_xstart"] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps

    @torch.no_grad()
    def gaussian_ddim_reverse_sample(
        self,
        features: Tensor,
        num_timesteps: int,
        out_dict: dict[str, Tensor],
    ) -> Tensor:
        """
        Produce the Gaussian DDIM reverse sample.

        Args:
            features: The input features.
            num_timesteps: The number of timesteps.
            out_dict: The output dictionary.

        Returns:
            The predicted features.
        """
        batch_size = features.shape[0]
        output_features = features.clone()

        for t in range(num_timesteps):
            log(DEBUG, f"Reverse timestep {t:4d}")
            t_array = (torch.ones(batch_size, device=self.device) * t).long()
            out_num = self._denoise_fn(output_features, t_array, **out_dict)
            output_features = self.gaussian_ddim_reverse_step(out_num, output_features, t_array)

        return output_features

    @torch.no_grad()
    def multinomial_ddim_step(
        self,
        model_out_cat: Tensor,
        log_x_t: Tensor,
        timestep: Tensor,
        eta: float = 0.0,
    ) -> Tensor:
        """
        Calculate the multinomial DDIM step.

        Args:
            model_out_cat: The categorical model output.
            log_x_t: The log samples of the features at the given timestep.
            timestep: The timestep.
            eta: The DDIM stochasticity coefficient. Optional, default is 0.0.

        Returns:
            The multinomial DDIM step.
        """
        log_x0 = self.predict_start(model_out_cat, log_x_t=log_x_t)

        alpha_bar = extract(self.alphas_cumprod, timestep, log_x_t.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, timestep, log_x_t.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)

        coef1 = sigma
        coef2 = alpha_bar_prev - sigma * alpha_bar
        coef3 = 1 - coef1 - coef2

        log_ps = torch.stack(
            [
                torch.log(coef1) + log_x_t,
                torch.log(coef2) + log_x0,
                torch.log(coef3) - torch.log(self.num_classes_expanded),
            ],
            dim=2,
        )

        log_prob = torch.logsumexp(log_ps, dim=2)

        return self.log_sample_categorical(log_prob)

    @torch.no_grad()
    def sample_ddim(
        self,
        batch_size: int,
        target_dist: Tensor,
        model_kwargs: dict[str, Any] | None = None,
        cond_fn: ConditioningFunction | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Sample using DDIM.

        Args:
            batch_size: The batch size.
            target_dist: Class distribution to sample labels from.
            model_kwargs: The model kwargs. Optional, default is None.
            cond_fn: The conditioning function. Optional, default is None.

        Returns:
            The samples and the output dictionary.
        """
        if model_kwargs is None:
            model_kwargs = {}

        z_norm = torch.randn((batch_size, self.num_numerical_features), device=self.device)

        assert self.num_classes is not None
        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((batch_size, 0), device=self.device).float()
        if has_cat:
            uniform_logits = torch.zeros((batch_size, len(self.num_classes_expanded)), device=self.device)
            log_z = self.log_sample_categorical(uniform_logits)

        y = torch.multinomial(target_dist, num_samples=batch_size, replacement=True)
        out_dict = {"y": y.long().to(self.device)}
        for i in reversed(range(0, self.num_timesteps)):
            log(DEBUG, f"Sample timestep {i:4d}")
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            model_out = self._denoise_fn(torch.cat([z_norm, log_z], dim=1).float(), t, **out_dict)
            model_out_num = model_out[:, : self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features :]
            z_norm = self.gaussian_ddim_step(
                model_out_num,
                z_norm,
                t,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
            )
            if has_cat:
                log_z = self.multinomial_ddim_step(model_out_cat, log_z, t)

        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = one_hot_encoding_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()

        return sample, out_dict

    @torch.no_grad()
    def conditional_sample(
        self,
        targets: Tensor,
        model_kwargs: dict[str, Any] | None = None,
        cond_fn: ConditioningFunction | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Sample using conditional DDIM.

        Args:
            targets: The targets.
            model_kwargs: The model kwargs. Optional, default is None.
            cond_fn: The conditioning function. Optional, default is None.

        Returns:
            The samples and the output dictionary.
        """
        if model_kwargs is None:
            model_kwargs = {}

        batch_size = len(targets)
        z_norm = torch.randn((batch_size, self.num_numerical_features), device=self.device)
        assert self.num_classes is not None
        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((batch_size, 0), device=self.device).float()

        out_dict = {"y": targets.long().to(self.device)}
        for i in reversed(range(0, self.num_timesteps)):
            log(DEBUG, f"Sample timestep {i:4d}")
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            model_out = self._denoise_fn(torch.cat([z_norm, log_z], dim=1).float(), t, **out_dict)
            model_out_num = model_out[:, : self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features :]
            z_norm = self.gaussian_p_sample(
                model_out_num,
                z_norm,
                t,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
            )["sample"]
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t)

        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = one_hot_encoding_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        target_dist: Tensor,
        model_kwargs: dict[str, Any] | None = None,
        cond_fn: ConditioningFunction | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Sample using ancestral (DDPM-style) sampling.

        Args:
            batch_size: The batch size.
            target_dist: Class distribution to sample labels from.
            model_kwargs: The model kwargs. Optional, default is None.
            cond_fn: The conditioning function. Optional, default is None.

        Returns:
            The samples and the output dictionary.
        """
        if model_kwargs is None:
            model_kwargs = {}

        z_norm = torch.randn((batch_size, self.num_numerical_features), device=self.device)

        assert self.num_classes is not None
        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((batch_size, 0), device=self.device).float()
        if has_cat:
            uniform_logits = torch.zeros((batch_size, len(self.num_classes_expanded)), device=self.device)
            log_z = self.log_sample_categorical(uniform_logits)

        y = torch.multinomial(target_dist, num_samples=batch_size, replacement=True)
        out_dict = {"y": y.long().to(self.device)}
        for i in reversed(range(0, self.num_timesteps)):
            log(DEBUG, f"Sample timestep {i:4d}")
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            model_out = self._denoise_fn(torch.cat([z_norm, log_z], dim=1).float(), t, **out_dict)
            model_out_num = model_out[:, : self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features :]
            z_norm = self.gaussian_p_sample(
                model_out_num,
                z_norm,
                t,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
            )["sample"]
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t)

        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = one_hot_encoding_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict

    def sample_all(
        self,
        num_samples: int,
        batch_size: int,
        target_dist: Tensor,
        ddim: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        cond_fn: ConditioningFunction | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Generate samples in batches of ``batch_size`` until ``num_samples`` are produced.
        Uses DDIM if ``ddim`` is ``True``.

        Args:
            num_samples: The number of samples.
            batch_size: The batch size.
            target_dist: Class distribution to sample labels from.
            ddim: Whether to use DDIM. Optional, default is False.
            model_kwargs: The model kwargs. Optional, default is None.
            cond_fn: The conditioning function. Optional, default is None.

        Returns:
            A tuple with the generated features and corresponding targets.
        """
        if ddim:
            log(INFO, "Sample using DDIM.")
            sample_fn = self.sample_ddim
        else:
            sample_fn = self.sample

        all_targets = []
        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            sample, out_dict = sample_fn(batch_size, target_dist, model_kwargs=model_kwargs, cond_fn=cond_fn)
            mask_nan = torch.any(sample.isnan(), dim=1)
            sample = sample[~mask_nan]
            out_dict["y"] = out_dict["y"][~mask_nan]

            all_samples.append(sample)
            all_targets.append(out_dict["y"].cpu())
            if sample.shape[0] != batch_size:
                raise FoundNaNsError
            num_generated += sample.shape[0]

        generated_features = torch.cat(all_samples, dim=0)[:num_samples]
        generated_targets = torch.cat(all_targets, dim=0)[:num_samples]

        return generated_features, generated_targets
