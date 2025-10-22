"""Utility functions for the diffusion models."""

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional


def normal_kl(
    mean1: Tensor | float,
    logvar1: Tensor | float,
    mean2: Tensor | float,
    logvar2: Tensor | float,
) -> Tensor:
    """
    Compute the KL divergence between two Gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.

    Note: at least one on the arguments must be a Tensor.

    Args:
        mean1: The mean of the first Gaussian.
        logvar1: The log variance of the first Gaussian.
        mean2: The mean of the second Gaussian.
        logvar2: The log variance of the second Gaussian.

    Returns:
        The KL divergence between the two Gaussians.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor) for x in (logvar1, logvar2)]

    return 0.5 * (
        -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x: Tensor) -> Tensor:
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.

    Args:
        x: The input tensor.

    Returns:
        The cumulative distribution function of the standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x: Tensor, *, means: Tensor, log_scales: Tensor) -> Tensor:
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    Args:
        x: The target images. It is assumed that this was uint8 values, rescaled to the range [-1, 1].
        means: The Gaussian mean Tensor.
        log_scales: The Gaussian log stddev Tensor.

    Returns:
        A tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def sum_except_batch(x: Tensor, num_dims: int = 1) -> Tensor:
    """
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    """
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def mean_flat(tensor: Tensor) -> Tensor:
    """
    Take the mean over all non-batch dimensions. The first dimension should be the batch.

    Args:
        tensor: The tensor.

    Returns:
        The mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def one_hot_encoding_to_categories(one_hot_encoded_features: Tensor, num_categories: np.ndarray) -> Tensor:
    """
    Convert one-hot encoded categorical data to categorical data.

    Args:
        one_hot_encoded_features: The one-hot encoded categorical data tensor.
        num_categories: The number of categories.

    Returns:
        The categorical data tensor.
    """
    categories = torch.from_numpy(num_categories)
    indices = torch.cat([torch.zeros((1,)), categories.cumsum(dim=0)], dim=0).int().tolist()

    result = []
    for i in range(len(indices) - 1):
        result.append(one_hot_encoded_features[:, indices[i] : indices[i + 1]].argmax(dim=1))

    return torch.stack(result, dim=1)


def log_1_min_a(a: Tensor) -> Tensor:
    """
    Compute the log of 1 minus the exponential of a tensor.

    Args:
        a: The tensor.

    Returns:
        The log of 1 minus the exponential of a tensor.
    """
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute the log of the sum of the exponential of two tensors.

    NOTE: This is a numerically stabilized form of performing this operation.

    Args:
        a: The first tensor.
        b: The second tensor.

    Returns:
        The log of the sum of the exponential of two tensors.
    """
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(input_tensor: Tensor, index: Tensor, output_shape: tuple[int, ...]) -> Tensor:
    """
    Extract the value at ``index`` from a the ``input_tensor``.

    Will return the extracted value as a tensor of shape ``output_shape``
    with the value at ``index`` repeated to fit the shape.

    Args:
        input_tensor: The tensor.
        index: The index of the value to be extracted.
        output_shape: The shape of the output tensor.

    Returns:
        The extracted value as a tensor of shape ``output_shape``.
    """
    index = index.to(input_tensor.device)
    output_tensor = input_tensor.gather(-1, index)
    while len(output_tensor.shape) < len(output_shape):
        # Adding a new dimension to the tensor until it reaches len(output_shape)
        output_tensor = output_tensor[..., None]
    return output_tensor.expand(output_shape)


def log_categorical(log_features_start: Tensor, log_probabilities: Tensor) -> Tensor:
    """
    Compute the expected log-probability under a categorical distribution.

    Args:
        log_features_start: Log of target category probabilities.
        log_probabilities: Log-probabilities over categories aligned with log_x_start.

    Returns:
        Tensor with expected log-probabilities along dim=1.
    """
    return (log_features_start.exp() * log_probabilities).sum(dim=1)


def index_to_log_onehot(input_tensor: Tensor, num_classes: Tensor) -> Tensor:
    """
    Convert the input tensor to one-hot and takes the log of that tensor.

    Will avoid producing NaN values by clamping them to a value just above zero.

    Args:
        input_tensor: The input tensor.
        num_classes: The number of classes.

    Returns:
        The log one-hot tensor.
    """
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(functional.one_hot(input_tensor[:, i], int(num_classes[i])))

    input_onehot = torch.cat(onehots, dim=1)
    return torch.log(input_onehot.float().clamp(min=1e-30))


def log_sum_exp_by_classes(input_tensor: Tensor, classes: Tensor) -> Tensor:
    """
    Compute the log of the sum of the exponential of the input tensor by classes.

    Args:
        input_tensor: The input tensor.
        classes: The classes.

    Returns:
        The log of the sum of the exponential of the input tensor by classes.
    """
    result = torch.zeros_like(input_tensor)
    for c in classes:
        result[:, c] = torch.logsumexp(input_tensor[:, c], dim=1, keepdim=True)

    assert input_tensor.size() == result.size()

    return result


@torch.jit.script
def log_sub_exp(first_tensor: Tensor, second_tensor: Tensor) -> Tensor:
    """
    Compute the log of the difference of the exponential of the input tensor.

    NOTE: This is a numerically stabilized form of performing this operation.

    Args:
        first_tensor: The first tensor.
        second_tensor: The second tensor.

    Returns:
        The log of the difference of the exponential of the input tensor.
    """
    maximum = torch.maximum(first_tensor, second_tensor)
    return torch.log(torch.exp(first_tensor - maximum) - torch.exp(second_tensor - maximum)) + maximum


@torch.jit.script
def sliced_logsumexp(input_tensor: Tensor, slices: Tensor) -> Tensor:
    """
    Compute the log of the sum of the exponential of the input tensor by slices.

    NOTE: Some padding is also being done, maybe investigate this later.

    Args:
        input_tensor: The input tensor.
        slices: The slices.

    Returns:
        The log of the sum of the exponential of the input tensor by slices.
    """
    padded_input_tensor = functional.pad(input_tensor, [1, 0, 0, 0], value=-float("inf"))
    lse = torch.logcumsumexp(padded_input_tensor, dim=-1)

    slice_starts = slices[:-1]
    slice_ends = slices[1:]

    slice_lse = log_sub_exp(lse[:, slice_ends], lse[:, slice_starts])
    return torch.repeat_interleave(slice_lse, slice_ends - slice_starts, dim=-1)


def log_onehot_to_index(log_one_hot_tensor: Tensor) -> Tensor:
    """
    Return the indices of the maximum value in the log one-hot tensor, i.e. the "hot" encoding.

    Args:
        log_one_hot_tensor: The log one-hot tensor.

    Returns:
        The indices of the maximum value in the log one-hot tensor, i.e. the "hot" encoding.
    """
    return log_one_hot_tensor.argmax(1)


class FoundNaNsError(BaseException):
    """Error to be raised whem NANs are found during sampling."""

    def __init__(self, message: str = "Found NANs during sampling."):
        """
        Initialize the FoundNaNsError.

        Args:
            message: The error message. Defaults to "Found NANs during sampling."
        """
        super(FoundNaNsError, self).__init__(message)
