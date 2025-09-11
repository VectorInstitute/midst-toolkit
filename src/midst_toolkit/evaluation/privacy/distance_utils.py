from enum import Enum

import torch


class NormType(Enum):
    L1 = "l1"
    L2 = "l2"


def compute_l1_distance(
    target_data: torch.Tensor, reference_data: torch.Tensor, skip_diagonal: bool = False
) -> torch.Tensor:
    """
    Compute the smallest l1 distance between each point in the target data tensor compared to all points in the
    reference data tensor.

    Args:
        target_data: Tensor of target data. Assumed to be a 2D tensor with batch size first, followed by
            data dimension.
        reference_data: Tensor of reference data. Assumed to be a 2D tensor with batch size first, followed by
            data dimension.
        skip_diagonal: Whether or not to skip computations on diagonal of distance matrix. This is generally only used
            when ``target_data`` and ``reference_data`` are the same set. In this case, the diagonal elements are the
            distance of the point from itself (which is 0). Defaults to False.

    Returns:
        A 1D tensor containing the l1 minimum distances between each data point in the target data and all points in
        the reference data. Order will be the same as the target data.
    """
    assert target_data.ndim == 2 and reference_data.ndim == 2, "Target and Reference data tensors should be 2D"
    assert target_data.shape[1] == reference_data.shape[1], "Data dimensions do not match for the provided tensors"

    # For target_data (n_target_points, data_dim), and reference_data (n_ref_points, data_dim), this subtracts
    # every point in reference_data from every point in target_data to create a tensor of shape
    # (n_target_points, n_ref_points, data_dim).
    point_differences = target_data[:, None] - reference_data
    distances = (point_differences).abs().sum(dim=2)

    # Minimum distance of points in n_target_points compared to all other points in reference_data.
    if not skip_diagonal:
        min_batch_distances, _ = distances.min(dim=1)
        return min_batch_distances

    # Bottom two distances, because one of them might be the reference point to itself.
    min_batch_distances, _ = torch.topk(distances, 2, dim=1, largest=False)
    return min_batch_distances


def compute_l2_distance(
    target_data: torch.Tensor, reference_data: torch.Tensor, skip_diagonal: bool = False
) -> torch.Tensor:
    """
    Compute the smallest l2 distance between each point in the target data tensor compared to all points in the
    reference data tensor.

    Args:
        target_data: Tensor of target data. Assumed to be a 2D tensor with batch size first, followed by
            data dimension.
        reference_data: Tensor of reference data. Assumed to be a 2D tensor with batch size first, followed by
            data dimension.
        skip_diagonal: Whether or not to skip computations on diagonal of distance matrix. This is generally only used
            when ``target_data`` and ``reference_data`` are the same set. In this case, the diagonal elements are the
            distance of the point from itself (which is 0). Defaults to False.

    Returns:
        A 1D tensor containing the l2 minimum distances between each data point in the target data and all points in
        the reference data. Order will be the same as the target data.
    """
    assert target_data.ndim == 2 and reference_data.ndim == 2, "Target and Reference data tensors should be 2D"
    assert target_data.shape[1] == reference_data.shape[1], "Data dimensions do not match for the provided tensors"
    # For target_data (n_target_points, data_dim), and reference_data (n_reference_points, data_dim), this subtracts
    # every point in reference_data from every point in target_data to create a tensor of shape
    # (n_target_points, n_reference_points, data_dim).
    point_differences = target_data[:, None] - reference_data
    distances = torch.sqrt(torch.pow(point_differences, 2.0).sum(dim=2))

    # Minimum distance of points in n_target_points compared to all other points in reference_data.
    if not skip_diagonal:
        min_batch_distances, _ = distances.min(dim=1)
        return min_batch_distances

    # Bottom two distances, because one of them might be the reference point to itself.
    min_batch_distances, _ = torch.topk(distances, 2, dim=1, largest=False)
    return min_batch_distances


def minimum_distances(
    target_data: torch.Tensor,
    reference_data: torch.Tensor,
    batch_size: int | None = None,
    norm: NormType = NormType.L1,
    skip_diagonal: bool = False,
) -> torch.Tensor:
    """
    Function to calculate minimum distances between each point in the target data to those of the reference data
    provided. This can be done in batches if specified. Otherwise, the entire computation is done at once.

    Args:
        target_data: The complete set of target data, stacked as a tensor with shape (n_samples, data dimension).
        reference_data: The complete set of reference data, stacked as a tensor with shape (n_samples, data dimension).
        batch_size: Size of the batches to facilitate computing the minimum distances, if specified. Defaults to None.
        norm: Which type of norm to use as the distance metric. Defaults to NormType.L1.
        skip_diagonal: Whether or not to skip computations on diagonal of distance matrix. This is generally only used
            when ``target_data`` and ``reference_data`` are the same set. In this case, the diagonal elements are the
            distance of the point from itself (which is 0). Defaults to False.

    Returns:
        A 1D tensor with the minimum distances. Should be of length n_samples. Order will be the same as
        ``target_data.``
    """
    if batch_size is None:
        # If batch size isn't specified, do it all at once.
        batch_size = target_data.size(0)

    # Create a minimum distance for each target data sample
    if skip_diagonal:
        min_distances = torch.full((target_data.size(0), 2), float("inf"), device=target_data.device)
    else:
        min_distances = torch.full((target_data.size(0),), float("inf"), device=target_data.device)

    # Iterate through the reference data in batches and compute distances
    for start_index in range(0, reference_data.size(0), batch_size):
        end_index = min(start_index + batch_size, reference_data.size(0))
        reference_data_batch = reference_data[start_index:end_index]

        if norm is NormType.L1:
            min_batch_distances = compute_l1_distance(target_data, reference_data_batch, skip_diagonal)
        elif norm is NormType.L2:
            min_batch_distances = compute_l2_distance(target_data, reference_data_batch, skip_diagonal)
        else:
            raise ValueError(f"Unrecognized norm type: {str(norm)}")
        if not skip_diagonal:
            min_distances = torch.minimum(min_distances, min_batch_distances)
        else:
            combined_distances = torch.cat((min_distances, min_batch_distances), dim=1)
            min_distances, _ = torch.topk(combined_distances, 2, dim=1, largest=False)
    if skip_diagonal:
        # Smallest distance should be point to itself. Second smallest is the rest.
        return min_distances[:, 1]
    return min_distances
