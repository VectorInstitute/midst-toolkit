from enum import Enum

import torch


class NormType(Enum):
    L1 = "l1"
    L2 = "l2"


def compute_l1_distances(target_data: torch.Tensor, reference_data: torch.Tensor) -> torch.Tensor:
    """
    This computes the l1 distances between every point in ``target_data`` to every point in ``reference_data``.
    The final distances are arranged with shape (n_target_points, n_ref_points), where rows correspond the distance
    of a single point in ``target_data`` to all points in ``reference_data``.

    Args:
        target_data: First tensor with shape (n_target_points, data_dim).
        reference_data: Second tensor with shape (n_reference_points, data_dim).

    Returns:
        A matrix with the l1 distances between all points in ``target_data`` to all points in ``reference_data``. Rows
        correspond the distance of a single point in ``target_data`` to all points in ``reference_data``. Order will
        be the same as ``target_data.``
    """
    # For target_data (n_target_points, data_dim), and reference_data (n_ref_points, data_dim), this subtracts
    # every point in reference_data from every point in target_data to create a tensor of shape
    # (n_target_points, n_ref_points, data_dim).
    point_differences = target_data[:, None] - reference_data
    return (point_differences).abs().sum(dim=2)


def compute_l2_distances(target_data: torch.Tensor, reference_data: torch.Tensor) -> torch.Tensor:
    """
    This computes the l2 distances between every point in ``target_data`` to every point in ``reference_data``.
    The final distances are arranged with shape (n_target_points, n_ref_points), where rows correspond the distance
    of a single point in ``target_data`` to all points in ``reference_data``.

    Args:
        target_data: First tensor with shape (n_target_points, data_dim).
        reference_data: Second tensor with shape (n_reference_points, data_dim).

    Returns:
        A matrix with the l2 distances between all points in ``target_data`` to all points in ``reference_data``. Rows
        correspond the distance of a single point in ``target_data`` to all points in ``reference_data``. Order will
        be the same as ``target_data.``
    """
    # For target_data (n_target_points, data_dim), and reference_data (n_reference_points, data_dim), this subtracts
    # every point in reference_data from every point in target_data to create a tensor of shape
    # (n_target_points, n_reference_points, data_dim).
    point_differences = target_data[:, None] - reference_data
    return torch.sqrt(torch.pow(point_differences, 2.0).sum(dim=2))


def compute_top_k_distances(
    target_data: torch.Tensor, reference_data: torch.Tensor, norm: NormType = NormType.L1, top_k: int = 1
) -> torch.Tensor:
    """
    This function computes the ``top_k`` SMALLEST distances for each point in ``target_data`` to points in
    ``reference_data``. A matrix is returned whose rows correspond to the smallest distances from a point in
    ``target_data`` to any points in reference data. The rows are in ascending order and ONLY the distances are
    returned.

    Args:
        target_data: A 2-D tensor with shape (``n_target_datapoints``, ``data_dim``). Each point is compared to all
            points in ``reference_data``.
        reference_data: A 2-D tensor with shape (``n_reference_datapoints``, ``data_dim``). Each point in
            ``target_data`` is compared to all points in this tensor.
        norm: Type of norm to apply when measuring distance between two points. Defaults to NormType.L1.
        top_k: Number of SMALLEST distances to return for each point in ``target_data``. Defaults to 1.

    Raises:
        ValueError: Thrown if the requested distance measure is not supported.

    Returns:
        A matrix of shape (``n_target_datapoints``, ``top_k``). Each row of this tensor corresponds to the SMALLEST
        ``top_k`` distances from a point in ``target_data`` to any point in ``reference_data``. Order will be the same
        as ``target_data.``
    """
    assert target_data.ndim == 2 and reference_data.ndim == 2, "Target and Reference data tensors should be 2D"
    assert target_data.shape[1] == reference_data.shape[1], "Data dimensions do not match for the provided tensors"

    if norm == NormType.L1:
        distances = compute_l1_distances(target_data, reference_data)
    elif norm == NormType.L2:
        distances = compute_l2_distances(target_data, reference_data)
    else:
        raise ValueError(f"Unsupported NormType: {norm.value}")

    # Smallest top_k distances
    top_k_distances, _ = torch.topk(distances, top_k, dim=1, largest=False)
    return top_k_distances


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
        target_data: The complete set of target data, stacked as a tensor with shape (``n_samples``, data dimension).
        reference_data: The complete set of reference data, stacked as a tensor with shape (``n_samples``,
            data dimension).
        batch_size: Size of the batches to facilitate computing the minimum distances, if specified. Defaults to None.
        norm: Which type of norm to use as the distance metric. Defaults to NormType.L1.
        skip_diagonal: Whether or not to skip computations on diagonal of distance matrix. This is generally only used
            when ``target_data`` and ``reference_data`` are the same set. In this case, the diagonal elements are the
            distance of the point from itself (which is always 0). Defaults to False.

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

        if skip_diagonal:
            min_batch_distances = compute_top_k_distances(target_data, reference_data_batch, norm, top_k=2)
            combined_distances = torch.cat((min_distances, min_batch_distances), dim=1)
            min_distances, _ = torch.topk(combined_distances, 2, dim=1, largest=False)

        min_batch_distances = compute_top_k_distances(target_data, reference_data_batch, norm, top_k=1)
        min_distances = torch.minimum(min_distances, min_batch_distances.squeeze())

    if skip_diagonal:
        # Smallest distance should be point to itself. Second smallest is the rest.
        return min_distances[:, 1]
    return min_distances
