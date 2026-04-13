from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch

from midst_toolkit.common.enumerations import TaskType


@torch.no_grad()
def split_num_cat_target(
    syn_data: np.ndarray,
    info: dict[str, Any],
    num_inverse: Callable,
    cat_inverse: Callable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task_type = info["task_type"]

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == TaskType.REGRESSION.value:
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    pre_decoder = info["pre_decoder"]
    token_dim = info["token_dim"]

    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)

    norm_input = pre_decoder(torch.tensor(syn_data))
    x_hat_num, x_hat_cat = norm_input

    syn_num = x_hat_num.cpu().numpy()
    syn_cat = get_synthetic_categorical_features(x_hat_cat)

    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)

    if info["task_type"] == TaskType.REGRESSION.value:
        syn_target = syn_num[:, : len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx) :]

    else:
        print(syn_cat.shape)
        syn_target = syn_cat[:, : len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx) :]

    return syn_num, syn_cat, syn_target


def get_synthetic_categorical_features(x_hat_cat: torch.Tensor) -> np.ndarray:
    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim=-1))

    return torch.stack(syn_cat).t().cpu().numpy()


def recover_data(
    syn_num: np.ndarray,
    syn_cat: np.ndarray,
    syn_target: np.ndarray,
    info: dict[str, Any],
) -> pd.DataFrame:
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    idx_mapping = info["idx_mapping"]
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info["task_type"] == TaskType.REGRESSION.value:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df
