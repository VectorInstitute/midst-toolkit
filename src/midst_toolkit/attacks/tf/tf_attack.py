import csv
import hashlib
import json
import os
import sys
from dataclasses import astuple, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve
from torch import nn, optim
from tqdm import tqdm
from midst_toolkit.models.clavaddpm.dataset import Dataset

from midst_toolkit.attacks.tf.data_utils import (
    CustomUnpickler,
    clava_clustering_force_load,
    load_configs,
    load_multi_table_customized,
    save_results_and_plot_roc_curve,
    prepare_data_for_attack,
    evaluate_attack_performance,
    get_tpr_at_fpr
)

# noqa: D103
# ======================================================
# 🧭 Local Modules (Project)
# ======================================================
from midst_toolkit.attacks.tf.lib import (
    # Dataset,
    TaskType,
    Transformations,
    prepare_fast_dataloader,
)


sys.path.append("/h/behnzaman/midst-experiments/deps/TF_attack/")





# In[34]:


# we define the loss function here
# global variables are assigned in the main process
def mixed_loss(
    diffusion,
    x,
    out_dict,
    noise=None,
    t=None,
    return_random=False,
    no_mean=False,
    parallel_batch=None,
    addt_value=None,
):
    x_num = x[:, : diffusion.num_numerical_features]
    x_cat = x[:, diffusion.num_numerical_features :]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_tensor = torch.tensor(noise, device=device, dtype=torch.float)
    batch_noise = noise_tensor.repeat(x_num.shape[0], 1)

    # there is actually no categorical classes, as we have examined the DM, so we just ignore x_cat here and later
    x_num = x_num.repeat_interleave(parallel_batch, dim=0)
    x_cat = x_cat.repeat_interleave(parallel_batch, dim=0)

    b = x_num.shape[0]


    device = x.device
    if t is None:
        #the defeualt is uniform sampling 
        t, pt = diffusion.sample_time(b, device)

    if return_random:
        return noise, t, pt

    additional_t = t * 0 + addt_value

    # forward x_num_t with (t+additional_t) timestamps
    x_num_t = diffusion.gaussian_q_sample(x_num, t + additional_t, noise=batch_noise)

    if not return_random:
        current_t = t
        # predict noises with t timestamps
        predicted_noise = diffusion._denoise_fn(x_num_t, current_t, **out_dict)
        current_loss = diffusion._gaussian_loss(predicted_noise, batch_noise, batch_noise, current_t, batch_noise)
        transformed_current_loss = current_loss.reshape(-1, parallel_batch)

    return transformed_current_loss * 0, transformed_current_loss


# ## 3. Loss Processing and Membership Inference
#
# Since raw loss values vary due to noise and different timesteps t_value and add,
# a simple threshold-based approach is insufficient for robust inference. To address this challenge,
# we propose a machine-learning-driven approach. Specifically, we introduce a
# three-layer Multi-Layer Perceptron (MLP) to model the relationship between loss values and membership status,
# improving attack accuracy.

# ### 3.1 Data Processing
#
# This part include all the source code used during preprocessing and dataset loading. We use these
# to preprocess the data to fit into diffusion models.
# Most of these functions are from TabDDPM with minor modifications.



CAT_MISSING_VALUE = "__nan__"
CAT_RARE_VALUE = "__rare__"
Normalization = Literal["standard", "quantile", "minmax"]
NumNanPolicy = Literal["drop-rows", "mean"]
CatNanPolicy = Literal["most_frequent"]
CatEncoding = Literal["one-hot", "counter"]
YPolicy = Literal["default"]
ArrayDict = Dict[str, np.ndarray]


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f"Unknown {unknown_what}: {unknown_value}")


def get_table_info(df, domain_dict, y_col):
    cat_cols = []
    num_cols = []
    for col in df.columns:
        if col in domain_dict and col != y_col:
            if domain_dict[col]["type"] == "discrete":
                cat_cols.append(col)
            else:
                num_cols.append(col)

    df_info = {}
    df_info["cat_cols"] = cat_cols
    df_info["num_cols"] = num_cols
    df_info["y_col"] = y_col
    df_info["n_classes"] = 0
    df_info["task_type"] = "multiclass"

    return df_info


def get_t_dict():
    return {
        "seed": 0,
        "normalization": "quantile",
        "num_nan_policy": None,
        "cat_nan_policy": None,
        "cat_min_frequency": None,
        "cat_encoding": None,
        "y_policy": "default",
    }


def get_model_params(rtdl_params=None):
    return {
        "num_classes": 0,
        "is_y_cond": "none",
        "rtdl_params": {"d_layers": [512, 1024, 1024, 1024, 1024, 512], "dropout": 0.0}
        if rtdl_params is None
        else rtdl_params,
    }


def build_target(y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {"policy": policy}
    if policy is None:
        pass
    elif policy == "default":
        if task_type == TaskType.REGRESSION:
            mean, std = float(y["train"].mean()), float(y["train"].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info["mean"] = mean
            info["std"] = std
    else:
        raise_unknown("policy", policy)
    return y, info


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = "default"


def transform_dataset(
    dataset: Dataset,
    transformations: Transformations,
    cache_dir: Optional[Path],
    transform_cols_num: int = 0,
    normalizer=None,
    cat_transform=None,
    num_transform=None,
) -> Dataset:
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.
    if cache_dir is not None:
        transformations_md5 = hashlib.md5(str(transformations).encode("utf-8")).hexdigest()
        transformations_str = "__".join(map(str, astuple(transformations)))
        cache_path = cache_dir / f"cache__{transformations_str}__{transformations_md5}.pickle"
        if cache_path.exists():
            cache_transformations, value = util.load_pickle(cache_path)
            if transformations == cache_transformations:
                return value
            raise RuntimeError(f"Hash collision for {cache_path}")
    else:
        cache_path = None

    cat_transform = None
    x_num = dataset.X_num
    x_num = {k: num_transform.transform(v) for k, v in x_num.items()}

    if dataset.X_cat is None:
        assert transformations.cat_nan_policy is None
        assert transformations.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        x_cat = None
    else:
        x_cat = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
        if transformations.cat_min_frequency is not None:
            x_cat = cat_drop_rare(x_cat, transformations.cat_min_frequency)

        if cat_transform is None:
            raise ValueError("See why no cat_tramsform")
        x_cat = {k: cat_transform.transform(v).astype("float32") for k, v in x_cat.items()}
        x_num = x_cat if x_num is None else {x: np.hstack([x_num[x], x_cat[x]]) for x in x_num}
        x_cat = None

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=x_num, X_cat=x_cat, y=y, y_info=y_info)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform

    return dataset


# def make_dataset_from_df_with_loaded(
#     df, transformation, is_y_cond,df_info=None, std=0, label_encoders=None, num_transform=None
# ):
#     cat_column_orders = []
#     num_column_orders = []
#     index_to_column = list(df.columns)
#     column_to_index = {col: i for i, col in enumerate(index_to_column)}

#     if df_info["n_classes"] > 0:
#         x_cat = {} if df_info["cat_cols"] is not None or is_y_cond == "concat" else None
#         x_num = {} if df_info["num_cols"] is not None else None
#         y = {}

#         cat_cols_with_y = []
#         if df_info["cat_cols"] is not None:
#             cat_cols_with_y += df_info["cat_cols"]
#         if is_y_cond == "concat":
#             cat_cols_with_y = [df_info["y_col"]] + cat_cols_with_y

#         if len(cat_cols_with_y) > 0:
#             x_cat["train"] = df[cat_cols_with_y].to_numpy(dtype=np.str_)

#         y["train"] = df[df_info["y_col"]].values.astype(np.float32)

#         if df_info["num_cols"] is not None:
#             x_num["train"] = df[df_info["num_cols"]].values.astype(np.float32)

#         cat_column_orders = [column_to_index[col] for col in cat_cols_with_y]
#         num_column_orders = [column_to_index[col] for col in df_info["num_cols"]]

#     else:
#         x_cat = {} if df_info["cat_cols"] is not None else None
#         x_num = {} if df_info["num_cols"] is not None or is_y_cond == "concat" else None
#         y = {}

#         num_cols_with_y = []
#         if df_info["num_cols"] is not None:
#             num_cols_with_y += df_info["num_cols"]
#         if is_y_cond == "concat":
#             num_cols_with_y = [df_info["y_col"]] + num_cols_with_y

#         if len(num_cols_with_y) > 0:
#             x_num["train"] = df[num_cols_with_y].values.astype(np.float32)

#         y["train"] = df[df_info["y_col"]].values.astype(np.float32)

#         if df_info["cat_cols"] is not None:
#             x_cat["train"] = df[df_info["cat_cols"]].to_numpy(dtype=np.str_)

#         cat_column_orders = [column_to_index[col] for col in df_info["cat_cols"]]
#         num_column_orders = [column_to_index[col] for col in num_cols_with_y]

#     column_orders = num_column_orders + cat_column_orders
#     column_orders = [index_to_column[index] for index in column_orders]

#     if x_cat is not None and len(df_info["cat_cols"]) > 0:
#         x_cat_all = x_cat["train"]
#         x_cat_converted = []
#         for col_index in range(x_cat_all.shape[1]):
#             if label_encoders is None:
#                 raise ValueError("Should be loaded: label_encoder")
#             pass

#             x_cat_converted.append(label_encoders[col_index].transform(x_cat_all[:, col_index]).astype(float))

#             if std > 0:
#                 # add noise
#                 x_cat_converted[-1] += np.random.normal(0, std, x_cat_converted[-1].shape)

#         x_cat_converted = np.vstack(x_cat_converted).T

#         train_num = x_cat["train"].shape[0]

#         x_cat["train"] = x_cat_converted[:train_num, :]

#         if len(x_num) > 0:
#             x_num["train"] = np.concatenate((x_num["train"], x_cat["train"]), axis=1)
#         else:
#             x_num = x_cat
#             x_cat = None

#     dataset = Dataset(
#         x_num,
#         None,
#         y,
#         y_info={},
#         task_type=TaskType(df_info["task_type"]),
#         n_classes=df_info["n_classes"],
#     )

#     return transform_dataset(dataset, transformation, None, num_transform=num_transform), label_encoders, column_orders


# def get_dataset(
#     data_path, config_path=None, save_dir_tmp=None, train_name="train_with_id.csv", phase=None, batch_size=None
# ):
#     configs, save_dir = load_configs(config_path)
#     tables, relation_order, dataset_meta = load_multi_table_customized(
#         data_path,
#         meta_dir="/h/behnzaman/midst-experiments/deps/TF_attack/midst_models/single_table_TabDDPM/configs",
#         train_name=train_name,
#     )
#     tables, all_group_lengths_prob_dicts = clava_clustering_force_load(tables, relation_order, save_dir, configs)
#     # global batch_size
#     train_loader_list = []
#     for parent, child in relation_order:
#         # print(f"Getting {parent} -> {child} model from scratch")
#         df_with_cluster = tables[child]["df"]

#         id_cols = [col for col in df_with_cluster.columns if "_id" in col]
#         df_without_id = df_with_cluster.drop(columns=id_cols)

#         child_df_with_cluster, child_domain_dict, parent_name, child_name = (
#             df_without_id,
#             tables[child]["domain"],
#             parent,
#             child,
#         )
#         if parent_name is None:
#             y_col = "placeholder"
#             child_df_with_cluster["placeholder"] = list(range(len(child_df_with_cluster)))
#         else:
#             y_col = f"{parent_name}_{child_name}_cluster"
#         child_info = get_table_info(child_df_with_cluster, child_domain_dict, y_col)
#         child_model_params = get_model_params(
#             {
#                 "d_layers": configs["diffusion"]["d_layers"],
#                 "dropout": configs["diffusion"]["dropout"],
#             }
#         )
#         child_t_dict = get_t_dict()
#         file_path = os.path.join(save_dir_tmp, f"{parent}_{child}_ckpt.pkl")
#         with open(file_path, "rb") as f:
#             model = CustomUnpickler(f).load()

#         # important, dev and final model is different from train one, so retrive transform from here
#         if phase == "train":
#             num_transform = model["dataset"].num_transform
#         elif phase in ("dev", "final"):
#             num_transform = model["inverse_transform"].__self__
#         else:
#             raise ValueError("Unknown Phase!!!")
#         transformations = Transformations(**child_t_dict)

#         dataset, label_encoders, column_orders = make_dataset_from_df_with_loaded(
#             child_df_with_cluster,
#             transformations,
#             is_y_cond=child_model_params["is_y_cond"],
#             df_info=child_info,
#             std=0,
#             label_encoders=model["label_encoders"],
#             num_transform=num_transform,
#         )
#         dataset.X_num["test"] = dataset.X_num["train"]

#         if dataset.X_cat is not None:
#             dataset.X_cat["test"] = dataset.X_cat["train"]
#         dataset.y["test"] = dataset.y["train"]
#         train_loader = prepare_fast_dataloader(dataset, split="test", batch_size=batch_size, y_type="long")
#         train_loader_list.append([train_loader, dataset.X_num["test"].shape[0], dataset])
#     return train_loader_list



def get_dataset(
    data_path, config_path=None, save_dir_tmp=None,
    train_name="train_with_id.csv", phase=None, batch_size=None
):
    configs, save_dir = load_configs(config_path)

    tables, relation_order, dataset_meta = load_multi_table_customized(
        data_path,
        meta_dir="/h/behnzaman/midst-experiments/deps/TF_attack/midst_models/single_table_TabDDPM/configs",
        train_name=train_name,
    )
    tables, all_group_lengths_prob_dicts = clava_clustering_force_load(
        tables, relation_order, save_dir, configs
    )

    train_loader_list = []

    for parent, child in relation_order:

        df_with_cluster = tables[child]["df"]

        # === Remove ID columns ============================
        id_cols = [col for col in df_with_cluster.columns if "_id" in col]
        df_without_id = df_with_cluster.drop(columns=id_cols)

        child_df_with_cluster = df_without_id
        child_domain_dict = tables[child]["domain"]
        parent_name, child_name = parent, child

        # === Construct y-column ============================
        if parent_name is None:
            y_col = "placeholder"
            child_df_with_cluster["placeholder"] = list(range(len(child_df_with_cluster)))
        else:
            y_col = f"{parent_name}_{child_name}_cluster"

        # Metadata needed for feature ordering (still necessary)
        child_info = get_table_info(child_df_with_cluster, child_domain_dict, y_col)

        # Model hyperparameters
        child_model_params = get_model_params(
            {
                "d_layers": configs["diffusion"]["d_layers"],
                "dropout": configs["diffusion"]["dropout"],
            }
        )

        # === Load trained model checkpoint ==================
        file_path = os.path.join(save_dir_tmp, f"{parent}_{child}_ckpt.pkl")

        with open(file_path, "rb") as f:
            model = CustomUnpickler(f).load()

        # === Retrieve fitted transformations =================
        # (instead of constructing Transformations(**child_t_dict))
        transformations = model.transformations

        # === Determine numerical transform depending on phase ==
        # if phase == "train":
        #     num_transform = model.dataset.num_transform
        # elif phase in ("dev", "final"):
        #     num_transform = model.inverse_transform.__self__
        # else:
        #     raise ValueError("Unknown Phase!")

        # === Build Dataset via new API ========================
        dataset, label_encoders, column_orders = Dataset.from_df(
            data=child_df_with_cluster,
            transformations=transformations,
            is_target_conditioned=model.model_parameters.is_target_conditioned,
            data_split_percentages=None,               # manually set test split below
            table_metadata=model.table_metadata,
            noise_scale=0,
        )
        print(type(dataset.numerical_features["train"]))
        # === Create a "test" split identical to "train" ====
        dataset.numerical_features["train"] = np.concatenate(
        [
            dataset.numerical_features["train"],
            dataset.numerical_features["val"],
            dataset.numerical_features["test"]
        ],
        axis=0
        )

        if dataset.categorical_features is not None:
            dataset.categorical_features["train"] = np.concatenate(
                [
                    dataset.categorical_features["train"],
                    dataset.categorical_features["val"],
                    dataset.categorical_features["test"]
                ],
                axis=0
            )

        dataset.target["train"] = np.concatenate(
            [
                dataset.target["train"],
                dataset.target["val"],
                dataset.target["test"]
            ],
            axis=0
        )

        dataset.numerical_features["test"] = dataset.numerical_features["train"].copy()

        if dataset.categorical_features is not None:
            dataset.categorical_features["test"] = dataset.categorical_features["train"].copy()

        dataset.target["test"] = dataset.target["train"].copy()

        # === Build DataLoader ================================
        train_loader = prepare_fast_dataloader(
            dataset, split="test", batch_size=batch_size, y_type="long"
        )

        train_loader_list.append([
            train_loader,
            dataset.numerical_features["test"].shape[0],
            dataset
        ])

    return train_loader_list

def get_score(
    data_path,
    save_dir,
    input_noise,
    config_path=None,
    type="tabddpm",
    phase=None,
    challenge_name=None,
    batch_size=None,
    parallel_batch=None,
    addt_value=None,
    t_value=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type == "tabddpm":
        relation_order = [("None", "trans")]
    elif type == "tabsyn":
        raise ValueError("Haven't done it yet!")

    train_loader_list = get_dataset(
        data_path, config_path, save_dir, train_name=challenge_name, phase=phase, batch_size=batch_size
    )

    """
    Computes the score for a given dataset using a diffusion model.

    Args:
        data_path (str): Path to the dataset.
        save_dir (str): Directory where model checkpoints are saved.
        input_noise (torch.Tensor): Noise tensor to be used in the loss computation.
        config_path (str, optional): Path to the configuration file. Defaults to None.
        type (str, optional): Type of model to use. Defaults to "tabddpm".
        phase (str, optional): Phase of the dataset (e.g., train, test). Defaults to None.
        challenge_name (str, optional): Name of the challenge dataset. Defaults to None.
        batch_size (int, optional): Batch size for data loading. Defaults to None.
        parallel_batch (int, optional): Number of parallel batches for processing. Defaults to None.
        addt_value (Any, optional): Additional value to be passed to the loss function. Defaults to None.
        t_value (Any, optional): Value of the time step `t` to be used in the loss computation. Defaults to None.

    Returns:
        torch.Tensor: A tensor containing the computed loss values.

    Raises:
        ValueError: If the specified `type` is not supported.
        AssertionError: If required model checkpoint files are not found or if `iter_max` is not equal to 1.
    """

    # for tabddpm, relation order only contains like None_trans
    loader_count = 0

    for parent, child in relation_order:
        assert os.path.exists(os.path.join(save_dir, f"{parent}_{child}_ckpt.pkl"))
        train_loader, iter_max, challenge_dataset = train_loader_list[loader_count]

        filepath = os.path.join(save_dir, f"{parent}_{child}_ckpt.pkl")

        # get the diffusion model
        with open(filepath, "rb") as f:
            model = CustomUnpickler(f).load()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diffusion = model.diffusion.to(device)

        iter_id = 0
        print(iter_max)
        iter_max = iter_max // batch_size
        return_res = torch.zeros([batch_size, parallel_batch])
        assert iter_max == 1
        iter_id = 0
        while iter_id < iter_max:
            x, out_dict = next(train_loader)
            out_dict = {"y": out_dict}
            x = x.to(device)
            for k in out_dict:
                out_dict[k] = out_dict[k].long().to(device)

            with torch.no_grad():
                # get loss here
                noise, t_cur, pt = mixed_loss(
                    diffusion,
                    x,
                    out_dict,
                    noise=input_noise,
                    return_random=True,
                    parallel_batch=parallel_batch,
                    addt_value=addt_value,
                )
                t = t_cur * 0 + t_value
                _, loss = mixed_loss(
                    diffusion,
                    x,
                    out_dict,
                    t=t,
                    noise=input_noise,
                    no_mean=True,
                    parallel_batch=parallel_batch,
                    addt_value=addt_value,
                )

            return_res = loss
            iter_id += 1
    return return_res


# ### 3.3 Model definition
#
# Here, we define the 3-layer MLP model, and the training function. During training,
# we also evaluate the model's performances on validation sets periodically (each 10 epochs)
# (defined as x_val and y_test)

# In[37]:


# here we define a MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        residual = torch.tanh(self.fc1(x))
        residual = torch.tanh(self.fc2(residual))
        return torch.sigmoid(self.fc3(residual))


def custom_loss_fn(model, x, y):
    """
    Computes the custom loss for a given model, input, and target.

    This function calculates the Binary Cross-Entropy (BCE) loss between the
    predicted confidences from the model and the target values. The target
    values are unsqueezed to match the shape required by the BCE loss function.

    Args:
        model (torch.nn.Module): The model used to generate predictions.
        x (torch.Tensor): The input tensor to the model.
        y (torch.Tensor): The target tensor containing ground truth values.

    Returns:
        torch.Tensor: The computed BCE loss.
    """
    confidences = model(x)
    x = x.float()
    y = y.float()
    return nn.BCELoss()(confidences, y.unsqueeze(1))


# train the model here
def fitmodel(
    regression_model,
    x_train,
    x_train_label,
    x_val,
    x_val_label,
    num_epochs=1000,
    learning_rate=1e-4,
    use_best_checkpoint=None,
    best_model_dir=None,
):
    """
    Trains a regression model using the provided training and testing data.

    Args:
        regression_model (torch.nn.Module): The regression model to be trained.
        x_train (numpy.ndarray or torch.Tensor): Training input data.
        x_train_label (numpy.ndarray or torch.Tensor): Training labels.
        x_val (numpy.ndarray or torch.Tensor): Testing input data.
        x_val_label (numpy.ndarray or torch.Tensor): Testing labels.
        num_epochs (int, optional): Number of training epochs. Defaults to 1000.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        use_best_checkpoint (bool, optional): Whether to load the best model checkpoint after training. Defaults to None.
        best_model_dir (Path or str, optional): Directory to save the best model checkpoint. Defaults to None.

    Returns:
        torch.nn.Module: The trained regression model.
    """
    pass
    def save_best_model(model, path):
        torch.save(model.state_dict(), path)

    def load_best_model(model, path, device):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)

    def evaluate_model(model, x, y):
        loss = custom_loss_fn(model, x, y)
        tpr = get_tpr_at_fpr(
            y.detach().cpu().numpy(),
            model(x).detach().cpu().numpy(),
        )
        return loss.item(), tpr

    best_model_path = best_model_dir / "best_model.pt"
    optimizer = optim.Adam(regression_model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train = map(lambda t: torch.tensor(t, dtype=torch.float32).to(device), (x_train, x_train_label))
    x_val, y_test = map(lambda t: torch.tensor(t, dtype=torch.float32).to(device), (x_val, x_val_label))

    indices = torch.randperm(x_train.size(0))
    x_train, y_train = x_train[indices], y_train[indices]

    regression_model.train()
    best_tpr, best_model_exists = 0.0, False

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = custom_loss_fn(regression_model, x_train, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            train_loss, train_tpr = evaluate_model(regression_model, x_train, y_train)
            if x_val is not None:
                test_loss, test_tpr = evaluate_model(regression_model, x_val, y_test)
                if test_tpr > best_tpr:
                    best_tpr = test_tpr
                    save_best_model(regression_model, best_model_path)
                    best_model_exists = True
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, "
                    f"Test Loss: {test_loss}, Train TPR: {train_tpr}, Test TPR: {test_tpr}"
                )
            else:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, Train TPR: {train_tpr}")

    if use_best_checkpoint and best_model_exists:
        load_best_model(regression_model, best_model_path, device)

    if  x_val is not None:
        test_loss, test_tpr = evaluate_model(regression_model, x_val, y_test)
        print(f"Final best loss: {test_loss}, best TPR: {test_tpr}")

    return regression_model

# ### 3.4 Data preparation
#
# In this section, we use the functions before to form a complete pipeline.
#
# 1. It starts from tabular data splits and preparation, including training
# and validation sets from 30 train phase models.
# 2. Then in the src_train phase, it generates a group of scores for each
# data according to the defined hyperparameters,
# and train the MLP model.
# 3. After training, the codes iterate each phase [train, dev, final] to predict the score using MLP.



# this function is the main pipeline entrance
def tf_attack(
    train_indices,
    val_indices,
    num_noise_per_time_step,
    samples_per_train_model,
    sample_per_val_model,
    tabddpm_data_dir,
    phases,
    model_type,
    target_model_subdir,
    timesteps_list,
    use_best_checkpoint,
    results_path,
    predictions_file_name=None,
    final_indices=None,
    predictions_file_format=None,
    base_path=None,
    classifier_num_epochs=None,
    classifier_hidden_dim=None,
    addt_value_list = [0],
    config_path=None,
    
): # noqa: C901, D103, PLR0913
    input_noise = [np.random.normal(size=8).tolist() for _ in range(num_noise_per_time_step)]

    predictions_file_name = f"{predictions_file_format}.csv"
    

    # --------------------------------------------------
    # Data Merging
    # --------------------------------------------------
    print(train_indices)
    print(base_path)
    
    df_train_merge, _, _ = prepare_data_for_attack(
        indices=train_indices,
        model_type=model_type,
        models_base_dir=base_path,
        keys_for_deduplication=["trans_id", "balance"],
    )
    
    df_test_merge, _, _= prepare_data_for_attack(
        indices=val_indices,
        model_type=model_type,
        models_base_dir=base_path,
        keys_for_deduplication=["trans_id", "balance"],
    )
    
    
    total_data_num_for_train = samples_per_train_model * 2 * len(train_indices)
    x_train = np.zeros([total_data_num_for_train, len(input_noise) * len(timesteps_list) * len(addt_value_list)])
    x_train_label = np.zeros([total_data_num_for_train])
    

    if val_indices:
        total_data_num_for_validation = sample_per_val_model * 2 * len(val_indices)
        x_val = np.zeros(
            [total_data_num_for_validation, len(input_noise) * len(timesteps_list) * len(addt_value_list)]
        )
        x_val_label = np.zeros([total_data_num_for_validation])
    else:
        x_val, x_val_label = None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regression_model = MLP(input_dim=len(input_noise)  * len(timesteps_list) * len(addt_value_list), hidden_dim=classifier_hidden_dim).to(
        device
    )
    train_count = 0
    test_count = 0
    final_count = 0
    for phase in phases:
            model_folders_indices = np.concatenate((train_indices, val_indices, final_indices))
            for model_number in tqdm(model_folders_indices, desc="Processing models", unit="model"):
                model_folder = f"{model_type}_{model_number}"

                model_dir = tabddpm_data_dir / model_folder
                model_path = model_dir / target_model_subdir

                if phase == "src_train":
                    # to train models, collect data to "data.csv"
                    if model_number in train_indices:
                        df_train = pd.read_csv(os.path.join(model_dir, "train_with_id.csv"))

                        # get data not chosen before and not in training set
                        df_exclusive = df_train_merge[
                            ~df_train_merge.set_index(["trans_id", "balance"]).index.isin(
                                df_train.set_index(["trans_id", "balance"]).index
                            )
                        ]

                        data_exclusive = df_exclusive.sample(samples_per_train_model)
                        data_from_train = df_train.sample(samples_per_train_model)

                        # store df_data in data.csv
                        df_data = pd.concat([data_exclusive, data_from_train], ignore_index=True)
                        df_data.to_csv(os.path.join(model_dir, "data_for_training_MIA.csv"), index=False)

                        # remove chosen data from df_train_merge
                        df_train_merge = df_train_merge[
                            ~df_train_merge.set_index(["trans_id", "balance"]).index.isin(
                                df_data.set_index(["trans_id", "balance"]).index
                            )
                        ]

                    elif model_number in val_indices:
                        df_test = pd.read_csv(os.path.join(model_dir, "train_with_id.csv"))
                        df_exclusive = df_test_merge[
                            ~df_test_merge.set_index(["trans_id", "balance"]).index.isin(
                                df_test.set_index(["trans_id", "balance"]).index
                            )
                        ]

                        data_test_exclusive = df_exclusive.sample(sample_per_val_model)
                        data_from_test = df_test.sample(sample_per_val_model)

                        df_test_data = pd.concat([data_test_exclusive, data_from_test], ignore_index=True)
                        df_test_data.to_csv(os.path.join(model_dir, "data_for_validating_MIA.csv"), index=False)

                        # remove chosen data from df_test_merge
                        df_test_merge = df_test_merge[
                            ~df_test_merge.set_index(["trans_id", "balance"]).index.isin(
                                df_test_data.set_index(["trans_id", "balance"]).index
                            )
                        ]

                    print('###############')
                    print("data train shape", df_data.shape)
                    t_value_count = 0
                    for t_value in timesteps_list:
                        for addt_value in [0]:
                            if model_number in train_indices:
                                # define challenge_name (global variable) to make the model access that file
                                # get predictions for these number of data
                                batch_size = samples_per_train_model * 2

                                with open(config_path, "r") as f:
                                    config_cur = json.load(f)


                                # Make workspace_dir a proper string path
                                config_cur["general"]["workspace_dir"] = str(model_dir / "workspace")

                                config_cur["general"]["exp_name"] = "train_1"

                                updated_config_path = model_dir / "updated_config.json"

                                with open(updated_config_path, "w") as f:
                                    json.dump(config_cur, f, indent=4)

                                predictions = get_score(
                                    model_dir,
                                    model_path,
                                    input_noise,
                                    updated_config_path,
                                    model_type,
                                    phase="train",
                                    challenge_name="data_for_training_MIA.csv",
                                    batch_size=batch_size,
                                    parallel_batch=num_noise_per_time_step,
                                    addt_value=addt_value,
                                    t_value=t_value,
                                )

                                # store these losses to the corresponding positions, each data has an array of losses
                                x_train[
                                    samples_per_train_model * 2 * train_count : samples_per_train_model * 2 * (train_count + 1),
                                    t_value_count * num_noise_per_time_step : (t_value_count + 1) * num_noise_per_time_step,
                                ] = predictions.detach().squeeze().cpu().numpy()

                                # the label is 1 for membership data and 0 for hold-out data
                                x_train_label[samples_per_train_model * 2 * train_count : samples_per_train_model * 2 * (train_count + 1)] = (
                                    np.concatenate([np.zeros(samples_per_train_model), np.ones(samples_per_train_model)])
                                )
                                t_value_count += 1

                            elif model_number in val_indices:
                                # validation sets
                                batch_size = sample_per_val_model * 2
                                with open(config_path, "r") as f:
                                    config_cur = json.load(f)

                                config_cur["general"]["workspace_dir"] = str(model_dir / "workspace")
                                config_cur["general"]["exp_name"] = "train_1"
                                updated_config_path = model_dir / "updated_config.json"
                                with open(updated_config_path, "w") as f:
                                    json.dump(config_cur, f, indent=4)
                                predictions = get_score(
                                    model_dir,
                                    model_path,
                                    input_noise,
                                    updated_config_path,
                                    model_type,
                                    phase="train",
                                    challenge_name="data_for_validating_MIA.csv",
                                    batch_size=batch_size,
                                    parallel_batch=num_noise_per_time_step,
                                    addt_value=addt_value,
                                    t_value=t_value,
                                )
                                x_val[
                                    sample_per_val_model * 2 * test_count : sample_per_val_model * 2 * (test_count + 1),
                                    t_value_count * num_noise_per_time_step : (t_value_count + 1) * num_noise_per_time_step,
                                ] = predictions.detach().squeeze().cpu().numpy()

                                x_val_label[sample_per_val_model * 2 * test_count : sample_per_val_model * 2 * (test_count + 1)] = (
                                    np.concatenate([np.zeros(sample_per_val_model), np.ones(sample_per_val_model)])
                                )
                                t_value_count += 1

                    # update index to locate the correct places in x_train (x_train_label) / x_val (x_val_label)
                    if model_number in train_indices:
                        train_count += 1
                        # print("train", train_count, index)
                    elif model_number in val_indices:
                        test_count += 1

                    elif model_number in final_indices:
                        final_count += 1
                        # print("test", test_count, )
                    ##########################################################

                else:
                    batch_size = 200
                    t_value_count = 0
                    current_input = []
                    for t_value in timesteps_list:
                        for addt_value in [0]:
                            with open(config_path, "r") as f:
                                config_cur = json.load(f)
                            config_cur["general"]["workspace_dir"] = str(model_dir / "workspace")
                            config_cur["general"]["exp_name"] = "train_1"
                            updated_config_path = model_dir / "updated_config.json"
                            with open(updated_config_path, "w") as f:
                                json.dump(config_cur, f, indent=4)
                            predictions = get_score(
                                model_dir,
                                model_path,
                                input_noise,
                                updated_config_path,
                                model_type,
                                phase=phase,
                                challenge_name='challenge_with_id.csv',
                                batch_size=batch_size,
                                parallel_batch=num_noise_per_time_step,
                                addt_value=addt_value,
                                t_value=t_value,
                            )
                            t_value_count += 1
                            current_input = current_input + [predictions]
                    predictions = torch.cat(current_input, dim=-1)

                    predictions = regression_model(predictions).detach().cpu().numpy()
                    # clip to [0, 1]
                    min_output, max_output = np.min(predictions), np.max(predictions)
                    predictions = (predictions - min_output) / (max_output - min_output)
                    predictions = torch.tensor(predictions)

                    assert torch.all((predictions >= 0) & (predictions <= 1))
                    with open(os.path.join(model_dir, predictions_file_name), mode="w", newline="") as file:
                        writer = csv.writer(file)

                        # Write each value in a separate row
                        for value in list(predictions.numpy().squeeze()):
                            writer.writerow([value])

            if phase == "src_train":
                # train the model
                fitmodel(
                    regression_model,
                    x_train,
                    x_train_label,
                    x_val,
                    x_val_label,
                    num_epochs=classifier_num_epochs,
                    use_best_checkpoint=use_best_checkpoint,
                    best_model_dir=results_path,
                )

    MIA_performance_train = evaluate_attack_performance(train_indices, "train", tabddpm_data_dir, model_type,  predictions_file_name)
    MIA_performance_test = evaluate_attack_performance(val_indices, "test", tabddpm_data_dir, model_type,  predictions_file_name)
    MIA_performance_final = evaluate_attack_performance(final_indices, "final", tabddpm_data_dir, model_type,  predictions_file_name)
    print('MIA performance for training set:' )
    print(MIA_performance_train)
    print('MIA performance for test set:' )
    print(MIA_performance_test)
    print('MIA performance for final set:' )
    print(MIA_performance_final)
    
    return MIA_performance_train, MIA_performance_test, MIA_performance_final
