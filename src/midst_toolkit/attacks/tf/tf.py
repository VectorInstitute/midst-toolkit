#!/usr/bin/env python
# coding: utf-8

# # Membership Inference over Diffusion-models-based Synthetic Tabular Data (MIDST) Challenge @ SaTML 2025.
#
# This notebook will walk you through the process of how we achieve a good score in **White Box Single Table Competition**.
#
# We only work on **TabDDPM** in this task.
#
# This competition focuses on White Box MIA on tabular diffusion models trained on a single table transaction dataset. The schema of the transaction dataset is as follows:
# | trans_id | account_id | trans_date | trans_type | operation | amount  | balance  | k_symbol | bank | account |
# |----------|------------|------------|------------|-----------|---------|----------|----------|------|---------|
# | integer  | integer    | integer    | integer    | integer   | float   | float    | integer  | integer | integer |
#

# **Package Imports and Evironment Setup**
#
# Ensure that you have installed the proper dependenices to run the notebook. The environment installation instructions are available [here](https://github.com/VectorInstitute/MIDSTModels/tree/main/starter_kits). Now that we have verfied we have the proper packages installed, lets import them and define global variables:

# ## 1. Model preparation
#
# In white box single table competition, we directly use the given models for our membership inferences.
#
# **You do not need to do anything in this section for white box!!!!!**

# ## 2. Loss function
#
# In this part, we define the loss function used in our pipeline. The loss function should maximize the loss difference between training and hold-out samples. As the model may remember training data better, so they should have lower loss between predicted loss and real loss for train data. In this process, we fix the noises to control variables.
#
# Note: We will only list the loss function used finally in this notebook.

# In[32]:

# ======================================================
# 📦 Standard Library
# ======================================================
# ======================================================
# 📝 Notes
# ======================================================
# - All duplicate imports have been removed
# - Imports are logically grouped
# - tqdm.notebook kept separately for Jupyter notebook support
# import warnings
# warnings.filterwarnings("ignore")
import argparse
import csv
import hashlib
import json
import os
from dataclasses import astuple, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

# ======================================================
# 🧪 Third-Party Libraries
# ======================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from midst_models.single_table_TabDDPM.complex_pipeline import (
    CustomUnpickler,
    clava_clustering_force_load,
    load_configs,
)

# ======================================================
# 🧭 Local Modules (Project)
# ======================================================
from midst_models.single_table_TabDDPM.lib import (
    Dataset,
    TaskType,
    Transformations,
    prepare_fast_dataloader,
    transform_dataset,
)
from sklearn.metrics import auc, roc_curve
from torch import nn, optim
from tqdm import tqdm


def get_tpr_at_fpr(true_membership: list, predictions: list, max_fpr=0.1) -> float:
    """Calculates the best True Positive Rate when the False Positive Rate is
    at most `max_fpr`.

    Args:
        true_membership (List): A list of values in {0,1} indicating the membership of a
            challenge point. 0: "non-member", 1: "member".
        predictions (List): A list of values in the range [0,1] indicating the confidence
            that a challenge point is a member. The closer the value to 1, the more
            confident the predictor is about the hypothesis that the challenge point is
            a member.
        max_fpr (float, optional): Threshold on the FPR. Defaults to 0.1.

    Returns:
        float: The TPR @ `max_fpr` FPR.
    """
    fpr, tpr, _ = roc_curve(true_membership, predictions)

    return max(tpr[fpr < max_fpr])


# In[34]:


# we define the loss function here
# global variables are assigned in the main process
def mixed_loss(
    diffusion,
    x,
    out_dict,
    noise=None,
    t=None,
    pt=None,
    return_random=False,
    no_mean=False,
    parallel_batch=None,
    addt_value=None,
):
    # global t_value
    # global input_noise
    # global parallel_batch

    x_num = x[:, : diffusion.num_numerical_features]
    x_cat = x[:, diffusion.num_numerical_features :]
    # if x_num.shape[1] > 0:
    #     if noise is None:
    #         noise = torch.randn_like(x_num)

    # noise = input_noise
    noise_tensor = torch.tensor(noise, device="cuda", dtype=torch.float)
    batch_noise = noise_tensor.repeat(x_num.shape[0], 1)

    # there is actually no categorical classes, as we have examined the DM, so we just ignore x_cat here and later
    x_num = x_num.repeat_interleave(parallel_batch, dim=0)
    x_cat = x_cat.repeat_interleave(parallel_batch, dim=0)

    b = x_num.shape[0]

    log_x_cat_t = x_cat

    device = x.device
    if t is None:
        t, pt = diffusion.sample_time(b, device, "uniform")

    if return_random:
        return noise, t, pt

    # global addt_value
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
# Since raw loss values vary due to noise and different timesteps t_value and add, a simple threshold-based approach is insufficient for robust inference. To address this challenge, we propose a machine-learning-driven approach. Specifically, we introduce a three-layer Multi-Layer Perceptron (MLP) to model the relationship between loss values and membership status, improving attack accuracy.

# ### 3.1 Data Processing
#
# This part include all the source code used during preprocessing and dataset loading. We use these to preprocess the data to fit into diffusion models. Most of these functions are from TabDDPM with minor modifications.

# In[35]:


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


def get_T_dict():
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
    X_num = dataset.X_num
    X_num = {k: num_transform.transform(v) for k, v in X_num.items()}

    if dataset.X_cat is None:
        assert transformations.cat_nan_policy is None
        assert transformations.cat_min_frequency is None
        # assert transformations.cat_encoding is None
        X_cat = None
    else:
        X_cat = cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
        if transformations.cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, transformations.cat_min_frequency)

        if cat_transform is None:
            raise ValueError("See why no cat_tramsform")
        X_cat = {k: cat_transform.transform(v).astype("float32") for k, v in X_cat.items()}
        X_num = X_cat if X_num is None else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
        X_cat = None

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform

    return dataset


def make_dataset_from_df_with_loaded(
    df, T, is_y_cond, ratios=[0.7, 0.2, 0.1], df_info=None, std=0, label_encoders=None, num_transform=None
):
    cat_column_orders = []
    num_column_orders = []
    index_to_column = list(df.columns)
    column_to_index = {col: i for i, col in enumerate(index_to_column)}

    if df_info["n_classes"] > 0:
        X_cat = {} if df_info["cat_cols"] is not None or is_y_cond == "concat" else None
        X_num = {} if df_info["num_cols"] is not None else None
        y = {}

        cat_cols_with_y = []
        if df_info["cat_cols"] is not None:
            cat_cols_with_y += df_info["cat_cols"]
        if is_y_cond == "concat":
            cat_cols_with_y = [df_info["y_col"]] + cat_cols_with_y

        if len(cat_cols_with_y) > 0:
            X_cat["train"] = df[cat_cols_with_y].to_numpy(dtype=np.str_)

        y["train"] = df[df_info["y_col"]].values.astype(np.float32)

        if df_info["num_cols"] is not None:
            X_num["train"] = df[df_info["num_cols"]].values.astype(np.float32)

        cat_column_orders = [column_to_index[col] for col in cat_cols_with_y]
        num_column_orders = [column_to_index[col] for col in df_info["num_cols"]]

    else:
        X_cat = {} if df_info["cat_cols"] is not None else None
        X_num = {} if df_info["num_cols"] is not None or is_y_cond == "concat" else None
        y = {}

        num_cols_with_y = []
        if df_info["num_cols"] is not None:
            num_cols_with_y += df_info["num_cols"]
        if is_y_cond == "concat":
            num_cols_with_y = [df_info["y_col"]] + num_cols_with_y

        if len(num_cols_with_y) > 0:
            X_num["train"] = df[num_cols_with_y].values.astype(np.float32)

        y["train"] = df[df_info["y_col"]].values.astype(np.float32)

        if df_info["cat_cols"] is not None:
            X_cat["train"] = df[df_info["cat_cols"]].to_numpy(dtype=np.str_)

        cat_column_orders = [column_to_index[col] for col in df_info["cat_cols"]]
        num_column_orders = [column_to_index[col] for col in num_cols_with_y]

    column_orders = num_column_orders + cat_column_orders
    column_orders = [index_to_column[index] for index in column_orders]

    if X_cat is not None and len(df_info["cat_cols"]) > 0:
        X_cat_all = X_cat["train"]
        X_cat_converted = []
        for col_index in range(X_cat_all.shape[1]):
            if label_encoders is None:
                raise ValueError("Should be loaded: label_encoder")
            pass

            X_cat_converted.append(label_encoders[col_index].transform(X_cat_all[:, col_index]).astype(float))

            if std > 0:
                # add noise
                X_cat_converted[-1] += np.random.normal(0, std, X_cat_converted[-1].shape)

        X_cat_converted = np.vstack(X_cat_converted).T

        train_num = X_cat["train"].shape[0]

        X_cat["train"] = X_cat_converted[:train_num, :]

        if len(X_num) > 0:
            X_num["train"] = np.concatenate((X_num["train"], X_cat["train"]), axis=1)
        else:
            X_num = X_cat
            X_cat = None

    D = Dataset(
        X_num,
        None,
        y,
        y_info={},
        task_type=TaskType(df_info["task_type"]),
        n_classes=df_info["n_classes"],
    )

    return transform_dataset(D, T, None, num_transform=num_transform), label_encoders, column_orders


def get_dataset(
    data_path, config_path=None, save_dir_tmp=None, train_name="train_with_id.csv", phase=None, batch_size=None
):
    configs, save_dir = load_configs(config_path)
    tables, relation_order, dataset_meta = load_multi_table_customized(
        data_path,
        meta_dir="/h/behnzaman/midst-experiments/deps/TF_attack/midst_models/single_table_TabDDPM/configs",
        train_name=train_name,
    )
    tables, all_group_lengths_prob_dicts = clava_clustering_force_load(tables, relation_order, save_dir, configs)
    # global batch_size
    train_loader_list = []
    for parent, child in relation_order:
        # print(f"Getting {parent} -> {child} model from scratch")
        df_with_cluster = tables[child]["df"]

        id_cols = [col for col in df_with_cluster.columns if "_id" in col]
        df_without_id = df_with_cluster.drop(columns=id_cols)

        child_df_with_cluster, child_domain_dict, parent_name, child_name = (
            df_without_id,
            tables[child]["domain"],
            parent,
            child,
        )
        if parent_name is None:
            y_col = "placeholder"
            child_df_with_cluster["placeholder"] = list(range(len(child_df_with_cluster)))
        else:
            y_col = f"{parent_name}_{child_name}_cluster"
        child_info = get_table_info(child_df_with_cluster, child_domain_dict, y_col)
        child_model_params = get_model_params(
            {
                "d_layers": configs["diffusion"]["d_layers"],
                "dropout": configs["diffusion"]["dropout"],
            }
        )
        child_T_dict = get_T_dict()
        file_path = os.path.join(save_dir_tmp, f"{parent}_{child}_ckpt.pkl")
        with open(file_path, "rb") as f:
            model = CustomUnpickler(f).load()

        diffusion = model["diffusion"].cuda()

        # important, dev and final model is different from train one, so retrive transform from here
        if phase == "train":
            num_transform = model["dataset"].num_transform
        elif phase == "dev" or phase == "final":
            num_transform = model["inverse_transform"].__self__
        else:
            raise ValueError("Unknown Phase!!!")
        T = Transformations(**child_T_dict)

        dataset, label_encoders, column_orders = make_dataset_from_df_with_loaded(
            child_df_with_cluster,
            T,
            is_y_cond=child_model_params["is_y_cond"],
            ratios=[0.99, 0.005, 0.005],
            df_info=child_info,
            std=0,
            label_encoders=model["label_encoders"],
            num_transform=num_transform,
        )
        dataset.X_num["test"] = dataset.X_num["train"]

        if dataset.X_cat is not None:
            dataset.X_cat["test"] = dataset.X_cat["train"]
        dataset.y["test"] = dataset.y["train"]
        train_loader = prepare_fast_dataloader(dataset, split="test", batch_size=batch_size, y_type="long")
        train_loader_list.append([train_loader, dataset.X_num["test"].shape[0], dataset])
    return train_loader_list


# def load_multi_table_customized(data_dir, meta_dir=None, train_name = "train.csv", verbose=True):
#     if meta_dir is None:
#         dataset_meta = json.load(open(os.path.join(data_dir, "dataset_meta.json"), "r"))
#     else:
#         dataset_meta =  json.load(open(os.path.join(meta_dir, "dataset_meta.json"), "r"))

#     relation_order = dataset_meta["relation_order"]
#     relation_order_reversed = relation_order[::-1]

#     tables = {}

#     for table, meta in dataset_meta["tables"].items():
#         if os.path.exists(os.path.join(data_dir, train_name)):
#             # print('exists')
#             train_df = pd.read_csv(os.path.join(data_dir, train_name))
#         else:
#             train_df = pd.read_csv(os.path.join(data_dir, f"{table}.csv"))

#         tables[table] = {
#             "df": train_df,
#             "domain": json.load(open(os.path.join(data_dir, f"{table}_domain.json"))),
#             "children": meta["children"],
#             "parents": meta["parents"],
#         }
#         tables[table]["original_cols"] = list(tables[table]["df"].columns)
#         tables[table]["original_df"] = tables[table]["df"].copy()
#         id_cols = [col for col in tables[table]["df"].columns if "_id" in col]
#         df_no_id = tables[table]["df"].drop(columns=id_cols)
#         info = get_info_from_domain(df_no_id, tables[table]["domain"])
#         data, info = pipeline_process_data(
#             name=table,
#             data_df=df_no_id,
#             info=info,
#             ratio=1,
#             save=False,
#             verbose=verbose,
#         )
#         tables[table]["info"] = info

#     return tables, relation_order, dataset_meta


# ### 3.2 Get score
#
# This part we are trying to get a score based on any input models. The basic logit is to use the loss function before, to get the L2 distances of added noises and predicted noises. We need to use functions in data processing to obtain scores.
# In[36]:
from midst_models.single_table_TabDDPM.pipeline_utils import load_multi_table_customized


# This function is used to get loss values from input data with given diffusion models
# def get_score(data_dir, save_dir, config_path=None, type='tabddpm', phase=None):

#     global challenge_name
#     if type == 'tabddpm':
#         relation_order=[("None", "trans")]
#     elif type == 'tabsyn':
#         raise ValueError("Haven't done it yet!")

#     # load data from the data path
#     train_loader_list = get_dataset(data_dir, config_path, save_dir, train_name=challenge_name, phase=phase)

#     # for tabddpm, relation order only contains like None_trans
#     loader_count = 0
#     global noise_batch_id
#     global parallel_batch

#     for parent, child in relation_order:
#         assert os.path.exists(
#             os.path.join(save_dir, f"{parent}_{child}_ckpt.pkl")
#         )
#         train_loader, iter_max, challenge_dataset = train_loader_list[loader_count]

#         filepath = os.path.join(save_dir, f"{parent}_{child}_ckpt.pkl")

#         # get the diffusion model
#         with open(filepath, "rb") as f:
#             model = CustomUnpickler(f).load()
#         diffusion = model['diffusion'].cuda()

#         device = 'cuda'
#         iter_id = 0
#         global batch_size

#         iter_max = iter_max//batch_size
#         # return_res = torch.zeros([batch_size*parallel_batch, 1])
#         return_res = torch.zeros([batch_size, parallel_batch])
#         assert iter_max == 1
#         iter_id = 0
#         while iter_id < iter_max:

#             x, out_dict = next(train_loader)
#             out_dict = {"y": out_dict}
#             x = x.to(device)
#             for k in out_dict:
#                 out_dict[k] = out_dict[k].long().to(device)

#             # This part we want to fix the random variables noise, t, pt. So they are dealing as gloabl variable
#             global noise
#             global t
#             global t_value
#             global pt

#             # This loss_dataset is an indicator to show statistic information about the training data (And we want to show it only once)
#             global loss_dataset
#             with torch.no_grad():

#                 # get loss here
#                 noise, _, pt =  mixed_loss (diffusion, x, out_dict, return_random=True)
#                 t = _ * 0 + t_value
#                 _, loss = mixed_loss (diffusion, x, out_dict, t=t, noise=noise, pt=pt, no_mean=True)

#             return_res = loss
#             iter_id += 1
#     return return_res


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
    # global challenge_name
    if type == "tabddpm":
        relation_order = [("None", "trans")]
    elif type == "tabsyn":
        raise ValueError("Haven't done it yet!")

    train_loader_list = get_dataset(
        data_path, config_path, save_dir, train_name=challenge_name, phase=phase, batch_size=batch_size
    )

    # for tabddpm, relation order only contains like None_trans
    loader_count = 0
    # global noise_batch_id
    # global parallel_batch

    for parent, child in relation_order:
        assert os.path.exists(os.path.join(save_dir, f"{parent}_{child}_ckpt.pkl"))
        train_loader, iter_max, challenge_dataset = train_loader_list[loader_count]

        filepath = os.path.join(save_dir, f"{parent}_{child}_ckpt.pkl")

        # get the diffusion model
        with open(filepath, "rb") as f:
            model = CustomUnpickler(f).load()
        diffusion = model["diffusion"].cuda()

        device = "cuda"
        iter_id = 0
        # global batch_size

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

            # This part we want to fix the random variables noise, t, pt. So they are dealing as gloabl variable
            # global noise
            # global t
            # global t_value
            # global pt

            # This loss_dataset is an indicator to show statistic information about the training data (And we want to show it only once)
            # global loss_dataset
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
                    pt=pt,
                    no_mean=True,
                    parallel_batch=parallel_batch,
                    addt_value=addt_value,
                )

            return_res = loss
            iter_id += 1
    return return_res


# ### 3.3 Model definition
#
# Here, we define the 3-layer MLP model, and the training function. During training, we also evaluate the model's performances on validation sets periodically (each 10 epochs) (defined as X_test and y_test)

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
        output = torch.sigmoid(self.fc3(residual))
        return output


def custom_loss_fn(model, X, y, fpr_target=0.5):
    confidences = model(X)
    X = X.float()
    y = y.float()
    mse_loss = nn.BCELoss()(confidences, y.unsqueeze(1))
    return mse_loss


# train the model here
def fitmodel(
    regression_model,
    X_train,
    X_label,
    X_test,
    X_label2,
    fpr_target=0.5,
    num_epochs=1000,
    learning_rate=1e-4,
    test_set_ratio=None,
    USE_BEST_CHECKPOINT=None,
):
    # global test_set_ratio

    optimizer = optim.Adam(regression_model.parameters(), lr=learning_rate)

    # -------------------------new data-------------------
    X_train = torch.tensor(X_train, dtype=torch.float32).cuda()
    y_train = torch.tensor(X_label, dtype=torch.float32).cuda()
    # For integration test
    indices = torch.randperm(X_train.size(0))
    X_train = X_train[indices].cuda()
    y_train = y_train[indices].cuda()

    X_train = X_train[indices]
    y_train = y_train[indices]

    X_test = torch.tensor(X_test, dtype=torch.float32).cuda()
    y_test = torch.tensor(X_label2, dtype=torch.float32).cuda()
    #  ---------------------------------------------------

    X_train.requires_grad = True
    y_train.requires_grad = True
    train_loss_res = []
    test_loss_res = []
    train_tpr_res = []
    test_tpr_res = []
    epoch_plot = []
    regression_model.train()
    best_tpr = 0.0
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = custom_loss_fn(regression_model, X_train, y_train, fpr_target)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if (epoch + 1) % 100 == 0:
                train_loss_res.append(loss.item())
                epoch_plot.append(epoch)
                tpr_at_fpr = get_tpr_at_fpr(
                    y_train.detach().cpu().numpy(), regression_model(X_train).detach().cpu().numpy()
                )
                train_tpr_res.append(tpr_at_fpr)

                # if there is validation set
                if test_set_ratio > 0:
                    test_loss = custom_loss_fn(regression_model, X_test, y_test, fpr_target)
                    test_tpr_at_fpr = get_tpr_at_fpr(
                        y_test.detach().cpu().numpy(), regression_model(X_test).detach().cpu().numpy()
                    )
                    test_loss_res.append(test_loss.item())
                    test_tpr_res.append(test_tpr_at_fpr)
                    if test_tpr_at_fpr > best_tpr:
                        best_tpr = test_tpr_at_fpr
                        torch.save(regression_model.state_dict(), "best_model.pt")

                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item()} Test Loss :{test_loss.item()} Train TPR: {tpr_at_fpr} Test TPR: {test_tpr_at_fpr}"
                    )
                else:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item()} Train TPR: {tpr_at_fpr}")
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_plot, train_loss_res, label="Train Loss", color="blue")
    if test_set_ratio > 0:
        plt.plot(epoch_plot, test_loss_res, label="Test Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_plot, train_tpr_res, label="Train TPR", color="green")
    if test_set_ratio > 0:
        plt.plot(epoch_plot, test_tpr_res, label="Test TPR", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("TPR")
    plt.title("Train and Test TPR")
    plt.legend()
    plt.grid(True)
    plt.show()
    if USE_BEST_CHECKPOINT:
        regression_model.load_state_dict(torch.load("best_model.pt"))
    if test_set_ratio > 0:
        test_loss = custom_loss_fn(regression_model, X_test, y_test, fpr_target)
        test_tpr_at_fpr = get_tpr_at_fpr(
            y_test.detach().cpu().numpy(), regression_model(X_test).detach().cpu().numpy()
        )
        print(f"final best loss is {test_loss} best tpr is {test_tpr_at_fpr}")
    return regression_model


# ### 3.4 Data preparation
#
# In this section, we use the functions before to form a complete pipeline.
#
# 1. It starts from tabular data splits and preparation, including training and validation sets from 30 train phase models.
# 2. Then in the src_train phase, it generates a group of scores for each data according to the defined hyperparameters, and train the MLP model.
# 3. After training, the codes iterate each phase [train, dev, final] to predict the score using MLP.

# In[ ]:


# this function is the main pipeline entrance
def up_main_function_process(
    train_indices,
    test_indices,
    regression_model,
    noise_num_sample,
    X_TRAIN,
    X_LABEL,
    X_TEST,
    X_LABEL2,
    df_train_merge,
    df_test_merge,
    DATA_PER_MODEL,
    TEST_DATA_MODEL,
    input_noise,
    test_set_ratio,
    TABDDPM_DATA_DIR,
    phases,
    model_type,
    NEW_MODEL,
    num_epochs,
    t_value_list,
    addt_value_list,
    parallel_batch,
    use_best_checkpoint,
    results_path,
    predictions_file_name=None,
    attack_type="white_box",
    X_FINAL=None,
    X_LABEL3=None,
    final_indices=None,
):
    # global regression_model
    # global noise_num_sample
    # global t_value
    global addt_value
    global sample_num
    # global challenge_label_name

    # noise_count = 0
    # train_noise_count = 0
    train_count = 0
    test_count = 0
    final_count = 0
    n_trained_models_used = len(train_indices)
    # predictions_file_name = prediction_file_format+'_'+str(n_trained_models_used)+'.csv'
    for base_dir, model_type in zip([TABDDPM_DATA_DIR], ["tabddpm"]):
        for phase in phases:
            # src_train phase is for training the MLP
            if phase == "src_train":
                root = os.path.join(base_dir, "train")
            else:
                root = os.path.join(base_dir, "train")

            # global X_TRAIN
            # global X_LABEL
            # global X_TEST
            # global X_LABEL2
            # global challenge_name
            # global batch_size

            # index = 0

            # Prepare the list of folders you actually want to process
            model_folders = sorted(os.listdir(root), key=lambda d: int(d.split("_")[1]))
            model_folders = [
                f
                for f in model_folders
                if int(f.split("_")[1]) in train_indices
                or int(f.split("_")[1]) in test_indices
                or int(f.split("_")[1]) in final_indices
            ]
            # Use tqdm to show progress
            for model_folder in tqdm(model_folders, desc="Processing models", unit="model"):
                model_number = int(model_folder.split("_")[1])

                if attack_type == "white_box":
                    config_path = "/projects/aieng/midst_competition/data/berka/tabddpm/trans.json"
                elif attack_type == "black_box":
                    config_path = os.path.join(
                        os.path.join(root, model_folder), "synthetic_data", "20k", "shadow_config.json"
                    )

                global loss_dataset
                global t
                global pt

                # Reset global varibale for each model
                loss_dataset = False
                t = None
                pt = None
                path = os.path.join(root, model_folder)
                model_path = os.path.join(path, NEW_MODEL)
                # global df_train_merge
                # global df_test_merge
                # global DATA_PER_MODEL
                # global TEST_DATA_MODEL

                if phase == "src_train":
                    # new data collection for training and validation
                    ###########################################################

                    # to train models, collect data to "data.csv"
                    if model_number in train_indices:
                        df_train = pd.read_csv(os.path.join(path, "train_with_id.csv"))

                        # get data not chosen before and not in training set
                        df_exclusive = df_train_merge[
                            ~df_train_merge.set_index(["trans_id", "balance"]).index.isin(
                                df_train.set_index(["trans_id", "balance"]).index
                            )
                        ]
                        # print('ola')
                        # print(df_train.shape)
                        # print(df_exclusive.shape)
                        # print(DATA_PER_MODEL)
                        data_exclusive = df_exclusive.sample(DATA_PER_MODEL)
                        data_from_train = df_train.sample(DATA_PER_MODEL)

                        # store df_data in data.csv
                        df_data = pd.concat([data_exclusive, data_from_train], ignore_index=True)
                        df_data.to_csv(os.path.join(path, "data.csv"), index=False)
                        # print('-*'*100)
                        # print(df_data.shape)

                        # remove chosen data from df_train_merge
                        df_train_merge = df_train_merge[
                            ~df_train_merge.set_index(["trans_id", "balance"]).index.isin(
                                df_data.set_index(["trans_id", "balance"]).index
                            )
                        ]

                    elif model_number in test_indices:
                        df_test = pd.read_csv(os.path.join(path, "train_with_id.csv"))
                        df_exclusive = df_test_merge[
                            ~df_test_merge.set_index(["trans_id", "balance"]).index.isin(
                                df_test.set_index(["trans_id", "balance"]).index
                            )
                        ]

                        data_test_exclusive = df_exclusive.sample(TEST_DATA_MODEL)
                        data_from_test = df_test.sample(TEST_DATA_MODEL)

                        # for store df_data in data.csv
                        df_test_data = pd.concat([data_test_exclusive, data_from_test], ignore_index=True)
                        df_test_data.to_csv(os.path.join(path, "data.csv"), index=False)

                        # remove chosen data from df_test_merge
                        df_test_merge = df_test_merge[
                            ~df_test_merge.set_index(["trans_id", "balance"]).index.isin(
                                df_test_data.set_index(["trans_id", "balance"]).index
                            )
                        ]

                    t_value_count = 0
                    t_value_count = 0
                    for t_value in t_value_list:
                        for addt_value in addt_value_list:
                            if model_number in train_indices:
                                # train sets

                                # define challenge_name (global variable) to make the model access that file
                                challenge_name = "data.csv"
                                # get predictions for these number of data
                                batch_size = DATA_PER_MODEL * 2
                                config_cur = json.load(open(config_path, "r"))
                                config_cur["general"]["workspace_dir"] = path + "/workspace"
                                config_cur["general"]["exp_name"] = "train_1"
                                updated_config_path = Path(path) / "updated_config.json"
                                with open(updated_config_path, "w") as f:
                                    json.dump(config_cur, f, indent=4)
                                predictions = get_score(
                                    path,
                                    model_path,
                                    input_noise,
                                    updated_config_path,
                                    model_type,
                                    phase="train",
                                    challenge_name=challenge_name,
                                    batch_size=batch_size,
                                    parallel_batch=parallel_batch,
                                    addt_value=addt_value,
                                    t_value=t_value,
                                )
                                # print('_'*100)
                                # print(len(predictions))
                                # get_score(data_path, save_dir, input_noise, config_path=None, type='tabddpm', phase=None):
                                # predictions = get_score(path, config_path, model_type, phase="train")

                                # store these losses to the corresponding positions, each data has an array of losses
                                X_TRAIN[
                                    DATA_PER_MODEL * 2 * train_count : DATA_PER_MODEL * 2 * (train_count + 1),
                                    t_value_count * noise_num_sample : (t_value_count + 1) * noise_num_sample,
                                ] = predictions.detach().squeeze().cpu().numpy()

                                # the label is 1 for membership data and 0 for hold-out data
                                X_LABEL[DATA_PER_MODEL * 2 * train_count : DATA_PER_MODEL * 2 * (train_count + 1)] = (
                                    np.concatenate([np.zeros(DATA_PER_MODEL), np.ones(DATA_PER_MODEL)])
                                )
                                t_value_count += 1

                            elif model_number in test_indices:
                                # validation sets
                                challenge_name = "data.csv"
                                batch_size = TEST_DATA_MODEL * 2
                                config_cur = json.load(open(config_path, "r"))
                                config_cur["general"]["workspace_dir"] = path + "/workspace"
                                config_cur["general"]["exp_name"] = "train_1"
                                updated_config_path = Path(path) / "updated_config.json"
                                with open(updated_config_path, "w") as f:
                                    json.dump(config_cur, f, indent=4)
                                # predictions = get_score(path, model_path, updated_config_path, model_type, phase="train", batch_size=batch_size, challenge_name = challenge_name)
                                predictions = get_score(
                                    path,
                                    model_path,
                                    input_noise,
                                    updated_config_path,
                                    model_type,
                                    phase="train",
                                    challenge_name=challenge_name,
                                    batch_size=batch_size,
                                    parallel_batch=parallel_batch,
                                    addt_value=addt_value,
                                    t_value=t_value,
                                )
                                X_TEST[
                                    TEST_DATA_MODEL * 2 * test_count : TEST_DATA_MODEL * 2 * (test_count + 1),
                                    t_value_count * noise_num_sample : (t_value_count + 1) * noise_num_sample,
                                ] = predictions.detach().squeeze().cpu().numpy()

                                X_LABEL2[TEST_DATA_MODEL * 2 * test_count : TEST_DATA_MODEL * 2 * (test_count + 1)] = (
                                    np.concatenate([np.zeros(TEST_DATA_MODEL), np.ones(TEST_DATA_MODEL)])
                                )
                                t_value_count += 1

                            # elif model_number in final_indices:
                            #     # validation sets
                            #     challenge_name = "data.csv"
                            #     batch_size = TEST_DATA_MODEL * 2
                            #     config_cur =  json.load(open(config_path, "r"))
                            #     config_cur["general"]["workspace_dir"] = path +'/workspace'
                            #     config_cur["general"]["exp_name"] = 'train_1'
                            #     updated_config_path = Path(path) / 'updated_config.json'
                            #     with open(updated_config_path, "w") as f:
                            #         json.dump(config_cur, f, indent=4)
                            #     # predictions = get_score(path, model_path, updated_config_path, model_type, phase="train", batch_size=batch_size, challenge_name = challenge_name)
                            #     predictions = get_score(path, model_path, input_noise, updated_config_path, model_type, phase="train", challenge_name = challenge_name,
                            #                             batch_size = batch_size, parallel_batch=parallel_batch, addt_value=addt_value,t_value=t_value)
                            #     X_FINAL[
                            #         TEST_DATA_MODEL * 2 * final_count : TEST_DATA_MODEL * 2 * (final_count + 1),
                            #         t_value_count * noise_num_sample : (t_value_count + 1) * noise_num_sample
                            #     ] = (
                            #         predictions.detach().squeeze().cpu().numpy()
                            #     )

                            #     X_LABEL3[TEST_DATA_MODEL*2*final_count : TEST_DATA_MODEL*2*(final_count+1)] = np.concatenate([np.zeros(TEST_DATA_MODEL), np.ones(TEST_DATA_MODEL)])
                            #     t_value_count += 1

                    # update index to locate the correct places in X_TRAIN (X_LABEL) / X_TEST (X_LABEL2)
                    if model_number in train_indices:
                        train_count += 1
                        # print("train", train_count, index)
                    elif model_number in test_indices:
                        test_count += 1

                    elif model_number in final_indices:
                        final_count += 1
                        # print("test", test_count, )
                    ##########################################################

                else:
                    batch_size = 200
                    challenge_name = "challenge_with_id.csv"
                    t_value_count = 0
                    current_input = []
                    for t_value in t_value_list:
                        for addt_value in addt_value_list:
                            config_cur = json.load(open(config_path, "r"))
                            config_cur["general"]["workspace_dir"] = path + "/workspace"
                            config_cur["general"]["exp_name"] = "train_1"
                            updated_config_path = Path(path) / "updated_config.json"
                            with open(updated_config_path, "w") as f:
                                json.dump(config_cur, f, indent=4)
                            predictions = get_score(
                                path,
                                model_path,
                                input_noise,
                                updated_config_path,
                                model_type,
                                phase=phase,
                                challenge_name=challenge_name,
                                batch_size=batch_size,
                                parallel_batch=parallel_batch,
                                addt_value=addt_value,
                                t_value=t_value,
                            )
                            # predictions = get_score(path, model_path, updated_config_path, model_type, phase=phase)
                            t_value_count += 1
                            current_input = current_input + [predictions]
                    predictions = torch.cat(current_input, dim=-1)

                    predictions = regression_model(predictions).detach().cpu().numpy()
                    # clip to [0, 1]
                    min_output, max_output = np.min(predictions), np.max(predictions)
                    predictions = (predictions - min_output) / (max_output - min_output)
                    predictions = torch.tensor(predictions)

                    assert torch.all((predictions >= 0) & (predictions <= 1))

                    # with open(os.path.join(path, f"prediction_whitebox_{n_trained_models_used}.csv"), mode="w", newline="") as file:
                    with open(os.path.join(path, predictions_file_name), mode="w", newline="") as file:
                        writer = csv.writer(file)

                        # Write each value in a separate row
                        for value in list(predictions.numpy().squeeze()):
                            writer.writerow([value])

            if phase == "src_train":
                # train the model
                global NUM_Epochs
                fitmodel(
                    regression_model,
                    X_TRAIN,
                    X_LABEL,
                    X_TEST,
                    X_LABEL2,
                    num_epochs=num_epochs,
                    test_set_ratio=test_set_ratio,
                    USE_BEST_CHECKPOINT=use_best_checkpoint,
                )

    # evaluate MIA performances in 30 train models
    tpr_at_fpr_list = []
    tpr_at_fpr2_list = []

    for base_dir in [TABDDPM_DATA_DIR]:
        predictions = []
        predictions2 = []
        solutions = []
        root = os.path.join(base_dir, "train")
        global global_noise_count
        global saved_tpr

        for i, model_folder in enumerate(sorted(os.listdir(root), key=lambda d: int(d.split("_")[1]))):
            model_number = int(model_folder.split("_")[1])
            if (
                int(model_number) not in train_indices
                and int(model_number) not in test_indices
                and int(model_number) not in final_indices
            ):
                continue
            path = os.path.join(root, model_folder)
            # predictions.append(np.loadtxt(os.path.join(path, f"prediction_whitebox_{n_trained_models_used}.csv")))
            predictions.append(np.loadtxt(os.path.join(path, predictions_file_name)))
            solutions.append(np.loadtxt(os.path.join(path, "challenge_label.csv"), skiprows=1))
        predictions = np.concatenate(predictions)
        solutions = np.concatenate(solutions)

        tpr_at_fpr = get_tpr_at_fpr(solutions, predictions)
        tpr_at_fpr_list.append(tpr_at_fpr)

    final_tpr_at_fpr = max(tpr_at_fpr_list)
    final_tpr_at_fpr2 = 0
    return final_tpr_at_fpr, final_tpr_at_fpr2


# ======================================================
# Import your own modules here
# ======================================================
# from your_module import MLP, main_function_process, get_tpr_at_fpr

import gc


# import torch


def cleanup_memory():
    # 🧠 Clear PyTorch cache and collected objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_experiment(
    phases,
    base_path,
    tabddpm_data_dir,
    n_synthetic_data_points,
    new_model,
    model_type,
    hidden_dim,
    num_epochs,
    data_per_model,
    test_data_model,
    noise_num,
    repeated_times,
    n_trained_models_list,
    test_model_num,
    t_value_list,
    addt_value_list,
    predictions_file_format,
    challenge_name,
    results_path,
    use_best_checkpoint=True,
    final_indices=None,
    train_indices=None,
    test_indices=None,
    pretrained_MIA_classifer_path=None,
):
    """Main experiment runner."""
    np.random.seed(42)
    input_noise_list = [np.random.normal(size=8).tolist() for _ in range(noise_num)]
    parallel_batch = noise_num
    noise_batch_num = 1
    # test_indices = list(range(11, test_model_num + 11))
    all_results = []

    for n_trained_models_used in n_trained_models_list:
        cleanup_memory()
        print(f"\n{'=' * 60}")
        print(f"🚀 Running experiment with {n_trained_models_used} trained models")
        print(f"{'=' * 60}\n")

        predictions_file_name = f"{predictions_file_format}_{n_trained_models_used}.csv"
        # train_indices = list(range(1, n_trained_models_used + 1))
        all_indices = train_indices + test_indices
        train_model_num = len(train_indices)
        test_model_num = len(test_indices)
        test_set_ratio = float(test_model_num / len(all_indices))

        # --------------------------------------------------
        # Data Merging
        # --------------------------------------------------
        print(train_indices)
        print(base_path)
        df_train_merge = pd.concat(
            [pd.read_csv(os.path.join(base_path + str(t), "train_with_id.csv")) for t in train_indices],
            ignore_index=True,
        ).drop_duplicates(subset=["trans_id", "balance"])

        df_train_challenge = pd.concat(
            [pd.read_csv(os.path.join(base_path + str(t), "challenge_with_id.csv")) for t in test_indices],
            ignore_index=True,
        ).drop_duplicates(subset=["trans_id", "balance"])

        df_train_merge = df_train_merge[
            ~df_train_merge.set_index(["trans_id", "balance"]).index.isin(
                df_train_challenge.set_index(["trans_id", "balance"]).index
            )
        ]

        df_test_merge = pd.concat(
            [pd.read_csv(os.path.join(base_path + str(t), "train_with_id.csv")) for t in test_indices],
            ignore_index=True,
        ).drop_duplicates(subset=["trans_id", "balance"])

        df_test_challenge = pd.concat(
            [pd.read_csv(os.path.join(base_path + str(t), "challenge_with_id.csv")) for t in test_indices],
            ignore_index=True,
        ).drop_duplicates(subset=["trans_id", "balance"])

        df_test_merge = df_test_merge[
            ~df_test_merge.set_index(["trans_id", "balance"]).index.isin(
                df_test_challenge.set_index(["trans_id", "balance"]).index
            )
        ]

        df_final_merge = pd.concat(
            [pd.read_csv(os.path.join(base_path + str(t + 1), "train_with_id.csv")) for t in final_indices],
            ignore_index=True,
        ).drop_duplicates(subset=["trans_id", "balance"])

        df_final_challenge = pd.concat(
            [pd.read_csv(os.path.join(base_path + str(t + 1), "challenge_with_id.csv")) for t in final_indices],
            ignore_index=True,
        ).drop_duplicates(subset=["trans_id", "balance"])

        df_final_merge = df_final_merge[
            ~df_final_merge.set_index(["trans_id", "balance"]).index.isin(
                df_final_challenge.set_index(["trans_id", "balance"]).index
            )
        ]

        print(
            f"🧪 n_trained_models_used={n_trained_models_used} | train merge: {df_train_merge.shape} | test merge: {df_test_merge.shape}"
        )

        # --------------------------------------------------
        # Allocate Arrays
        # --------------------------------------------------
        total_data_num = data_per_model * 2 * train_model_num
        X_TRAIN = np.zeros([total_data_num, noise_num * len(t_value_list) * len(addt_value_list)])
        X_LABEL = np.zeros([total_data_num])

        X_TEST = np.zeros([test_data_model * 2 * test_model_num, noise_num * len(t_value_list) * len(addt_value_list)])
        X_LABEL2 = np.zeros([test_data_model * 2 * test_model_num])

        X_FINAL = np.zeros(
            [test_data_model * 2 * test_model_num, noise_num * len(t_value_list) * len(addt_value_list)]
        )
        X_LABEL3 = np.zeros([test_data_model * 2 * test_model_num])

        # --------------------------------------------------
        # Model Definition
        # --------------------------------------------------
        regression_model = MLP(
            input_dim=noise_num * len(t_value_list) * len(addt_value_list), hidden_dim=hidden_dim
        ).cuda()

        plot_res = []

        # --------------------------------------------------
        # Main Loop
        # --------------------------------------------------
        for noise_batch_id in tqdm(range(noise_batch_num), desc=f"Noise batches ({n_trained_models_used} models)"):
            input_noise = input_noise_list[0 * parallel_batch : (0 + 1) * parallel_batch]
            final_tpr_at_fpr, _ = up_main_function_process(
                train_indices,
                test_indices,
                regression_model,
                noise_num,
                X_TRAIN,
                X_LABEL,
                X_TEST,
                X_LABEL2,
                df_train_merge,
                df_test_merge,
                data_per_model,
                test_data_model,
                input_noise,
                test_set_ratio,
                TABDDPM_DATA_DIR=tabddpm_data_dir,
                phases=phases,
                model_type=model_type,
                NEW_MODEL=new_model,
                num_epochs=num_epochs,
                t_value_list=t_value_list,
                addt_value_list=addt_value_list,
                parallel_batch=parallel_batch,
                use_best_checkpoint=use_best_checkpoint,
                predictions_file_name=predictions_file_name,
                results_path=results_path,
                X_FINAL=X_FINAL,
                X_LABEL3=X_LABEL3,
                final_indices=final_indices,
            )
            plot_res.append(final_tpr_at_fpr)
            print(f"📈 TPR@FPR for batch {noise_batch_id}: {final_tpr_at_fpr}")

        # ======================================================
        # Evaluation
        # ======================================================
        tpr_at_fpr_list = []
        for base_dir in [tabddpm_data_dir]:
            predictions, solutions = [], []
            root = os.path.join(base_dir, "train")
            for model_folder in sorted(os.listdir(root), key=lambda d: int(d.split("_")[1])):
                model_number = int(model_folder.split("_")[1])
                if model_number not in final_indices:
                    continue
                path = os.path.join(root, model_folder)
                predictions.append(np.loadtxt(os.path.join(path, predictions_file_name)))
                solutions.append(np.loadtxt(os.path.join(path, "challenge_label.csv"), skiprows=1))

            predictions = np.concatenate(predictions)
            solutions = np.concatenate(solutions)
            tpr_at_fpr = get_tpr_at_fpr(solutions, predictions)
            tpr_at_fpr_list.append(tpr_at_fpr)
            print(f"{base_dir.split('_')[0]} Train Attack TPR at FPR==10%: {tpr_at_fpr}")

        fpr, tpr, _ = roc_curve(solutions, predictions)
        roc_auc = auc(fpr, tpr)

        all_results.append({"n_trained_models": n_trained_models_used, "max_tpr": tpr_at_fpr, "roc_auc": roc_auc})

        print(all_results)
        os.makedirs(results_path, exist_ok=True)
        plot_filename = f"roc_curve_{n_trained_models_used}_models.png"
        plot_path = os.path.join(results_path, plot_filename)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — White Box-{n_trained_models_used} trained models")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)  # 💾 save figure here
        plt.close()

    results_df = pd.DataFrame(all_results)
    os.makedirs(results_path, exist_ok=True)
    results_path = os.path.join(results_path, "results_summary.csv")
    results_df.to_csv(results_path, index=False)
    print(f"✅ All runs completed. Results saved to {results_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run MIA experiment pipeline.")
    # The phases are basically training model with validation set (test_indices) and then testing on final indices
    parser.add_argument("--phases", nargs="+", default=["src_train", "train"])
    parser.add_argument("--base-path", type=str)
    parser.add_argument("--tabddpm-data-dir", type=str)
    parser.add_argument("--new-model", type=str)
    parser.add_argument("--model-type", type=str)
    parser.add_argument("--n-synthetic-data-points", type=str)
    parser.add_argument("--hidden-dim", type=int, default=200)
    parser.add_argument("--num-epochs", type=int, default=5000)
    parser.add_argument("--data-per-model", type=int, default=3000)
    parser.add_argument("--test-data-model", type=int, default=1000)
    parser.add_argument("--noise-num", type=int, default=300)
    parser.add_argument("--repeated-times", type=int, default=1)
    parser.add_argument("--n-trained-models-list", type=int, nargs="+", default=[10])
    parser.add_argument("--test-model-num", type=int, default=10)
    parser.add_argument("--t-value-list", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50, 100])
    parser.add_argument("--addt-value-list", type=int, nargs="+", default=[0])
    parser.add_argument("--predictions-file-format", type=str, default="prediction_BB_attacker_power")
    parser.add_argument("--challenge-name", type=str, default="challenge_with_id.csv")
    parser.add_argument(
        "--results_path",
        type=str,
        default="/h/behnzaman/midst-experiments/deps/TF_attack/MIA/experiments/attacker_power/attacker_power_BB_datasize_1",
    )
    parser.add_argument("--debug", action="store_true", help="Use hardcoded debug settings")

    return parser.parse_args()


import numpy as np


def apply_debug_defaults(args):
    base_path = (
        "/h/behnzaman/midst-experiments/deps/TF_attack/MIA/experiments/"
        "attacker_access_to_different_distribution_1/different_dist_data_tabddpm"
    )

    # Automatically detect all folders inside base_path
    debug_folders = [
        "tabddpm_access_to_diff_accounts",
        "tabddpm_access_to_diff_times",
        "tabddpm_access_to_diff_cont_cat_dist",
        "tabddpm_access_to_same_marginals",
    ]

    debug_folders = [
        "tabddpm_access_to_same_marginals",
    ]
    print("📁 Debug folders found:", debug_folders)

    # Store for later use
    args.debug_folders = debug_folders
    args.base_path_root = base_path

    # Phases
    args.phases = ["src_train", "train"]

    args.new_model = "synthetic_data/20k/workspace_for_trained_shadow_model/train_1/models"
    args.model_type = "tabddpm"
    args.predictions_file_format = "BB_prediction"

    # How many models to load
    args.n_trained_models_list = [2]

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        print("⚡ DEBUG MODE ENABLED — auto-looping folders")
        args = apply_debug_defaults(args)

        # Loop through each detected subfolder
        for folder_name in args.debug_folders:
            print(f"\n🔁 Running experiment for folder: {folder_name}")

            # Build paths for this folder
            args.tabddpm_data_dir = os.path.join(args.base_path_root, folder_name)
            args.base_path = os.path.join(args.base_path_root, folder_name, "train", "tabddpm_")
            args.results_path = os.path.join(args.base_path_root, "results_BB", folder_name)

            print("  → tabddpm_data_dir:", args.tabddpm_data_dir)
            print("  → base_path:", args.base_path)
            print("  → results_path:", args.results_path)

            # Run experiment
            run_experiment(
                phases=args.phases,
                base_path=args.base_path,
                tabddpm_data_dir=args.tabddpm_data_dir,
                n_synthetic_data_points=args.n_synthetic_data_points,
                new_model=args.new_model,
                model_type=args.model_type,
                hidden_dim=args.hidden_dim,
                num_epochs=args.num_epochs,
                data_per_model=args.data_per_model,
                test_data_model=args.test_data_model,
                noise_num=args.noise_num,
                repeated_times=args.repeated_times,
                n_trained_models_list=args.n_trained_models_list,
                test_model_num=args.test_model_num,
                t_value_list=args.t_value_list,
                addt_value_list=args.addt_value_list,
                predictions_file_format=args.predictions_file_format,
                challenge_name=args.challenge_name,
                results_path=args.results_path,
                use_best_checkpoint=True,
                final_indices=np.arange(21, 31).tolist(),
                train_indices=np.arange(1, 11).tolist(),
                test_indices=np.arange(11, 21).tolist(),
            )

    else:
        # Normal (non-debug) mode
        parallel_batch = args.noise_num
        run_experiment(
            phases=args.phases,
            base_path=args.base_path,
            tabddpm_data_dir=args.tabddpm_data_dir,
            n_synthetic_data_points=args.n_synthetic_data_points,
            new_model=args.new_model,
            model_type=args.model_type,
            hidden_dim=args.hidden_dim,
            num_epochs=args.num_epochs,
            data_per_model=args.data_per_model,
            test_data_model=args.test_data_model,
            noise_num=args.noise_num,
            repeated_times=args.repeated_times,
            n_trained_models_list=args.n_trained_models_list,
            test_model_num=args.test_model_num,
            t_value_list=args.t_value_list,
            addt_value_list=args.addt_value_list,
            predictions_file_format=args.predictions_file_format,
            challenge_name=args.challenge_name,
            results_path=args.results_path,
            use_best_checkpoint=True,
            final_indices=np.arange(21, 31).tolist(),
            train_indices=np.arange(1, 11).tolist(),
            test_indices=np.arange(11, 21).tolist(),
        )
