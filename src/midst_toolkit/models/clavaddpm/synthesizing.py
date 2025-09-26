import os
import pickle
import random
import time

import faiss
import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.spatial.distance import cdist

from midst_toolkit.models.clavaddpm.train import get_df_without_id


def round_columns(X_real, X_synth, columns):
    for col in columns:
        uniq = np.unique(X_real[:, col])
        dist = cdist(
            X_synth[:, col][:, np.newaxis].astype(float),
            uniq[:, np.newaxis].astype(float),
        )
        X_synth[:, col] = uniq[dist.argmin(axis=1)]
    return X_synth


def sample_from_dict(probabilities):
    # Generate a random number between 0 and 1
    random_number = random.random()

    # Initialize cumulative sum and the selected key
    cumulative_sum = 0
    selected_key = None

    # Iterate through the dictionary
    for key, probability in probabilities.items():
        cumulative_sum += probability
        if cumulative_sum >= random_number:
            selected_key = key
            break

    return selected_key


def convert_to_unique_indices(indices):
    occurrence = set()
    max_index = len(indices)  # Assuming the range is the length of the list
    replacement_candidates = set(range(max_index)) - set(indices)

    for i, num in enumerate(tqdm(indices)):
        if num in occurrence:
            # Find the smallest number not in the list
            replacement = min(replacement_candidates)
            indices[i] = replacement
            replacement_candidates.remove(replacement)
        else:
            occurrence.add(num)

    return indices


def match_tables(A, B, n_clusters=25, unique_matching=True, batch_size=100):
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    # Dimension of vectors
    d = B.shape[1]

    if unique_matching:
        quantiser = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantiser, d, n_clusters, faiss.METRIC_L2)
    else:
        res = faiss.StandardGpuResources()
        quantiser = faiss.IndexFlatL2(d)
        index_cpu = faiss.IndexIVFFlat(quantiser, d, n_clusters, faiss.METRIC_L2)
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

    index.train(B)
    index.add(B)

    # Initialize lists to store the results
    all_indices = []
    all_distances = []

    if unique_matching:
        batch_size = 1
        n_batches = (A.shape[0] + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, A.shape[0])
            D, I = index.search(A[start:end], k=1)
            index.remove_ids(I.flatten())
            all_distances.append(D)
            all_indices.append(I)

        # Concatenate the results from all batches
        all_distances = np.vstack(all_distances)
        all_indices = np.vstack(all_indices)
        distances = all_distances.flatten().tolist()
        indices = all_indices.flatten().tolist()
    else:
        n_batches = (A.shape[0] + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, A.shape[0])
            D, I = index.search(A[start:end], k=1)
            all_distances.append(D)
            all_indices.append(I)

        # Concatenate the results from all batches
        all_distances = np.vstack(all_distances)
        all_indices = np.vstack(all_indices)
        distances = all_distances.flatten().tolist()
        indices = all_indices.flatten().tolist()
        indices = convert_to_unique_indices(indices)
        assert len(indices) == len(set(indices))

    return indices, distances


def handle_multi_parent(
    child,
    parents,
    synthetic_tables,
    n_clusters,
    unique_matching=True,
    batch_size=100,
    no_matching=False,
):
    synthetic_child_dfs = [(synthetic_tables[(parent, child)]["df"].copy(), parent) for parent in parents]
    anchor_index = np.argmin([len(df) for df, _ in synthetic_child_dfs])
    anchor = synthetic_child_dfs[anchor_index]
    synthetic_child_dfs.pop(anchor_index)
    for df, parent in synthetic_child_dfs:
        df_without_ids = get_df_without_id(df)
        anchor_df_without_ids = get_df_without_id(anchor[0])
        df_val = df_without_ids.values.astype(float)
        anchor_val = anchor_df_without_ids.values.astype(float)
        if len(df_val.shape) == 1:
            df_val = df_val.reshape(-1, 1)
            anchor_val = anchor_val.reshape(-1, 1)

        indices, _ = match_tables(
            anchor_val,
            df_val,
            n_clusters=n_clusters,
            unique_matching=unique_matching,
            batch_size=batch_size,
        )
        if no_matching:
            # randomly shuffle the array
            indices = np.random.permutation(indices)

        df = df.iloc[indices]
        anchor[0][f"{parent}_id"] = df[f"{parent}_id"].values
    return anchor[0]


def sample_from_diffusion(
    df,
    df_info,
    diffusion,
    dataset,
    label_encoders,
    sample_size,
    model_params,
    T_dict,
    sample_batch_size=8192,
):
    num_numerical_features = dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0

    K = np.array(dataset.get_category_sizes("train"))
    if len(K) == 0 or T_dict["cat_encoding"] == "one-hot":
        K = np.array([0])
    # print(K)

    d_in = np.sum(K) + num_numerical_features
    model_params["d_in"] = d_in
    # print(d_in)
    _, empirical_class_dist = torch.unique(torch.from_numpy(dataset.y["train"]), return_counts=True)
    x_gen, y_gen = diffusion.sample_all(sample_size, sample_batch_size, empirical_class_dist.float(), ddim=False)
    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()
    num_numerical_features_sample = num_numerical_features + int(
        dataset.is_regression and not model_params["is_y_cond"]
    )

    X_num_real = df[df_info["num_cols"]].to_numpy().astype(float)
    X_cat_real = df[df_info["cat_cols"]].to_numpy().astype(str)
    y_real = np.round(df[df_info["y_col"]].to_numpy().astype(float)).astype(int).reshape(-1, 1)

    X_num_ = X_gen

    if num_numerical_features != 0:
        X_num_ = dataset.num_transform.inverse_transform(X_gen[:, :num_numerical_features_sample])
        actual_num_numerical_features = num_numerical_features - len(label_encoders)
        X_num = X_num_[:, :actual_num_numerical_features]
        if len(label_encoders) > 0:
            X_cat = X_num_[:, actual_num_numerical_features:]
            X_cat = np.round(X_cat).astype(int)
            decoded_x_cat = []
            for col in range(X_cat.shape[1]):
                x_cat_col = X_cat[:, col]
                x_cat_col = np.clip(x_cat_col, 0, len(label_encoders[col].classes_) - 1)
                decoded_x_cat.append(label_encoders[col].inverse_transform(x_cat_col))
            X_cat = np.column_stack(decoded_x_cat)
        else:
            X_cat = np.empty((X_num.shape[0], 0))

        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        # print("Discrete cols:", disc_cols)
        if model_params["is_y_cond"] == "concat":
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if disc_cols:
            X_num = round_columns(X_num_real, X_num, disc_cols)

    y_gen = y_gen.reshape(-1, 1)

    if X_cat_real is not None:
        total_real = np.concatenate((X_num_real, X_cat_real, y_real), axis=1)
        gen_real = np.concatenate((X_num, X_cat, np.round(y_gen).astype(int)), axis=1)
    else:
        total_real = np.concatenate((X_num_real, y_real), axis=1)
        gen_real = np.concatenate((X_num, np.round(y_gen).astype(int)), axis=1)

    df_total = pd.DataFrame(total_real)
    df_gen = pd.DataFrame(gen_real)
    columns = [str(x) for x in list(df_total.columns)]

    df_total.columns = columns
    df_gen.columns = columns

    for col in df_total.columns:
        if int(col) < X_num_real.shape[1]:
            df_total[col] = df_total[col].astype(float)
            df_gen[col] = df_gen[col].astype(float)
        elif X_cat_real is not None and int(col) < X_num_real.shape[1] + X_cat_real.shape[1]:
            df_total[col] = df_total[col].astype(str)
            df_gen[col] = df_gen[col].astype(str)
        else:
            df_total[col] = df_total[col].astype(float)
            df_gen[col] = df_gen[col].astype(float)

    return df_total, df_gen


def conditional_sampling_by_group_size(
    df,
    df_info,
    dataset,
    label_encoders,
    classifier,
    diffusion,
    group_labels,
    sample_batch_size,
    group_lengths_prob_dicts,
    is_y_cond,
    classifier_scale,
):
    def cond_fn(x, t, y=None, remove_first_col=False):
        assert y is not None
        with torch.enable_grad():
            if remove_first_col:
                x_in = x[:, 1:].detach().requires_grad_(True).float()
            else:
                x_in = x.detach().requires_grad_(True).float()
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

    sampled_group_sizes = []
    ys = []
    for group_label in group_labels:
        if group_label not in group_lengths_prob_dicts:
            sampled_group_sizes.append(0)
            continue
        sampled_group_size = sample_from_dict(group_lengths_prob_dicts[group_label])
        sampled_group_sizes.append(sampled_group_size)
        ys.extend([group_label] * sampled_group_size)

    all_rows = []
    all_clusters = []
    curr_index = 0
    while curr_index < len(ys):
        end_index = min(curr_index + sample_batch_size, len(ys))
        curr_ys = torch.tensor(np.array(ys[curr_index:end_index]).reshape(-1, 1), requires_grad=False)
        curr_model_kwargs = {}
        curr_model_kwargs["y"] = curr_ys
        curr_sample, _ = diffusion.conditional_sample(ys=curr_ys, model_kwargs=curr_model_kwargs, cond_fn=cond_fn)
        all_rows.extend([sample.cpu().numpy() for sample in [curr_sample]])
        all_clusters.extend([curr_ys.cpu().numpy() for curr_ys in [curr_ys]])
        curr_index += sample_batch_size

    arr = np.concatenate(all_rows, axis=0)
    cluster_arr = np.concatenate(all_clusters, axis=0)

    num_numerical_features = dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0

    X_gen, y_gen = arr, cluster_arr
    num_numerical_features_sample = num_numerical_features + int(dataset.is_regression and not is_y_cond)

    X_num_real = df[df_info["num_cols"]].to_numpy().astype(float)
    X_cat_real = df[df_info["cat_cols"]].to_numpy().astype(str)
    y_real = np.round(df[df_info["y_col"]].to_numpy().astype(float)).astype(int).reshape(-1, 1)

    X_num_ = X_gen

    if num_numerical_features != 0:
        X_num_ = dataset.num_transform.inverse_transform(X_gen[:, :num_numerical_features_sample])
        actual_num_numerical_features = num_numerical_features - len(label_encoders)
        X_num = X_num_[:, :actual_num_numerical_features]
        if len(label_encoders) > 0:
            X_cat = X_num_[:, actual_num_numerical_features:]
            X_cat = np.round(X_cat).astype(int)
            decoded_x_cat = []
            for col in range(X_cat.shape[1]):
                decoded_x_cat.append(label_encoders[col].inverse_transform(X_cat[:, col]))
            X_cat = np.column_stack(decoded_x_cat)

        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        # print("Discrete cols:", disc_cols)
        if is_y_cond == "concat":
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if disc_cols:
            X_num = round_columns(X_num_real, X_num, disc_cols)

    y_gen = y_gen.reshape(-1, 1)

    if X_cat_real is not None and X_cat_real.shape[1] > 0:
        total_real = np.concatenate((X_num_real, X_cat_real, y_real), axis=1)
        gen_real = np.concatenate((X_num, X_cat, np.round(y_gen).astype(int)), axis=1)

    else:
        total_real = np.concatenate((X_num_real, y_real), axis=1)
        gen_real = np.concatenate((X_num, np.round(y_gen).astype(int)), axis=1)

    df_total = pd.DataFrame(total_real)
    df_gen = pd.DataFrame(gen_real)
    columns = [str(x) for x in list(df_total.columns)]

    df_total.columns = columns
    df_gen.columns = columns

    for col in df_total.columns:
        if int(col) < X_num_real.shape[1]:
            df_total[col] = df_total[col].astype(float)
            df_gen[col] = df_gen[col].astype(float)
        elif X_cat_real is not None and int(col) < X_num_real.shape[1] + X_cat_real.shape[1]:
            df_total[col] = df_total[col].astype(str)
            df_gen[col] = df_gen[col].astype(str)
        else:
            df_total[col] = df_total[col].astype(float)
            df_gen[col] = df_gen[col].astype(float)

    return df_total, df_gen, sampled_group_sizes


def clava_synthesizing(
    tables,
    relation_order,
    save_dir,
    all_group_lengths_prob_dicts,
    models,
    configs,
    sample_scale=1,
):
    synthesizing_start_time = time.time()
    synthetic_tables = {}

    # Synthesize
    for parent, child in relation_order:
        print(f"Generating {parent} -> {child}")
        result = models[(parent, child)]
        df_with_cluster = tables[child]["df"]
        df_without_id = get_df_without_id(df_with_cluster)

        print("Sample size: {}".format(int(sample_scale * len(df_without_id))))

        if parent is None:
            _, child_generated = sample_from_diffusion(
                df=df_without_id,
                df_info=result["df_info"],
                diffusion=result["diffusion"],
                dataset=result["dataset"],
                label_encoders=result["label_encoders"],
                sample_size=int(sample_scale * len(df_without_id)),
                model_params=result["model_params"],
                T_dict=result["T_dict"],
                sample_batch_size=configs["sampling"]["batch_size"],
            )
            child_keys = list(range(len(child_generated)))
            generated_final_arr = np.concatenate(
                [np.array(child_keys).reshape(-1, 1), child_generated.to_numpy()],
                axis=1,
            )
            generated_final_df = pd.DataFrame(
                generated_final_arr,
                columns=[f"{child}_id"]
                + result["df_info"]["num_cols"]
                + result["df_info"]["cat_cols"]
                + [result["df_info"]["y_col"]],
            )
            # generated_final_df = generated_final_df[tables[child]['df'].columns]
            generated_final_df = generated_final_df[[f"{child}_id"] + df_without_id.columns.tolist()]
            synthetic_tables[(parent, child)] = {
                "df": generated_final_df,
                "keys": child_keys,
            }
        else:
            for key, val in synthetic_tables.items():
                if key[1] == parent:
                    parent_synthetic_df = val["df"]
                    parent_keys = val["keys"]
                    parent_result = models[key]
                    break

            child_result = models[(parent, child)]
            parent_label_index = parent_result["column_orders"].index(child_result["df_info"]["y_col"])

            parent_synthetic_df_without_id = get_df_without_id(parent_synthetic_df)

            (
                _,
                child_generated,
                child_sampled_group_sizes,
            ) = conditional_sampling_by_group_size(
                df=df_without_id,
                df_info=child_result["df_info"],
                dataset=child_result["dataset"],
                label_encoders=child_result["label_encoders"],
                classifier=child_result["classifier"],
                diffusion=child_result["diffusion"],
                group_labels=parent_synthetic_df_without_id.values[:, parent_label_index]
                .astype(float)
                .astype(int)
                .tolist(),
                group_lengths_prob_dicts=all_group_lengths_prob_dicts[(parent, child)],
                sample_batch_size=configs["sampling"]["batch_size"],
                is_y_cond="none",
                classifier_scale=configs["sampling"]["classifier_scale"],
            )

            child_foreign_keys = np.repeat(parent_keys, child_sampled_group_sizes, axis=0).reshape((-1, 1))
            child_foreign_keys_arr = np.array(child_foreign_keys).reshape(-1, 1)
            child_primary_keys_arr = np.arange(len(child_generated)).reshape(-1, 1)

            child_generated_final_arr = np.concatenate(
                [
                    child_primary_keys_arr,
                    child_generated.to_numpy(),
                    child_foreign_keys_arr,
                ],
                axis=1,
            )

            child_final_columns = (
                [f"{child}_id"]
                + result["df_info"]["num_cols"]
                + result["df_info"]["cat_cols"]
                + [result["df_info"]["y_col"]]
                + [f"{parent}_id"]
            )

            child_final_df = pd.DataFrame(child_generated_final_arr, columns=child_final_columns)
            original_columns = []
            for col in tables[child]["df"].columns:
                if col in child_final_df.columns:
                    original_columns.append(col)
            child_final_df = child_final_df[original_columns]
            synthetic_tables[(parent, child)] = {
                "df": child_final_df,
                "keys": child_primary_keys_arr.flatten().tolist(),
            }
        pickle.dump(
            synthetic_tables,
            open(os.path.join(save_dir, "before_matching/synthetic_tables.pkl"), "wb"),
        )

    synthesizing_end_time = time.time()
    synthesizing_time_spent = synthesizing_end_time - synthesizing_start_time

    matching_start_time = time.time()

    # Matching
    final_tables = {}
    for parent, child in relation_order:
        if child not in final_tables:
            if len(tables[child]["parents"]) > 1:
                final_tables[child] = handle_multi_parent(
                    child,
                    tables[child]["parents"],
                    synthetic_tables,
                    configs["matching"]["num_matching_clusters"],
                    unique_matching=configs["matching"]["unique_matching"],
                    batch_size=configs["matching"]["matching_batch_size"],
                    no_matching=configs["matching"]["no_matching"],
                )
            else:
                final_tables[child] = synthetic_tables[(parent, child)]["df"]

    matching_end_time = time.time()
    matching_time_spent = matching_end_time - matching_start_time

    cleaned_tables = {}
    for key, val in final_tables.items():
        if "account_id" in tables[key]["original_cols"]:
            cols = tables[key]["original_cols"]
            cols.remove("account_id")
        else:
            cols = tables[key]["original_cols"]
        cleaned_tables[key] = val[cols]

    for key, val in cleaned_tables.items():
        table_dir = os.path.join(
            configs["general"]["workspace_dir"],
            configs["general"]["exp_name"],
            key,
            f"{configs['general']['sample_prefix']}_final",
        )
        os.makedirs(table_dir, exist_ok=True)
        if f"{key}_id" in val.columns:
            val.to_csv(os.path.join(table_dir, f"{key}_synthetic_with_id.csv"), index=False)
            val_no_id = val.drop(columns=[f"{key}_id"])
            val_no_id.to_csv(os.path.join(table_dir, f"{key}_synthetic.csv"), index=False)
        else:
            val.to_csv(os.path.join(table_dir, f"{key}_synthetic.csv"), index=False)
    return cleaned_tables, synthesizing_time_spent, matching_time_spent
