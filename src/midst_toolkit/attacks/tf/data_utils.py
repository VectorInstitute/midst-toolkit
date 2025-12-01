import io
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Fix module renaming
        if module.startswith("midst_competition.single_table_ClavaDDPM"):
            module = module.replace("midst_competition.single_table_ClavaDDPM", "midst_models.single_table_TabDDPM", 1)

        # Force CUDA tensors to load on CPU
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")

        return super().find_class(module, name)


def clava_clustering_force_load(tables, relation_order, save_dir, configs):
    relation_order_reversed = relation_order[::-1]
    all_group_lengths_prob_dicts = {}

    for parent, child in relation_order_reversed:
        if parent is not None:
            print(f"Clustering {parent} -> {child}")
            if isinstance(configs["clustering"]["num_clusters"], dict):
                num_clusters = configs["clustering"]["num_clusters"][child]
            else:
                num_clusters = configs["clustering"]["num_clusters"]
            (
                parent_df_with_cluster,
                child_df_with_cluster,
                group_lengths_prob_dicts,
            ) = pair_clustering_keep_id(
                tables[child]["df"],
                tables[child]["domain"],
                tables[parent]["df"],
                tables[parent]["domain"],
                f"{child}_id",
                f"{parent}_id",
                num_clusters,
                configs["clustering"]["parent_scale"],
                1,  # not used for now
                parent,
                child,
                clustering_method=configs["clustering"]["clustering_method"],
            )
            tables[parent]["df"] = parent_df_with_cluster
            tables[child]["df"] = child_df_with_cluster
            all_group_lengths_prob_dicts[(parent, child)] = group_lengths_prob_dicts

    for parent, child in relation_order:
        if parent is None:
            tables[child]["df"]["placeholder"] = list(range(len(tables[child]["df"])))

    return tables, all_group_lengths_prob_dicts


def load_configs(config_path):
    configs = json.load(open(config_path, "r"))

    save_dir = os.path.join(configs["general"]["workspace_dir"], configs["general"]["exp_name"])
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "before_matching"), exist_ok=True)

    with open(os.path.join(save_dir, "args"), "w") as file:
        json.dump(configs, file, indent=4)

    return configs, save_dir


def load_multi_table_customized(data_dir, meta_dir=None, train_name="train.csv", verbose=True):
    if meta_dir is None:
        dataset_meta = json.load(open(os.path.join(data_dir, "dataset_meta.json"), "r"))
    else:
        dataset_meta = json.load(open(os.path.join(meta_dir, "dataset_meta.json"), "r"))

    relation_order = dataset_meta["relation_order"]
    relation_order_reversed = relation_order[::-1]

    tables = {}

    for table, meta in dataset_meta["tables"].items():
        # debug
        if os.path.exists(os.path.join(data_dir, train_name)):
            train_df = pd.read_csv(os.path.join(data_dir, train_name))
        else:
            train_df = pd.read_csv(os.path.join(data_dir, f"{table}.csv"))

        tables[table] = {
            "df": train_df,
            "domain": json.load(open(os.path.join(data_dir, f"{table}_domain.json"))),
            "children": meta["children"],
            "parents": meta["parents"],
        }
        tables[table]["original_cols"] = list(tables[table]["df"].columns)
        tables[table]["original_df"] = tables[table]["df"].copy()
        id_cols = [col for col in tables[table]["df"].columns if "_id" in col]
        df_no_id = tables[table]["df"].drop(columns=id_cols)
        info = get_info_from_domain(df_no_id, tables[table]["domain"])

        # Columns containing '?'
        qmark_cols = (df_no_id == "?").any()
        if qmark_cols.any():
            bad_cols = qmark_cols[qmark_cols].index.tolist()
            raise ValueError(f"Invalid values '?' detected in columns {bad_cols} of table '{table}'.")

        # Numeric columns containing strings
        num_cols = df_no_id.select_dtypes(include=["number"]).columns
        for col in num_cols:
            if df_no_id[col].map(lambda v: isinstance(v, str)).any():
                raise TypeError(f"Numeric column '{col}' in table '{table}' contains string values.")

        data, info = pipeline_process_data(
            name=table,
            data_df=df_no_id,
            info=info,
            ratio=1,
            save=False,
            verbose=verbose,
        )
        tables[table]["info"] = info

    return tables, relation_order, dataset_meta


def get_info_from_domain(data_df, domain_dict):
    info = {}
    info["num_col_idx"] = []
    info["cat_col_idx"] = []
    columns = data_df.columns.tolist()
    for i in range(len(columns)):
        if domain_dict[columns[i]]["type"] == "discrete":
            info["cat_col_idx"].append(i)
        else:
            info["num_col_idx"].append(i)

    info["target_col_idx"] = []
    info["task_type"] = "None"
    info["column_names"] = columns

    return info


def pipeline_process_data(name, data_df, info, ratio=0.9, save=False, verbose=True):
    num_data = data_df.shape[0]

    column_names = info["column_names"] if info["column_names"] else data_df.columns.tolist()

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
        data_df, num_col_idx, cat_col_idx, target_col_idx, column_names
    )

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]

    # Train/ Test Split, 90% Training, 10% Testing (Validation set will be selected from Training set)
    num_train = int(num_data * ratio)
    num_test = num_data - num_train

    if ratio < 1:
        train_df, test_df, seed = train_val_test_split(data_df, cat_columns, num_train, num_test)
    else:
        train_df = data_df.copy()

    train_df.columns = range(len(train_df.columns))

    if ratio < 1:
        test_df.columns = range(len(test_df.columns))

    col_info = {}

    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info["type"] = "numerical"
        col_info["max"] = float(train_df[col_idx].max())
        col_info["min"] = float(train_df[col_idx].min())

    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info["type"] = "categorical"
        col_info["categorizes"] = list(set(train_df[col_idx]))

    for col_idx in target_col_idx:
        if info["task_type"] == "regression":
            col_info[col_idx] = {}
            col_info["type"] = "numerical"
            col_info["max"] = float(train_df[col_idx].max())
            col_info["min"] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info["type"] = "categorical"
            col_info["categorizes"] = list(set(train_df[col_idx]))

    info["column_info"] = col_info

    train_df.rename(columns=idx_name_mapping, inplace=True)
    if ratio < 1:
        test_df.rename(columns=idx_name_mapping, inplace=True)

    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()

    if ratio < 1:
        X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
        X_cat_test = test_df[cat_columns].to_numpy()
        y_test = test_df[target_columns].to_numpy()

    if save:
        save_dir = f"data/{name}"
        np.save(f"{save_dir}/X_num_train.npy", X_num_train)
        np.save(f"{save_dir}/X_cat_train.npy", X_cat_train)
        np.save(f"{save_dir}/y_train.npy", y_train)

        if ratio < 1:
            np.save(f"{save_dir}/X_num_test.npy", X_num_test)
            np.save(f"{save_dir}/X_cat_test.npy", X_cat_test)
            np.save(f"{save_dir}/y_test.npy", y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)

    if ratio < 1:
        test_df[num_columns] = test_df[num_columns].astype(np.float32)

    if save:
        train_df.to_csv(f"{save_dir}/train.csv", index=False)

        if ratio < 1:
            test_df.to_csv(f"{save_dir}/test.csv", index=False)

        if not os.path.exists(f"synthetic/{name}"):
            os.makedirs(f"synthetic/{name}")

        train_df.to_csv(f"synthetic/{name}/real.csv", index=False)

        if ratio < 1:
            test_df.to_csv(f"synthetic/{name}/test.csv", index=False)

    info["column_names"] = column_names
    info["train_num"] = train_df.shape[0]

    if ratio < 1:
        info["test_num"] = test_df.shape[0]

    info["idx_mapping"] = idx_mapping
    info["inverse_idx_mapping"] = inverse_idx_mapping
    info["idx_name_mapping"] = idx_name_mapping

    metadata = {"columns": {}}
    task_type = info["task_type"]
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    for i in num_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "numerical"
        metadata["columns"][i]["computer_representation"] = "Float"

    for i in cat_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "categorical"

    if task_type == "regression":
        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "numerical"
            metadata["columns"][i]["computer_representation"] = "Float"

    else:
        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "categorical"

    info["metadata"] = metadata

    if save:
        with open(f"{save_dir}/info.json", "w") as file:
            json.dump(info, file, indent=4)

    if verbose:
        if ratio < 1:
            str_shape = "Train dataframe shape: {}, Test dataframe shape: {}, Total dataframe shape: {}".format(
                train_df.shape, test_df.shape, data_df.shape
            )
        else:
            str_shape = "Table name: {}, Total dataframe shape: {}".format(name, data_df.shape)

        str_shape += ", Numerical data shape: {}".format(X_num_train.shape)
        str_shape += ", Categorical data shape: {}".format(X_cat_train.shape)

    data = {
        "df": {"train": train_df},
        "numpy": {
            "X_num_train": X_num_train,
            "X_cat_train": X_cat_train,
            "y_train": y_train,
        },
    }

    if ratio < 1:
        data["df"]["test"] = test_df
        data["numpy"]["X_num_test"] = X_num_test
        data["numpy"]["X_cat_test"] = X_cat_test
        data["numpy"]["y_test"] = y_test

    return data, info


def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names=None):
    if not column_names:
        column_names = np.array(data_df.columns.tolist())

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):
        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k

    idx_name_mapping = {}

    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train=0, num_test=0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)

    seed = 1234

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]

        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        seed += 1

    return train_df, test_df, seed


def pair_clustering_keep_id(
    child_df,
    child_domain_dict,
    parent_df,
    parent_domain_dict,
    child_primary_key,
    parent_primary_key,
    num_clusters,
    parent_scale,
    key_scale,
    parent_name,
    child_name,
    clustering_method="kmeans",
):
    original_child_cols = list(child_df.columns)
    original_parent_cols = list(parent_df.columns)

    relation_cluster_name = f"{parent_name}_{child_name}_cluster"

    child_data = child_df.to_numpy()
    parent_data = parent_df.to_numpy()

    child_num_cols = []
    child_cat_cols = []

    parent_num_cols = []
    parent_cat_cols = []

    for col_index, col in enumerate(original_child_cols):
        if col in child_domain_dict:
            if child_domain_dict[col]["type"] == "discrete":
                child_cat_cols.append((col_index, col))
            else:
                child_num_cols.append((col_index, col))

    for col_index, col in enumerate(original_parent_cols):
        if col in parent_domain_dict:
            if parent_domain_dict[col]["type"] == "discrete":
                parent_cat_cols.append((col_index, col))
            else:
                parent_num_cols.append((col_index, col))

    parent_primary_key_index = original_parent_cols.index(parent_primary_key)
    foreing_key_index = original_child_cols.index(parent_primary_key)

    # sort child data by foreign key
    sorted_child_data = child_data[np.argsort(child_data[:, foreing_key_index])]
    child_group_data_dict = get_group_data_dict(
        sorted_child_data,
        [
            foreing_key_index,
        ],
    )

    # sort parent data by primary key
    sorted_parent_data = parent_data[np.argsort(parent_data[:, parent_primary_key_index])]

    group_lengths = []
    unique_group_ids = sorted_parent_data[:, parent_primary_key_index]
    for group_id in unique_group_ids:
        group_id = tuple([group_id])
        if group_id not in child_group_data_dict:
            group_lengths.append(0)
        else:
            group_lengths.append(len(child_group_data_dict[group_id]))

    group_lengths = np.array(group_lengths, dtype=int)

    sorted_parent_data_repeated = np.repeat(sorted_parent_data, group_lengths, axis=0)
    assert (sorted_parent_data_repeated[:, parent_primary_key_index] == sorted_child_data[:, foreing_key_index]).all()

    child_group_data = get_group_data(
        sorted_child_data,
        [
            foreing_key_index,
        ],
    )

    sorted_child_num_data = sorted_child_data[:, [col_index for col_index, col in child_num_cols]]
    sorted_child_cat_data = sorted_child_data[:, [col_index for col_index, col in child_cat_cols]]
    sorted_parent_num_data = sorted_parent_data_repeated[:, [col_index for col_index, col in parent_num_cols]]
    sorted_parent_cat_data = sorted_parent_data_repeated[:, [col_index for col_index, col in parent_cat_cols]]

    joint_num_matrix = np.concatenate([sorted_child_num_data, sorted_parent_num_data], axis=1)
    joint_cat_matrix = np.concatenate([sorted_child_cat_data, sorted_parent_cat_data], axis=1)

    if joint_cat_matrix.shape[1] > 0:
        joint_cat_matrix_p_index = sorted_child_cat_data.shape[1]
        joint_num_matrix_p_index = sorted_child_num_data.shape[1]

        cat_converted = []
        label_encoders = []
        for i in range(joint_cat_matrix.shape[1]):
            # A threshold of 1000 unique values is used to prevent the one-hot encoding of large categorical columns
            if len(np.unique(joint_cat_matrix[:, i])) > 1000:
                continue
            label_encoder = LabelEncoder()
            cat_converted.append(label_encoder.fit_transform(joint_cat_matrix[:, i]).astype(float))
            label_encoders.append(label_encoder)

        cat_converted = np.vstack(cat_converted).T

        # Initialize an empty array to store the encoded values
        cat_one_hot = np.empty((cat_converted.shape[0], 0))

        # Loop through each column in the data and encode it
        for col in range(cat_converted.shape[1]):
            encoder = OneHotEncoder(sparse_output=False)
            column = cat_converted[:, col].reshape(-1, 1)
            encoded_column = encoder.fit_transform(column)
            cat_one_hot = np.concatenate((cat_one_hot, encoded_column), axis=1)

        cat_one_hot[:, joint_cat_matrix_p_index:] = parent_scale * cat_one_hot[:, joint_cat_matrix_p_index:]

    # Perform quantile normalization using QuantileTransformer
    num_quantile = quantile_normalize_sklearn(joint_num_matrix)
    num_min_max = min_max_normalize_sklearn(joint_num_matrix)

    key_quantile = quantile_normalize_sklearn(sorted_parent_data_repeated[:, parent_primary_key_index].reshape(-1, 1))
    key_min_max = min_max_normalize_sklearn(sorted_parent_data_repeated[:, parent_primary_key_index].reshape(-1, 1))

    # key_scaled = key_scaler * key_quantile
    key_scaled = key_scale * key_min_max

    num_quantile[:, joint_num_matrix_p_index:] = parent_scale * num_quantile[:, joint_num_matrix_p_index:]
    num_min_max[:, joint_num_matrix_p_index:] = parent_scale * num_min_max[:, joint_num_matrix_p_index:]

    if joint_cat_matrix.shape[1] > 0:
        cluster_data = np.concatenate((num_min_max, cat_one_hot, key_scaled), axis=1)
    else:
        cluster_data = np.concatenate((num_min_max, key_scaled), axis=1)

    child_group_lengths = np.array([len(group) for group in child_group_data], dtype=int)
    num_clusters = min(num_clusters, len(cluster_data))

    # print('clustering')
    if clustering_method == "kmeans":
        kmeans = KMeans(n_clusters=num_clusters, n_init="auto", init="k-means++")
        kmeans.fit(cluster_data)
        cluster_labels = kmeans.labels_
    elif clustering_method == "both":
        gmm = GaussianMixture(
            n_components=num_clusters,
            verbose=1,
            covariance_type="diag",
            init_params="k-means++",
            tol=0.0001,
        )
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict(cluster_data)
    elif clustering_method == "variational":
        gmm = BayesianGaussianMixture(
            n_components=num_clusters,
            verbose=1,
            covariance_type="diag",
            init_params="k-means++",
            tol=0.0001,
        )
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict_proba(cluster_data)
    elif clustering_method == "gmm":
        gmm = GaussianMixture(
            n_components=num_clusters,
            verbose=1,
            covariance_type="diag",
        )
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict(cluster_data)

    if clustering_method == "variational":
        group_cluster_labels, agree_rates = aggregate_and_sample(cluster_labels, child_group_lengths)
    else:
        # voting to determine the cluster label for each parent
        group_cluster_labels = []
        curr_index = 0
        agree_rates = []
        for group_length in child_group_lengths:
            # First, determine the most common label in the current group
            most_common_label_count = np.max(np.bincount(cluster_labels[curr_index : curr_index + group_length]))
            group_cluster_label = np.argmax(np.bincount(cluster_labels[curr_index : curr_index + group_length]))
            group_cluster_labels.append(group_cluster_label)

            # Compute agree rate using the most common label count
            agree_rate = most_common_label_count / group_length
            agree_rates.append(agree_rate)

            # Then, update the curr_index for the next iteration
            curr_index += group_length

    # Compute the average agree rate across all groups
    average_agree_rate = np.mean(agree_rates)
    print("Average agree rate: ", average_agree_rate)

    group_assignment = np.repeat(group_cluster_labels, child_group_lengths, axis=0).reshape((-1, 1))

    # obtain the child data with clustering
    sorted_child_data_with_cluster = np.concatenate([sorted_child_data, group_assignment], axis=1)

    group_labels_list = group_cluster_labels
    group_lengths_list = child_group_lengths.tolist()

    group_lengths_dict = {}
    for i in range(len(group_labels_list)):
        group_label = group_labels_list[i]
        if group_label not in group_lengths_dict:
            group_lengths_dict[group_label] = defaultdict(int)
        group_lengths_dict[group_label][group_lengths_list[i]] += 1

    group_lengths_prob_dicts = {}
    for group_label, freq_dict in group_lengths_dict.items():
        group_lengths_prob_dicts[group_label] = freq_to_prob(freq_dict)

    # recover the preprocessed data back to dataframe
    child_df_with_cluster = pd.DataFrame(
        sorted_child_data_with_cluster,
        columns=original_child_cols + [relation_cluster_name],
    )

    # recover child df order
    child_df_with_cluster = pd.merge(
        child_df[[child_primary_key]],
        child_df_with_cluster,
        on=child_primary_key,
        how="left",
    )

    parent_id_to_cluster = {}
    for i in range(len(sorted_child_data)):
        parent_id = sorted_child_data[i, foreing_key_index]
        if parent_id in parent_id_to_cluster:
            assert parent_id_to_cluster[parent_id] == sorted_child_data_with_cluster[i, -1]
            continue
        parent_id_to_cluster[parent_id] = sorted_child_data_with_cluster[i, -1]

    max_cluster_label = max(parent_id_to_cluster.values())

    parent_data_clusters = []
    for i in range(len(parent_data)):
        if parent_data[i, parent_primary_key_index] in parent_id_to_cluster:
            parent_data_clusters.append(parent_id_to_cluster[parent_data[i, parent_primary_key_index]])
        else:
            parent_data_clusters.append(max_cluster_label + 1)

    parent_data_clusters = np.array(parent_data_clusters).reshape(-1, 1)
    parent_data_with_cluster = np.concatenate([parent_data, parent_data_clusters], axis=1)
    parent_df_with_cluster = pd.DataFrame(
        parent_data_with_cluster, columns=original_parent_cols + [relation_cluster_name]
    )

    new_col_entry = {
        "type": "discrete",
        "size": len(set(parent_data_clusters.flatten())),
    }

    print("Number of cluster centers: ", len(set(parent_data_clusters.flatten())))

    parent_domain_dict[relation_cluster_name] = new_col_entry.copy()
    child_domain_dict[relation_cluster_name] = new_col_entry.copy()

    return parent_df_with_cluster, child_df_with_cluster, group_lengths_prob_dicts
