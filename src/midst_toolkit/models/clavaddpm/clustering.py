def clava_clustering(
    tables: Tables,
    relation_order: RelationOrder,
    save_dir: Path,
    configs: Configs,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[int, float]]]:
    """
    Clustering function for the mutli-table function of theClavaDDPM model.

    Args:
        tables: Definition of the tables and their relations. Example:
            {
                "table1": {
                    "children": ["table2"],
                    "parents": []
                },
                "table2": {
                    "children": [],
                    "parents": ["table1"]
                }
            }
        relation_order: List of tuples of parent and child tables. Example:
            [("table1", "table2"), ("table1", "table3")]
        save_dir: Directory to save the clustering checkpoint.
        configs: Dictionary of configurations. The following config keys are required:
            {
                num_clusters = int | dict,
                parent_scale = float,
                clustering_method = str["kmeans" | "both" | "variational" | "gmm"],
            }

    """
    relation_order_reversed = relation_order[::-1]
    all_group_lengths_prob_dicts = {}

    # Clustering
    if os.path.exists(save_dir / "cluster_ckpt.pkl"):
        print("Clustering checkpoint found, loading...")
        cluster_ckpt = pickle.load(open(save_dir / "cluster_ckpt.pkl", "rb"))
        # ruff: noqa: SIM115
        tables = cluster_ckpt["tables"]
        all_group_lengths_prob_dicts = cluster_ckpt["all_group_lengths_prob_dicts"]
    else:
        for parent, child in relation_order_reversed:
            if parent is not None:
                print(f"Clustering {parent} -> {child}")
                if isinstance(configs["num_clusters"], dict):
                    num_clusters = configs["num_clusters"][child]
                else:
                    num_clusters = configs["num_clusters"]
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
                    configs["parent_scale"],
                    1,  # not used for now
                    parent,
                    child,
                    clustering_method=configs["clustering_method"],
                )
                tables[parent]["df"] = parent_df_with_cluster
                tables[child]["df"] = child_df_with_cluster
                all_group_lengths_prob_dicts[(parent, child)] = group_lengths_prob_dicts

        cluster_ckpt = {
            "tables": tables,
            "all_group_lengths_prob_dicts": all_group_lengths_prob_dicts,
        }
        pickle.dump(cluster_ckpt, open(save_dir / "cluster_ckpt.pkl", "wb"))
        # ruff: noqa: SIM115

    for parent, child in relation_order:
        if parent is None:
            tables[child]["df"]["placeholder"] = list(range(len(tables[child]["df"])))

    return tables, all_group_lengths_prob_dicts
