import argparse
from logging import INFO
from pathlib import Path

import pandas as pd

from midst_toolkit.common.logger import log
from midst_toolkit.data_processing.midst_data_processing import (
    load_midst_data,
    process_midst_data_for_alpha_precision_evaluation,
)
from midst_toolkit.evaluation.generation.alpha_precision import AlphaPrecision
from midst_toolkit.evaluation.generation.utils import (
    create_quality_metrics_directory,
    dump_metrics_dict,
    extract_columns_based_on_meta_info,
    one_hot_encode_categoricals_and_merge_with_numerical,
)


# Killing a benign pandas warning
pd.options.mode.chained_assignment = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic_data_path",
        required=False,
        type=str,
        default=None,
        help="The file path of the synthetic data. If specified, overrides dataname and model args",
    )
    parser.add_argument(
        "--real_data_path",
        required=False,
        type=str,
        default=None,
        help="The file path of the real data. If specified, overrides dataname and model args",
    )
    parser.add_argument(
        "--meta_info_path",
        required=False,
        type=str,
        default=None,
        help="The file path of the meta data for the synthetic generation. If specified, overrides dataname",
    )
    parser.add_argument(
        "--save_directory",
        required=False,
        type=str,
        default=None,
        help="The directory path to which metrics are saved. Created if it doesn't exist. If not provided, not saved",
    )
    parser.add_argument(
        "--dataname",
        required=True,
        type=str,
        default="adult",
        help=(
            "Used to construct default paths for the real or synthetic data if not specified. "
            "Also used to potentially trigger dataset preprocessing if one of 'default', 'shoppers', 'faults', 'news'"
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        default="model",
        help=(
            "Used to construct default paths for the real or synthetic data if not specified. "
            "Also used to potentially trigger dataset preprocessing if '...codi', '...great' specified. "
        ),
    )

    args = parser.parse_args()

    dataname = args.dataname
    model = args.model
    synthetic_data_path = args.synthetic_data_path
    real_data_path = args.real_data_path

    if args.synthetic_data_path:
        synthetic_data_path = Path(args.synthetic_data_path)
    else:
        assert dataname is not None and model is not None, (
            "Both dataname and model must be defined to construct default synthetic data path"
        )
        synthetic_data_path = Path(f"synthetic/{dataname}/{model}.csv")
        log(INFO, f"Synthetic data path not provided. Default constructed: {synthetic_data_path}")

    if args.real_data_path:
        real_data_path = Path(args.real_data_path)
    else:
        assert dataname is not None, "Dataname must be defined to construct default real data path"
        real_data_path = Path(f"synthetic/{dataname}/real.csv")
        log(INFO, f"Real data path not provided. Default constructed: {real_data_path}")

    if args.meta_info_path:
        meta_info_path = Path(args.meta_info_path)
    else:
        assert dataname is not None, "Dataname must be defined to construct default meta info path"
        meta_info_path = Path(f"data/{dataname}/info.json")
        log(INFO, f"Meta info path not provided. Default constructed: {meta_info_path}")

    real_data, synthetic_data, meta_info = load_midst_data(real_data_path, synthetic_data_path, meta_info_path)

    numerical_real_data, categorical_real_data = extract_columns_based_on_meta_info(real_data, meta_info)
    numerical_synthetic_data, categorical_synthetic_data = extract_columns_based_on_meta_info(
        synthetic_data, meta_info
    )

    numerical_real_numpy, categorical_real_numpy, numerical_synthetic_numpy, categorical_synthetic_numpy = (
        process_midst_data_for_alpha_precision_evaluation(
            numerical_real_data,
            categorical_real_data,
            numerical_synthetic_data,
            categorical_synthetic_data,
            dataname,
            model,
        )
    )

    real_dataframe, synthetic_dataframe = one_hot_encode_categoricals_and_merge_with_numerical(
        categorical_real_numpy, categorical_synthetic_numpy, numerical_real_numpy, numerical_synthetic_numpy
    )

    log(INFO, f"Data shape: {synthetic_dataframe.shape}")

    alpha_precision_metric = AlphaPrecision(naive_only=False)

    quality_results = alpha_precision_metric.compute(real_dataframe, synthetic_dataframe)
    if args.save_directory:
        save_directory = Path(args.save_directory)
        create_quality_metrics_directory(save_directory)
        dump_metrics_dict(quality_results, save_directory / f"{model}.txt")
