import argparse
from logging import INFO
from pathlib import Path

import pandas as pd

from midst_toolkit.common.logger import log
from midst_toolkit.data_processing.midst_data_processing import load_midst_data_with_test
from midst_toolkit.evaluation.generation.distance_closest_record import (
    DistanceToClosestRecordScore,
    preprocess,
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
        "--real_data_train_path",
        required=False,
        type=str,
        default=None,
        help=(
            "The file path of the real data that was used to train the generating model. If specified, overrides "
            "dataname and model args."
        ),
    )
    parser.add_argument(
        "--real_data_test_path",
        required=False,
        type=str,
        default=None,
        help=(
            "The file path of the real data that was NOT used to train the generating model. If specified, overrides "
            "dataname and model args."
        ),
    )
    parser.add_argument(
        "--meta_info_path",
        required=False,
        type=str,
        default=None,
        help="The file path of the meta data for the synthetic generation. If specified, overrides dataname",
    )
    parser.add_argument(
        "--dataname",
        required=True,
        type=str,
        default="adult",
        help="Used to construct default paths for the real or synthetic data if not specified.",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        default="model",
        help="Used to construct default paths for the real or synthetic data if not specified.",
    )

    args = parser.parse_args()

    dataname = args.dataname
    model = args.model
    synthetic_data_path = args.synthetic_data_path
    real_data_train_path = args.real_data_train_path
    real_data_test_path = args.real_data_test_path

    if args.synthetic_data_path:
        synthetic_data_path = Path(args.synthetic_data_path)
    else:
        assert dataname is not None and model is not None, (
            "Both dataname and model must be defined to construct default synthetic data path"
        )
        synthetic_data_path = Path(f"synthetic/{dataname}/{model}.csv")
        log(INFO, f"Synthetic data path not provided. Default constructed: {synthetic_data_path}")

    if args.real_data_train_path:
        real_data_train_path = Path(args.real_data_train_path)
    else:
        assert dataname is not None, "Dataname must be defined to construct default real data train path"
        real_data_train_path = Path(f"synthetic/{dataname}/real.csv")
        log(INFO, f"Real data train path not provided. Default constructed: {real_data_train_path}")

    if args.real_data_test_path:
        real_data_test_path = Path(args.real_data_test_path)
    else:
        assert dataname is not None, "Dataname must be defined to construct default real data test path"
        real_data_test_path = Path(f"synthetic/{dataname}/test.csv")
        log(INFO, f"Real data test path not provided. Default constructed: {real_data_test_path}")

    if args.meta_info_path:
        meta_info_path = Path(args.meta_info_path)
    else:
        assert dataname is not None, "Dataname must be defined to construct default meta info path"
        meta_info_path = Path(f"data/{dataname}/info.json")
        log(INFO, f"Meta info path not provided. Default constructed: {meta_info_path}")

    real_data_train, synthetic_data, real_data_test, meta_info = load_midst_data_with_test(
        real_data_train_path, synthetic_data_path, meta_info_path, real_data_test_path
    )

    synthetic_data, real_data_train, real_data_test = preprocess(
        meta_info, synthetic_data, real_data_train, real_data_test
    )
    metric = DistanceToClosestRecordScore()
    dcr_score = metric.compute(synthetic_data, real_data_train, real_data_test)
    log(INFO, f"{dataname}-{model}, DCR Score = {dcr_score}")
