from logging import INFO
import itertools
import pandas as pd
import numpy as np

import hydra
from datetime import datetime

from midst_toolkit.common.logger import log


def filter_data(features_df: pd.DataFrame, columns_list: list[str]) -> np.ndarray:
    """
    Filters columns from a single DataFrame based on specified suffixes.

    This function processes a pandas DataFrame, selecting columns based on
    suffixes that correspond to the types specified in `columns_list` (e.g.,
    'actual', 'error'). It then returns the data from these selected columns
    as a NumPy array.

    Args:
        features_df: The pandas DataFrame to process.
        columns_lst: A list of strings specifying the types of columns
                    to select. 

    Returns:
        np.ndarray: A NumPy array containing the data from the selected columns.
    """

    suffix_mapping = {
        'actual': lambda x: not (x.endswith('error') or x.endswith('error_ratio') or x.endswith('accuracy') or x.endswith('prediction')),
        'error': lambda x: x.endswith('error'),
        'error_ratio': lambda x: x.endswith('error_ratio'),
        'accuracy': lambda x: x.endswith('accuracy'),
        'prediction': lambda x: x.endswith('prediction'),
    }

    # Filter columns for each type in args.columns_lst
    selected_columns = [col for col_type in columns_list for col in features_df.columns if suffix_mapping[col_type](col)]

    return features_df[selected_columns].values



def train_attack_classifier(classifier_types: list[str], column_types: list[str], x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> list[dict]:

    all_results = []
    for classifier in classifier_types:
        for r in range(1, len(column_types) + 1):
            for selected_columns_tuple in itertools.combinations(column_types, r):

                # import pdb; pdb.set_trace()
                # x_train is a dataframe of 5000 rows and 28 columns (account is included). 5000 rows correspond to 25 shadow models * 200 records each. 

                selected_columns = list(selected_columns_tuple)

                x_train_filtered = filter_data(x_train, selected_columns)
                
                log(INFO, f"Training {classifier} classifier using features from columns: {selected_columns}")

                results = {
                "classifier": classifier,
                "columns_lst": " ".join(selected_columns)
                }

                all_results.append(results)

    return all_results

