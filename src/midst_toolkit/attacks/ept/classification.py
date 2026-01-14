from logging import INFO

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from torch import nn, optim
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc


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
        "actual": lambda x: not (
            x.endswith("error") or x.endswith("error_ratio") or x.endswith("accuracy") or x.endswith("prediction")
        ),
        "error": lambda x: x.endswith("error"),
        "error_ratio": lambda x: x.endswith("error_ratio"),
        "accuracy": lambda x: x.endswith("accuracy"),
        "prediction": lambda x: x.endswith("prediction"),
    }

    # Filter columns for each type in args.columns_lst
    selected_columns = [
        col for col_type in columns_list for col in features_df.columns if suffix_mapping[col_type](col)
    ]

    return features_df[selected_columns].values


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron (MLP) classifier.
    """

    def __init__(self, input_size=100, hidden_size=64, output_size=1):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size), nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def train_mlp(x_train, y_train, x_test, y_test, device, eval):
    """
    Train an MLP classifier and evaluate it on the test set.
    """
    epochs = 10
    input_size = x_train.shape[1]
    model = MLPClassifier(input_size=input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x_train, y_train = (
        torch.tensor(x_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.float32).to(device),
    )
    if eval:
        x_test, y_test = (
            torch.tensor(x_test, dtype=torch.float32).to(device),
            torch.tensor(y_test, dtype=torch.float32).to(device),
        )

    # Train the model
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    y_pred, y_proba = None, None
    if eval:
        model.eval()
        with torch.no_grad():
            # Get probabilities
            y_proba = model(x_test).squeeze().cpu().numpy()
            # Convert probabilities to binary predictions
            y_pred = (y_proba > 0.5).astype(float)

    return model, y_pred, y_proba

def get_scores(y_true, y_proba, y_pred, fpr_thresholds=[0.1, 0.01, 0.001]) -> dict[str, float]: 
    """
    Calculate evaluation scores for the classifier.
    """
    accuracy = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_roc = auc(fpr, tpr)

    # Compute TPR at specific FPR thresholds
    tpr_at_fpr = {}
    for threshold in fpr_thresholds:
        tpr_at_fpr[threshold] = max(tpr[fpr < threshold])
    
    scores = {
        "accuracy": accuracy,
        "AUC-ROC": auc_roc,
    }
    for threshold, tpr_value in tpr_at_fpr.items():
        scores[f"TPR at FPR {threshold}"] = tpr_value

    return scores

def train_attack_classifier(
    classifier_type: str,
    columns_list: list[str],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[dict]:
    """
    Train an attack classifier for EPT-MIA attack using specified classifier and specific selection of columns.
    """
    log(INFO, f"Training {classifier_type} classifier using features from columns: {columns_list}")

    all_results = {
        prediction_results := {},
        scores := {}
    }

    x_train = filter_data(x_train, columns_list)
    y_train = np.hstack(y_train)

    x_test = filter_data(x_test, columns_list)
    y_test = np.hstack(y_test)

    assert x_train.shape[0] == y_train.shape[0], "Mismatch in number of training samples and labels"
    assert x_test.shape[0] == y_test.shape[0], "Mismatch in number of test samples and labels"
    assert x_train.shape[1] == x_test.shape[1], "Mismatch in number of features between train and test sets"
    
    assert classifier_type in ["XGBoost", "CatBoost", "MLP"], f"Unsupported classifier type: {classifier_type}"

    if classifier_type == "XGBoost":
        model = XGBClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]
    elif classifier_type == "CatBoost":
        model = CatBoostClassifier(verbose=0)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]
        import pdb; pdb.set_trace()

    elif classifier_type == "MLP":
        model, y_pred, y_proba = train_mlp(
            x_train, y_train, x_test, y_test, torch.device("cuda" if torch.cuda.is_available() else "cpu"), eval=True
        )

    prediction_results = {
        "y_true": y_test,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


    fpr_thresholds = [0.1, 0.01, 0.001]

    all_results.prediction_results = prediction_results
    all_results.scores = get_scores(y_test, y_proba, y_pred, fpr_thresholds)
    

    return all_results
