from logging import INFO
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn, optim

from midst_toolkit.common.logger import log
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.evaluation.privacy.mia_scoring import TprAtFpr


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """
        Initializes the MLP (Multi-Layer Perceptron) model.

        Args:
            input_dim: The number of input features.
            hidden_dim: The number of units in the hidden layers.

        Attributes:
            fc1: The first fully connected layer.
            fc2: The second fully connected layer.
            fc3: The output fully connected layer with a single output unit.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Performs a forward pass through the neural network.

        Args:
            input_tensor: Input tensor to the network.

        Returns:
            Output tensor after passing through three fully connected layers with tanh activations on the first
            two layers and a sigmoid activation on the final layer.
        """
        output = torch.tanh(self.fc1(input_tensor))
        output = torch.tanh(self.fc2(output))
        return torch.sigmoid(self.fc3(output))


def bce_loss_from_model_and_input(model: torch.nn.Module, input_tensor: Tensor, target: Tensor) -> Tensor:
    """
    Computes the custom loss for a given model, input, and target.

    This function calculates the Binary Cross-Entropy (BCE) loss between the predicted confidences from the model
    and the target values. The target values are unsqueezed to match the shape required by the BCE loss function.

    Args:
        model: The model used to generate predictions.
        input_tensor: The input tensor to the model.
        target: The target tensor containing ground truth values.

    Returns:
        The computed BCE loss.
    """
    confidences = model(input_tensor)
    return nn.BCELoss()(confidences, target.unsqueeze(1))


def save_model_state(model: nn.Module, path: Path) -> None:
    """
    Saves the model's state dictionary to the specified path.

    Args:
        model: The PyTorch model to save.
        path: The file path where the model's state dictionary will be saved.
    """
    torch.save(model.state_dict(), path)


def load_model_state(model: nn.Module, path: Path, device: torch.device) -> None:
    """
    Loads the model's state dictionary from the specified path.

    Args:
        model: The PyTorch model to load the state dictionary into.
        path: The file path from which the model's state dictionary will be loaded.
        device: The device to map the model to after loading.
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)


def evaluate_model(
    model: nn.Module, input_tensor: Tensor, target: Tensor, fpr_threshold: float = 0.1
) -> tuple[float, float]:
    """
    Evaluates the model on the provided input and target tensors, returning the loss and TPR at the pre-specified FPR
    threshold.

    Args:
        model: The PyTorch model to evaluate.
        input_tensor: Input tensor for evaluation.
        target: Target tensor containing ground truth labels.
        fpr_threshold: FPR threshold at which to evaluate TPR at FPR metric for MIA success.

    Returns:
        A tuple containing the loss and the TPR at FPR = fpr_threshold.
    """
    model.eval()
    with torch.no_grad():
        loss = bce_loss_from_model_and_input(model, input_tensor, target).item()
        probs = model(input_tensor).detach().cpu().numpy()
        labels = target.detach().cpu().numpy()

    # Evaluate TRP at FPR = fpr_threshold
    tpr_at_fpr = TprAtFpr.get_tpr_at_fpr(labels, probs, fpr_threshold)

    return loss, tpr_at_fpr


def fit_model(
    regression_model: nn.Module,
    train_features: np.ndarray | Tensor,
    train_targets: np.ndarray | Tensor,
    validation_features: np.ndarray | Tensor | None,
    validation_targets: np.ndarray | Tensor | None,
    num_epochs: int,
    learning_rate: float,
    best_model_checkpoint_dir: Path | None,
    reporting_interval: int = 5,
) -> nn.Module:
    """
    Trains a classifier for MIA using the provided training data, with optional validation and model checkpointing.

    Args:
        regression_model: The PyTorch model to be trained.
        train_features: Training input features.
        train_targets: Training labels.
        validation_features: Validation input features, or None if not provided.
        validation_targets: Validation labels, or None if not provided.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        best_model_checkpoint_dir: Directory to save the best model checkpoint.
        reporting_interval: Interval (in epochs) for reporting training/validation metrics. Defaults to 5.

    Returns:
        The trained regression model (best model if validation and checkpointing are enabled).
    """
    if best_model_checkpoint_dir is not None:
        best_model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = best_model_checkpoint_dir / "best_model.pt"

    regression_model.to(DEVICE)

    optimizer = optim.Adam(regression_model.parameters(), lr=learning_rate)

    has_validation = validation_features is not None and validation_targets is not None

    # Convert tensors
    train_input_tensor = torch.tensor(train_features, dtype=torch.float32).to(DEVICE)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32).to(DEVICE)

    if has_validation:
        validation_input_tensor = torch.tensor(validation_features, dtype=torch.float32).to(DEVICE)
        validation_targets_tensor = torch.tensor(validation_targets, dtype=torch.float32).to(DEVICE)

    best_tpr = -float("inf")
    best_model_exists = False

    ########## Training loop ##########

    for epoch in range(num_epochs):
        # Shuffle every epoch. Note that we're not strictly doing batching here. Just permutation of the rows.
        row_permutation = torch.randperm(train_input_tensor.size(0))
        input_batch = train_input_tensor[row_permutation]
        target_batch = train_targets_tensor[row_permutation]

        regression_model.train()
        optimizer.zero_grad()
        loss = bce_loss_from_model_and_input(regression_model, input_batch, target_batch)
        loss.backward()
        optimizer.step()

        ########## Reporting ##########
        if (epoch + 1) % reporting_interval == 0:
            train_loss, train_tpr = evaluate_model(regression_model, train_input_tensor, train_targets_tensor)

            if has_validation:
                val_loss, val_tpr = evaluate_model(
                    regression_model, validation_input_tensor, validation_targets_tensor
                )

                # Save best model
                if best_model_checkpoint_dir is not None and val_tpr > best_tpr:
                    best_tpr = val_tpr
                    save_model_state(regression_model, best_model_path)
                    best_model_exists = True

                log(
                    INFO,
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train TPR: {train_tpr:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val TPR: {val_tpr:.4f}",
                )
            else:
                log(INFO, f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train TPR: {train_tpr:.4f}")

    ########## Load best model if available ##########
    if has_validation and best_model_exists:
        load_model_state(regression_model, best_model_path, DEVICE)

    ########## Final evaluation ##########
    regression_model.eval()

    if has_validation:
        final_loss, final_tpr = evaluate_model(regression_model, validation_input_tensor, validation_targets_tensor)
        log(INFO, f"Final best validation — Loss: {final_loss:.4f}, TPR: {final_tpr:.4f}")
    else:
        log(INFO, "Training complete (no validation set provided).")

    return regression_model
