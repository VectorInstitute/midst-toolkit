# ruff: noqa: PLR0915
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from midst_toolkit.attacks.tartan_federer.data_utils import get_tpr_at_fpr


Tensor = torch.Tensor


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
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, input: Tensor) -> Tensor:
        """
        Performs a forward pass through the neural network.

        Args:
            input: Input tensor to the network.

        Returns:
            Tensor: Output tensor after passing through three fully connected layers with
            tanh activations on the first two layers and a sigmoid activation on the final layer.
        """
        residual = torch.tanh(self.fc1(input))
        residual = torch.tanh(self.fc2(residual))
        return torch.sigmoid(self.fc3(residual))


def bce_loss_from_model_and_input(model: torch.nn.Module, x: Tensor, y: Tensor) -> Tensor:
    """
    Computes the custom loss for a given model, input, and target.

    This function calculates the Binary Cross-Entropy (BCE) loss between the
    predicted confidences from the model and the target values. The target
    values are unsqueezed to match the shape required by the BCE loss function.

    Args:
        model: The model used to generate predictions.
        x: The input tensor to the model.
        y: The target tensor containing ground truth values.

    Returns:
        Tensor: The computed BCE loss.
    """
    confidences = model(x)
    return nn.BCELoss()(confidences, y.unsqueeze(1))


def save_best_model(model: nn.Module, path: Path) -> None:
    """
    Saves the model's state dictionary to the specified path.

    Args:
        model: The PyTorch model to save.
        path: The file path where the model's state dictionary will be saved.
    """
    torch.save(model.state_dict(), path)


def load_best_model(model: nn.Module, path: Path, device: torch.device) -> None:
    """
    Loads the model's state dictionary from the specified path.

    Args:
        model: The PyTorch model to load the state dictionary into.
        path: The file path from which the model's state dictionary will be loaded.
        device: The device to map the model to after loading.

    Returns:
        None
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)


def evaluate_model(model: nn.Module, x: Tensor, y: Tensor) -> tuple[float, float]:
    """
    Evaluates the model on the provided input and target tensors,
    returning the loss and TPR at the prespecified 10% FPR.

    Args:
        model: The PyTorch model to evaluate.
        x: Input tensor for evaluation.
        y: Target tensor containing ground truth labels.

    Returns:
        tuple[float, float]: A tuple containing the loss and the TPR at 10% FPR.
    """
    model.eval()
    with torch.no_grad():
        loss = bce_loss_from_model_and_input(model, x, y).item()
        probs = model(x).detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
    tpr = get_tpr_at_fpr(labels, probs)
    return loss, tpr


def fitmodel(
    regression_model: nn.Module,
    x_train: np.ndarray | Tensor,
    y_train: np.ndarray | Tensor,
    x_val: np.ndarray | Tensor | None,
    y_val: np.ndarray | Tensor | None,
    num_epochs: int,
    learning_rate: float,
    best_model_checkpoint_dir: Path | None,
    reporting_interval: int = 5,
) -> nn.Module:
    """
    Trains a classifier for MIA using the provided training data, with optional validation and model checkpointing.

    Args:
        regression_model: The PyTorch model to be trained.
        x_train: Training input features.
        y_train: Training labels.
        x_val: Validation input features, or None if not provided.
        y_val: Validation labels, or None if not provided.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        best_model_checkpoint_dir: Directory to save the best model checkpoint.
        reporting_interval: Interval (in epochs) for reporting training/validation metrics. Defaults to 5.

    Returns:
        nn.Module: The trained regression model (best model if validation and checkpointing are enabled).
    """
    if best_model_checkpoint_dir is not None:
        best_model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = best_model_checkpoint_dir / "best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regression_model.to(device)

    optimizer = optim.Adam(regression_model.parameters(), lr=learning_rate)

    has_validation = x_val is not None and y_val is not None

    # Convert tensors
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    if has_validation:
        x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_tpr = -float("inf")
    best_model_exists = False

    ########## Training loop ##########

    for epoch in range(num_epochs):
        # Shuffle every epoch
        perm = torch.randperm(x_train_t.size(0))
        x_batch = x_train_t[perm]
        y_batch = y_train_t[perm]

        regression_model.train()
        optimizer.zero_grad()
        loss = bce_loss_from_model_and_input(regression_model, x_batch, y_batch)
        loss.backward()
        optimizer.step()

        ########## Reporting ##########
        if (epoch + 1) % reporting_interval == 0:
            train_loss, train_tpr = evaluate_model(regression_model, x_train_t, y_train_t)

            if has_validation:
                val_loss, val_tpr = evaluate_model(regression_model, x_val_t, y_val_t)

                # Save best model
                if best_model_checkpoint_dir is not None and val_tpr > best_tpr:
                    best_tpr = val_tpr
                    save_best_model(regression_model, best_model_path)
                    best_model_exists = True

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train TPR: {train_tpr:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val TPR: {val_tpr:.4f}"
                )
            else:
                print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train TPR: {train_tpr:.4f}")

    ########## Load best model if available ##########
    if has_validation and best_model_exists:
        load_best_model(regression_model, best_model_path, device)

    ########## Final evaluation ##########
    regression_model.eval()

    if has_validation:
        final_loss, final_tpr = evaluate_model(regression_model, x_val_t, y_val_t)
        print(f"Final best validation — Loss: {final_loss:.4f}, TPR: {final_tpr:.4f}")
    else:
        print("Training complete (no validation set provided).")

    return regression_model
