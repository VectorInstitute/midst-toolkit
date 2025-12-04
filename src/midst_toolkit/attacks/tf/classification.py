# mypy: disable-error-code=no-untyped-def
from pathlib import Path

import torch
from torch import nn, optim

from midst_toolkit.attacks.tf.data_utils import get_tpr_at_fpr


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Initializes the MLP (Multi-Layer Perceptron) model.

        Args:
            input_dim (int): The number of input features.
            hidden_dim (int): The number of units in the hidden layers.

        Attributes:
            fc1 (nn.Linear): The first fully connected layer.
            fc2 (nn.Linear): The second fully connected layer.
            fc3 (nn.Linear): The output fully connected layer with a single output unit.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Performs a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after passing through three fully connected layers with
            tanh activations on the first two layers and a sigmoid activation on the final layer.
        """
        residual = torch.tanh(self.fc1(x))
        residual = torch.tanh(self.fc2(residual))
        return torch.sigmoid(self.fc3(residual))


def custom_loss_fn(model, x, y):
    """
    Computes the custom loss for a given model, input, and target.

    This function calculates the Binary Cross-Entropy (BCE) loss between the
    predicted confidences from the model and the target values. The target
    values are unsqueezed to match the shape required by the BCE loss function.

    Args:
        model (torch.nn.Module): The model used to generate predictions.
        x (torch.Tensor): The input tensor to the model.
        y (torch.Tensor): The target tensor containing ground truth values.

    Returns:
        torch.Tensor: The computed BCE loss.
    """
    confidences = model(x)
    x = x.float()
    y = y.float()
    return nn.BCELoss()(confidences, y.unsqueeze(1))


def fitmodel(
    regression_model,
    x_train,
    x_train_label,
    x_val,
    x_val_label,
    num_epochs=1000,
    learning_rate=1e-4,
    use_best_checkpoint=None,
    best_model_dir=None,
):
    """
    Trains a regression model using the provided training and testing data.

    Args:
        regression_model (torch.nn.Module): The regression model to be trained.
        x_train (numpy.ndarray or torch.Tensor): Training input data.
        x_train_label (numpy.ndarray or torch.Tensor): Training labels.
        x_val (numpy.ndarray or torch.Tensor): Testing input data.
        x_val_label (numpy.ndarray or torch.Tensor): Testing labels.
        num_epochs (int, optional): Number of training epochs. Defaults to 1000.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        use_best_checkpoint (bool, optional): Whether to load the best model checkpoint after training.
        best_model_dir (Path or str, optional): Directory to save the best model checkpoint. Defaults to None.

    Returns:
        torch.nn.Module: The trained regression model.
    """

    def save_best_model(model, path):
        torch.save(model.state_dict(), path)

    def load_best_model(model, path, device):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)

    def evaluate_model(model, x, y):
        loss = custom_loss_fn(model, x, y)
        tpr = get_tpr_at_fpr(
            y.detach().cpu().numpy(),
            model(x).detach().cpu().numpy(),
        )
        return loss.item(), tpr

    if use_best_checkpoint and best_model_dir is not None:
        best_model_dir = Path(".")  # or raise ValueError
        print(f"Best model will be saved to: {best_model_dir}")

    best_model_path = best_model_dir / "best_model.pt"
    optimizer = optim.Adam(regression_model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    has_validation = x_val is not None
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(x_train_label, dtype=torch.float32).to(device)
    if has_validation:
        x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(x_val_label, dtype=torch.float32).to(device)

    indices = torch.randperm(x_train.size(0))
    x_train, y_train = x_train[indices], y_train[indices]

    regression_model.train()
    best_tpr, best_model_exists = 0.0, False

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = custom_loss_fn(regression_model, x_train, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            train_loss, train_tpr = evaluate_model(regression_model, x_train, y_train)
            if x_val is not None:
                test_loss, test_tpr = evaluate_model(regression_model, x_val, y_val)
                if test_tpr > best_tpr:
                    best_tpr = test_tpr
                    save_best_model(regression_model, best_model_path)
                    best_model_exists = True
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, "
                    f"Test Loss: {test_loss}, Train TPR: {train_tpr}, Test TPR: {test_tpr}"
                )
            else:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, Train TPR: {train_tpr}")

    if use_best_checkpoint and best_model_exists:
        load_best_model(regression_model, best_model_path, device)

    if x_val is not None:
        test_loss, test_tpr = evaluate_model(regression_model, x_val, y_val)
        print(f"Final best loss: {test_loss}, best TPR: {test_tpr}")

    return regression_model
