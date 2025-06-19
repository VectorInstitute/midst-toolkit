import torch
from torch import nn


class TitanicModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        """
        Model.

        Args:
            input_dim (int): Number of input features
            num_classes (int, optional): Number of classes to predict. Defaults to 2.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 96)
        self.linear4 = nn.Linear(96, 32)
        self.linear5 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.dropout(self.relu(self.linear3(x)))
        x = self.dropout(self.relu(self.linear4(x)))
        return self.softmax(self.linear5(x))
