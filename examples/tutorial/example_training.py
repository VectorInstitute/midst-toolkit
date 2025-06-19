from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from midst_toolkit.models.titanic_model import TitanicModel


# Settings for matplotlib
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# Specify float format for pandas tables
pd.options.display.float_format = "{:.3f}".format


class TabularDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor | None = None) -> None:
        """
        Torch dataset class for tabular data.

        Args:
            x (torch.Tensor): input data
            y (torch.Tensor, optional): label data. Defaults to None.
        """
        self.x = x
        self.y = y

    def __len__(self) -> int:
        """
        Length of this dataset based on the input data.

        Returns:
            int: Length of this dataset based on the input data
        """
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Gets an item from the dataset at the provided index.

        Args:
            idx (int): data index to fetch

        Returns:
            tuple[torch.Tensor, torch.Tensor]: input and label data in a tuple
        """
        if self.y is None:
            # For test dataset
            return self.x[idx], None
        # For train and validation dataset
        return self.x[idx], self.y[idx]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Do a number of feature engineering tasks based on the provided dataframe.

    Args:
        df (pd.DataFrame): Dataframe on which to perform feature engineering

    Returns:
        pd.DataFrame: Dataframe resulting from the feature engineering.
    """
    # The Fare column is skewed, so taking the natural log will make it more even
    df["LogFare"] = np.log1p(df["Fare"])

    # Taking the first character of the Cabin column gives the deck, and mapping single
    # characters to groups of decks; other decks will be NaN
    df["DeckGroup"] = (
        df["Cabin"].str[0].map({"A": "ABC", "B": "ABC", "C": "ABC", "D": "DE", "E": "DE", "F": "FG", "G": "FG"})
    )

    # Add up all family members
    df["Family"] = df["SibSp"] + df["Parch"]

    # If the person traveled alone (=1) or has any family members (=0)
    df["Alone"] = (df["Family"] == 0).map({True: 1, False: 0})

    # Specify the ticket frequency (how common someone's ticket is)
    df["TicketFreq"] = df.groupby("Ticket")["Ticket"].transform("count")

    # Extract someone's title (e.g., Mr, Mrs, Miss, Rev)
    df["Title"] = df["Name"].str.split(", ", expand=True).iloc[:, 1].str.split(".", expand=True).iloc[:, 0]

    # Limit titles to those in the dictionary below; other titles will be NaN
    df["Title"] = df["Title"].map({"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master"})

    # Change sex to numbers (male=1, female=0)
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    return df


def fill_missing(df: pd.DataFrame, modes: pd.Series) -> pd.DataFrame:
    """
    Fill in any missing information using the provided modes.

    Args:
        df (pd.DataFrame): Dataframe to fill in
        modes (pd.Series): Modes

    Returns:
        pd.DataFrame: Resulting dataframe after fill
    """
    return df.fillna(modes)


# Perform min-max scaling
def scale_min_max(df: pd.DataFrame, col_name: str, xmin: float, xmax: float) -> pd.DataFrame:
    """
    Scale the column in the dataframe to be within xmin and xmax.

    Args:
        df (pd.DataFrame): Dataframe to be scaled
        col_name (str): Column to be scaled
        xmin (float): minimum value
        xmax (float): maximum value

    Returns:
        pd.DataFrame: New dataframe with column scaled.
    """
    df[col_name] = (df[col_name] - xmin) / (xmax - xmin)
    return df


# Add dummy variables
def add_dummies(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Add dummies to the dataframe for the specific columns.

    Args:
        df (pd.DataFrame): Dataframe to operate on
        cols (list[str]): columns to which the dummies should correspond

    Returns:
        pd.DataFrame: New dataframe.
    """
    return pd.get_dummies(df, columns=cols, dtype=int)
    return df


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute a standard accuracy metric.

    Args:
        y_true (torch.Tensor): Label tensor
        y_pred (torch.Tensor): Prediction tensor

    Returns:
        float: standard accuracy metric.
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_pred) * 100


# Load train and test data
df = pd.read_csv("examples/tutorial/data/train.csv")
test_df = pd.read_csv("examples/tutorial/data/test.csv")

# Separate training features from label column
train_labels = df["Survived"].copy()
df = df.drop("Survived", axis=1)

# Apply feature engineering
df = feature_engineering(df)

# Remove columns I no longer need
df = df.drop(["Name", "Ticket", "Cabin", "PassengerId", "Fare", "SibSp", "Parch"], axis=1)

# Fill missing values with the modes
train_modes = df.mode().iloc[0]


df = fill_missing(df, train_modes)


train_age_min = df["Age"].min()
train_age_max = df["Age"].max()
df = scale_min_max(df, "Age", train_age_min, train_age_max)


cols = ["Pclass", "Embarked", "DeckGroup", "Title"]
df = add_dummies(df, cols)

# Apply the same data processing steps to the test data
test_proc = (
    test_df.pipe(feature_engineering)
    .drop(["Name", "Ticket", "Cabin", "PassengerId", "Fare", "SibSp", "Parch"], axis=1)
    .pipe(fill_missing, train_modes)
    .pipe(scale_min_max, "Age", train_age_min, train_age_max)
    .pipe(add_dummies, cols)
)

x_data = torch.tensor(df.values, dtype=torch.float32)
y_data = torch.tensor(train_labels.values, dtype=torch.float32)

x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.25, shuffle=True)

x_test = torch.tensor(test_proc.values, dtype=torch.float32)

# Create dataset
train_dataset = TabularDataset(x_train, y_train)
validation_dataset = TabularDataset(x_valid, y_valid)
test_dataset = TabularDataset(test_proc, None)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
validation_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

# Assign the available processor to device
device = "cuda" if torch.cuda.is_available() else "cpu"

model = TitanicModel(19).to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.003)

# Number of epochs
epochs = 10000


# Empty loss lists to track values
epoch_count, train_loss_values, valid_loss_values = [], [], []

# Loop through the data
for epoch in range(epochs):
    # Put the model in training mode
    model.train()

    total_train_loss: float = 0
    total_train_acc: float = 0
    num_train_batches: float = 0

    for x_train, y_train in train_loader:
        # Send data to the device
        x_train = x_train.to(device)
        y_train = y_train.type(torch.LongTensor).to(device)

        # Forward pass to get predictions
        y_logits = model(x_train)
        # Convert logits into predictions
        y_pred = torch.argmax(y_logits, dim=1)

        # Compute the loss
        loss = loss_fn(y_logits, y_train)
        # Calculate the accuracy; convert the labels to integers
        acc = accuracy_fn(y_train, y_pred)

        # Reset the gradients so they don't accumulate each iteration
        optimizer.zero_grad()
        # Backward pass: backpropagate the prediction loss
        loss.backward()
        # Gradient descent: adjust the parameters by the gradients collected in the backward pass
        optimizer.step()

        # Accumulate loss and accuracy
        total_train_loss += loss.item()
        total_train_acc += acc
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches
    avg_train_acc = total_train_acc / num_train_batches

    # Put the model in evaluation mode
    model.eval()

    total_valid_loss: float = 0
    total_valid_acc: float = 0
    num_valid_batches: float = 0

    with torch.inference_mode():
        for x_valid, y_valid in train_loader:
            # Send data to the device
            x_valid = x_valid.to(device)
            y_valid = y_valid.type(torch.LongTensor).to(device)

            valid_logits = model(x_valid)
            y_pred = torch.argmax(valid_logits, dim=1)  # convert logits into predictions

            valid_loss = loss_fn(valid_logits, y_valid)
            valid_acc = accuracy_fn(y_pred, y_valid)

            # Accumulate validation loss and accuracy
            total_valid_loss += valid_loss.item()
            total_valid_acc += valid_acc
            num_valid_batches += 1

    avg_valid_loss = total_valid_loss / num_valid_batches
    avg_valid_acc = total_valid_acc / num_valid_batches

    # Print progress a total of 20 times
    if epoch % int(epochs / 20) == 0:
        print(
            f"Epoch: {epoch:4d} | Train Loss: {avg_train_loss:.5f}, Accuracy: {avg_train_acc:.2f}% | "
            f"Validation Loss: {avg_valid_loss:.5f}, Accuracy: {avg_valid_acc:.2f}%"
        )
        epoch_count.append(epoch)
        train_loss_values.append(avg_train_loss)
        valid_loss_values.append(avg_valid_loss)

# Create a directory for models
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create a model save path
MODEL_NAME = "pytorch_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

model.eval()

all_preds_list = []

for x_test, y_test in test_loader:
    # Send data to the device
    x_test = x_test.to(device)
    y_test = y_test.type(torch.LongTensor).to(device)

    test_logits = model(x_test)
    y_pred = torch.argmax(test_logits, dim=1)  # convert logits into predictions

    all_preds_list.append(y_pred.cpu().detach())

all_preds = torch.cat(all_preds_list)

print(all_preds)
