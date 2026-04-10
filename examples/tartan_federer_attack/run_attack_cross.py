"""
Cross-dataset MIA attack:
- Train MLP on diabetes folds 1-10
- Test MLP on berka folds 1-30
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from midst_toolkit.attacks.tartan_federer.tartan_federer_attack import (
    get_score, MLP, DEVICE
)
from midst_toolkit.common.random import set_all_random_seeds

set_all_random_seeds(133742)

# Config
NUM_NOISE = 300
TIMESTEPS = [5, 10, 20, 30, 40, 50, 100]
ADDITIONAL_TIMESTEPS = [0]
SAMPLES_PER_MODEL = 1994
HIDDEN_DIM = 200
LR = 1e-4
NUM_EPOCHS = 5000
MODEL_TYPE = "tabddpm"
MODEL_SUBDIR = "models"

DI_BASE = Path("whitebox_single_table_DI")
BK_BASE = Path("whitebox_single_table_70")

# Step 1: Get noise dimension from diabetes model
di_model_path = DI_BASE / "tabddpm_1" / MODEL_SUBDIR
with open(di_model_path / "None_trans_ckpt.pkl", "rb") as f:
    import pickle
    from midst_toolkit.attacks.tartan_federer.tartan_federer_attack import CustomUnpickler
    probe = CustomUnpickler(open(di_model_path / "None_trans_ckpt.pkl", "rb")).load()
di_noise_dim = probe.diffusion.num_numerical_features
print(f"Diabetes noise dim: {di_noise_dim}")

# Step 2: Get noise dimension from berka model
bk_model_path = BK_BASE / "tabddpm_1" / MODEL_SUBDIR
probe_bk = CustomUnpickler(open(bk_model_path / "None_trans_ckpt.pkl", "rb")).load()
bk_noise_dim = probe_bk.diffusion.num_numerical_features
print(f"Berka noise dim: {bk_noise_dim}")

input_dimension = NUM_NOISE * len(TIMESTEPS) * len(ADDITIONAL_TIMESTEPS)
print(f"MLP input dimension: {input_dimension}")

# Step 3: Collect diabetes training features
print("\n=== Collecting diabetes training features ===")
di_input_noise = [np.random.normal(size=di_noise_dim).tolist() for _ in range(NUM_NOISE)]

x_train_list = []
y_train_list = []

for fold_i in range(1, 11):
    print(f"  Processing diabetes fold {fold_i}...")
    model_dir = DI_BASE / f"tabddpm_{fold_i}"
    model_path = model_dir / MODEL_SUBDIR
    
    fold_features = np.zeros([SAMPLES_PER_MODEL * 2, input_dimension])
    t_count = 0
    for t in TIMESTEPS:
        for at in ADDITIONAL_TIMESTEPS:
            preds = get_score(
                model_dir, model_path, di_input_noise, MODEL_TYPE,
                meta_dir=model_dir,
                challenge_name="data_for_training_MIA.csv",
                batch_size=SAMPLES_PER_MODEL * 2,
                parallel_batch=NUM_NOISE,
                additional_timestep=at, timestep=t
            )
            fold_features[:, t_count*NUM_NOISE:(t_count+1)*NUM_NOISE] = \
                preds.detach().squeeze().cpu().numpy()
            t_count += 1
    
    x_train_list.append(fold_features)
    y_train_list.append(np.concatenate([np.zeros(SAMPLES_PER_MODEL), np.ones(SAMPLES_PER_MODEL)]))

x_train = np.vstack(x_train_list)
y_train = np.concatenate(y_train_list)
print(f"Training data shape: {x_train.shape}")

# Step 4: Train MLP on diabetes data
print("\n=== Training MLP on diabetes data ===")
model = MLP(input_dim=input_dimension, hidden_dim=HIDDEN_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

x_t = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

for epoch in range(NUM_EPOCHS):
    model.train()
    optimizer.zero_grad()
    out = model(x_t).squeeze()
    loss = criterion(out, y_t)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1000 == 0:
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")

# Evaluate on diabetes training data
model.eval()
with torch.no_grad():
    train_preds = torch.sigmoid(model(x_t)).cpu().numpy().squeeze()
train_auc = roc_auc_score(y_train, train_preds)
print(f"\nDiabetes train AUC: {train_auc:.4f}")

# Step 5: Evaluate on berka test folds
print("\n=== Evaluating on berka test folds ===")
bk_input_noise = [np.random.normal(size=bk_noise_dim).tolist() for _ in range(NUM_NOISE)]

x_test_list = []
y_test_list = []

for fold_i in range(1, 31):
    print(f"  Processing berka fold {fold_i}...")
    model_dir = BK_BASE / f"tabddpm_{fold_i}"
    model_path = model_dir / MODEL_SUBDIR
    
    fold_features = np.zeros([200, input_dimension])
    t_count = 0
    for t in TIMESTEPS:
        for at in ADDITIONAL_TIMESTEPS:
            preds = get_score(
                model_dir, model_path, bk_input_noise, MODEL_TYPE,
                meta_dir=model_dir,
                challenge_name="challenge_with_id.csv",
                batch_size=200,
                parallel_batch=NUM_NOISE,
                additional_timestep=at, timestep=t
            )
            fold_features[:, t_count*NUM_NOISE:(t_count+1)*NUM_NOISE] = \
                preds.detach().squeeze().cpu().numpy()
            t_count += 1
    
    x_test_list.append(fold_features)
    # challenge_with_id has 100 members + 100 non-members
    y_test_list.append(np.concatenate([np.zeros(100), np.ones(100)]))

x_test = np.vstack(x_test_list)
y_test = np.concatenate(y_test_list)
print(f"Test data shape: {x_test.shape}")

x_te = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
model.eval()
with torch.no_grad():
    test_preds = torch.sigmoid(model(x_te)).cpu().numpy().squeeze()
x_test = np.vstack(x_test_list)
y_test = np.concatenate(y_test_list)
print(f"Test data shape: {x_test.shape}")
x_te = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
model.eval()
with torch.no_grad():
    test_preds = torch.sigmoid(model(x_te)).cpu().numpy().squeeze()

from sklearn.metrics import roc_curve
test_auc = roc_auc_score(y_test, test_preds)
fpr, tpr, _ = roc_curve(y_test, test_preds)
max_tpr_test = float(tpr[fpr <= 0.1][-1]) if any(fpr <= 0.1) else 0.0

train_fpr, train_tpr, _ = roc_curve(y_train, train_preds)
max_tpr_train = float(train_tpr[train_fpr <= 0.1][-1]) if any(train_fpr <= 0.1) else 0.0

print(f"\nDiabetes train AUC: {train_auc:.4f}, max TPR @ 10% FPR: {max_tpr_train:.4f}")
print(f"Berka test AUC: {test_auc:.4f}, max TPR @ 10% FPR: {max_tpr_test:.4f}")

os.makedirs("whitebox_cross_DI_berka/cross_attack_results", exist_ok=True)
with open("whitebox_cross_DI_berka/cross_attack_results/mia_performance.txt", "w") as f:
    f.write(f"Cross-dataset MIA (Train: Diabetes 1-10, Test: Berka 1-30)\n")
    f.write(f"Diabetes train AUC: {train_auc:.4f}, max TPR @ 10% FPR: {max_tpr_train:.4f}\n")
    f.write(f"Berka test AUC: {test_auc:.4f}, max TPR @ 10% FPR: {max_tpr_test:.4f}\n")
print("\nDone! Results saved.")

# Score distribution analysis
print("\n=== Attack Score Distribution (Berka test) ===")
print(f"  Min:    {test_preds.min():.6f}")
print(f"  Max:    {test_preds.max():.6f}")
print(f"  Mean:   {test_preds.mean():.6f}")
print(f"  Std:    {test_preds.std():.6f}")
print(f"  Median: {np.median(test_preds):.6f}")
print(f"  % > 0.5: {(test_preds > 0.5).mean()*100:.1f}%")
print(f"  % > 0.9: {(test_preds > 0.9).mean()*100:.1f}%")
print(f"  % < 0.1: {(test_preds < 0.1).mean()*100:.1f}%")
members_bk = test_preds[y_test == 1]
nonmembers_bk = test_preds[y_test == 0]
print(f"  Members mean:     {members_bk.mean():.6f}")
print(f"  Non-members mean: {nonmembers_bk.mean():.6f}")

print("\n=== Attack Score Distribution (Diabetes train) ===")
print(f"  Min:    {train_preds.min():.6f}")
print(f"  Max:    {train_preds.max():.6f}")
print(f"  Mean:   {train_preds.mean():.6f}")
print(f"  Std:    {train_preds.std():.6f}")
print(f"  Median: {np.median(train_preds):.6f}")
print(f"  % > 0.5: {(train_preds > 0.5).mean()*100:.1f}%")
members_di = train_preds[y_train == 1]
nonmembers_di = train_preds[y_train == 0]
print(f"  Members mean:     {members_di.mean():.6f}")
print(f"  Non-members mean: {nonmembers_di.mean():.6f}")
