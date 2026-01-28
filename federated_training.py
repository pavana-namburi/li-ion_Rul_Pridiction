import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# ====================== CONFIG ======================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIENTS = ["client_1", "client_2", "client_3", "client_4"]
DATA_DIR = "processed_data"

WINDOW_SIZE = 30
NUM_FEATURES = 7

LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3

MAX_ROUNDS = 50
PATIENCE = 5
EPS = 1e-6
DELTA = 1e-4  # for early stopping threshold

# reproducibility
torch.manual_seed(42)
np.random.seed(42)

os.makedirs("outputs", exist_ok=True)

# ====================== MODEL ======================

class CNNLSTMAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv1d(NUM_FEATURES, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )

        self.attention = nn.Linear(64, 1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)          # (B, F, T)
        x = self.relu(self.conv(x))     # (B, 64, T)
        x = x.permute(0, 2, 1)          # (B, T, 64)

        lstm_out, _ = self.lstm(x)      # (B, T, 64)

        attn_scores = self.attention(lstm_out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context)

        return out.view(-1), attn_weights.squeeze(-1)  # keep attention shape (B, T)

# ====================== CLIENT ======================

class Client:
    def __init__(self, client_id):
        self.client_id = client_id
        self.data = np.load(f"{DATA_DIR}/{client_id}_data.npz")

    def train(self, global_model):
        model = copy.deepcopy(global_model).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.L1Loss()

        X = torch.from_numpy(self.data["X_train"]).float().to(DEVICE)
        y = torch.from_numpy(self.data["y_train"]).float().to(DEVICE)

        model.train()
        for _ in range(LOCAL_EPOCHS):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                preds, _ = model(X[idx])
                loss = loss_fn(preds, y[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model

    def local_validation_loss(self, model):
        X = torch.from_numpy(self.data["X_val"]).float().to(DEVICE)
        y = np.array(self.data["y_val"]).reshape(-1)

        model.eval()
        with torch.no_grad():
            preds, _ = model(X)
        return mean_absolute_error(y, preds.cpu().numpy())

    def local_test_metrics(self, model):
        X = torch.from_numpy(self.data["X_test"]).float().to(DEVICE)
        y = np.array(self.data["y_test"]).reshape(-1)

        model.eval()
        with torch.no_grad():
            preds, _ = model(X)
        mae = mean_absolute_error(y, preds.cpu().numpy())
        rmse = np.sqrt(mean_squared_error(y, preds.cpu().numpy()))
        return mae, rmse

# ====================== FEDERATED TRAINER ======================

class FederatedTrainer:
    def __init__(self):
        self.global_model = CNNLSTMAttention().to(DEVICE)
        self.clients = [Client(cid) for cid in CLIENTS]
        self.history = []
        self.fedwavg_weights = []

    def aggregate(self, client_models, weights):
    # Start from global model structure
        new_state = copy.deepcopy(self.global_model.state_dict())

        for k in new_state.keys():
            new_state[k] = torch.zeros_like(new_state[k])
            for i in range(len(client_models)):
                new_state[k] += weights[i] * client_models[i].state_dict()[k]

        self.global_model.load_state_dict(new_state)


    def global_validation(self):
        losses = []
        for cid in CLIENTS:
            data = np.load(f"{DATA_DIR}/{cid}_global_val.npz")
            X = torch.from_numpy(data["X"]).float().to(DEVICE)
            y = np.array(data["y"]).reshape(-1)

            self.global_model.eval()
            with torch.no_grad():
                preds, _ = self.global_model(X)
            losses.append(mean_absolute_error(y, preds.cpu().numpy()))
        return np.mean(losses)

    def global_test_metrics(self):
        metrics = {}
        for cid in CLIENTS:
            data = np.load(f"{DATA_DIR}/{cid}_global_test.npz")
            X = torch.from_numpy(data["X"]).float().to(DEVICE)
            y = np.array(data["y"]).reshape(-1)

            self.global_model.eval()
            with torch.no_grad():
                preds, _ = self.global_model(X)
            mae = mean_absolute_error(y, preds.cpu().numpy())
            rmse = np.sqrt(mean_squared_error(y, preds.cpu().numpy()))
            metrics[cid] = {"MAE": mae, "RMSE": rmse}
        return metrics

    def train(self):
        best_val = float("inf")
        patience_counter = 0

        for rnd in range(1, MAX_ROUNDS + 1):
            client_models = []
            val_losses = []

            for client in self.clients:
                local_model = client.train(self.global_model)
                loss = client.local_validation_loss(local_model)
                client_models.append(local_model)
                val_losses.append(loss)

            inv_losses = [1 / (l + EPS) for l in val_losses]
            weights = [v / sum(inv_losses) for v in inv_losses]

            self.fedwavg_weights.append(weights)
            self.aggregate(client_models, weights)

            global_val = self.global_validation()
            self.history.append([rnd, global_val])

            print(f"Round {rnd} | Global Val MAE: {global_val:.4f}")

            # Early stopping with delta threshold
            if global_val < best_val - DELTA:
                best_val = global_val
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("Early stopping triggered")
                    break

        self.save_outputs()

    def save_outputs(self):
        # Save global model
        torch.save(self.global_model.state_dict(), "outputs/global_model.pt")

        # Save training history
        pd.DataFrame(self.history, columns=["round", "global_val_mae"])\
            .to_csv("outputs/training_log.csv", index=False)

        # Save FedWAvg weights
        np.save("outputs/fedwavg_weights.npy", np.array(self.fedwavg_weights))

        # Save attention weights & local test metrics
        local_test_results = {}
        for client in self.clients:
            data = np.load(f"{DATA_DIR}/{client.client_id}_data.npz")
            X = torch.from_numpy(data["X_test"]).float().to(DEVICE)

            self.global_model.eval()
            with torch.no_grad():
                _, attn = self.global_model(X)
            np.save(f"outputs/attention_{client.client_id}.npy", attn.cpu().numpy())

            # Local test metrics
            mae, rmse = client.local_test_metrics(self.global_model)
            local_test_results[client.client_id] = {"MAE": mae, "RMSE": rmse}

        # Save local test metrics
        pd.DataFrame(local_test_results).T.to_csv("outputs/local_test_metrics.csv")

        # Save global test metrics
        global_test_metrics = self.global_test_metrics()
        pd.DataFrame(global_test_metrics).T.to_csv("outputs/global_test_metrics.csv")

# ====================== RUN ======================

if __name__ == "__main__":
    trainer = FederatedTrainer()
    trainer.train()
