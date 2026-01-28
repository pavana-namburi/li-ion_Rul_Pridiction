import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ================== CONFIG ==================

BASE_PATH = Path("data_splits")

CLIENTS = {
    "client_1": "B0005",
    "client_2": "B0006",
    "client_3": "B0007",
    "client_4": "B0025"
}

WINDOW_SIZE = 30

FEATURES = [
    "capacity",
    "ambient_temperature",
    "voltage_measured",
    "current_measured",
    "temperature_measured",
    "current_load",
    "voltage_load"
]

TARGET = "RUL"

# ================== LOADERS ==================

def load_csv_from_dir(dir_path: Path) -> pd.DataFrame:
    files = list(dir_path.glob("*.csv"))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one CSV file in {dir_path}, found {len(files)}"
        )
    return pd.read_csv(files[0]).reset_index(drop=True)

# ================== PREPROCESS HELPERS ==================

def apply_scaler(df, scaler):
    df = df.copy()
    df[FEATURES] = scaler.transform(df[FEATURES])
    return df


def create_sliding_windows(df, features, target, window):
    X, y = [], []
    data = df[features].values
    rul = df[target].values

    for i in range(window - 1, len(df)):
        X.append(data[i - window + 1 : i + 1])
        y.append(rul[i])

    return np.array(X), np.array(y)

# ================== CLIENT-SIDE PROCESSING ==================

all_clients_data = {}

for client, battery_id in CLIENTS.items():
    print(f"\nProcessing {client} ({battery_id})")

    client_path = BASE_PATH / client

    train_df = load_csv_from_dir(client_path / "train")
    val_df   = load_csv_from_dir(client_path / "val")
    test_df  = load_csv_from_dir(client_path / "local_test")

    # Fit MinMax ONLY on local train
    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURES])

    # Apply scaler
    train_df = apply_scaler(train_df, scaler)
    val_df   = apply_scaler(val_df, scaler)
    test_df  = apply_scaler(test_df, scaler)

    # Sliding windows
    X_train, y_train = create_sliding_windows(train_df, FEATURES, TARGET, WINDOW_SIZE)
    X_val, y_val     = create_sliding_windows(val_df, FEATURES, TARGET, WINDOW_SIZE)
    X_test, y_test   = create_sliding_windows(test_df, FEATURES, TARGET, WINDOW_SIZE)

    all_clients_data[client] = {
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

    print(
        f"Train {X_train.shape} | "
        f"Val {X_val.shape} | "
        f"Local Test {X_test.shape}"
    )

# ================== GLOBAL DATASETS ==================

print("\nProcessing global validation (B0018)")
global_val_df = load_csv_from_dir(BASE_PATH / "global_val" / "B0018")

print("Processing global test (B0026)")
global_test_df = load_csv_from_dir(BASE_PATH / "global_test" / "B0026")

global_data = {
    "global_val": {},
    "global_test": {}
}

for client, data in all_clients_data.items():
    scaler = data["scaler"]

    # Global validation
    gv_scaled = apply_scaler(global_val_df, scaler)
    X_gv, y_gv = create_sliding_windows(
        gv_scaled, FEATURES, TARGET, WINDOW_SIZE
    )

    global_data["global_val"][client] = {
        "X": X_gv,
        "y": y_gv
    }

    # Global test
    gt_scaled = apply_scaler(global_test_df, scaler)
    X_gt, y_gt = create_sliding_windows(
        gt_scaled, FEATURES, TARGET, WINDOW_SIZE
    )

    global_data["global_test"][client] = {
        "X": X_gt,
        "y": y_gt
    }

    print(
        f"{client} → "
        f"Global Val {X_gv.shape} | "
        f"Global Test {X_gt.shape}"
    )

print("\n✅ All clients + global datasets preprocessed successfully.")

OUTPUT_DIR = Path("processed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Save client data
for client, data in all_clients_data.items():
    np.savez(
        OUTPUT_DIR / f"{client}_data.npz",
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        X_test=data["X_test"],
        y_test=data["y_test"]
    )

# Save global data
for split in global_data:
    for client, data in global_data[split].items():
        np.savez(
            OUTPUT_DIR / f"{client}_{split}.npz",
            X=data["X"],
            y=data["y"]
        )

print("✅ Preprocessed data saved to disk.")
