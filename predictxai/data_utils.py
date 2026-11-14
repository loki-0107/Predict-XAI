import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import WINDOW_SIZE, STRIDE, RUL_CLIP, FAILURE_HORIZON

COLUMNS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]

def load_cmapss(train_path, test_path, rul_path):
    train = pd.read_csv(train_path, sep=r"\s+", header=None)
    test = pd.read_csv(test_path, sep=r"\s+", header=None)
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None)

    train.columns = COLUMNS[:train.shape[1]]
    test.columns = COLUMNS[:test.shape[1]]
    rul.columns = ["RUL"]

    train = train.dropna(axis=1, how="all")
    test = test.dropna(axis=1, how="all")
    return train, test, rul

def add_rul_and_labels(df, rul_df=None, is_train=True):
    df = df.copy()
    if is_train:
        max_cycle = df.groupby("unit")["cycle"].transform("max")
        df["RUL"] = (max_cycle - df["cycle"]).clip(upper=RUL_CLIP)
        df["failure_label"] = (df["RUL"] <= FAILURE_HORIZON).astype(int)
    else:
        # For test data, compute RUL consistently with training data:
        # RUL = max_cycle - current_cycle (same formula as training)
        # This ensures the model is tested on the same RUL range it was trained on
        df["max_cycle"] = df.groupby("unit")["cycle"].transform("max")
        df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=RUL_CLIP)
        df["failure_label"] = (df["RUL"] <= FAILURE_HORIZON).astype(int)
    return df

def preprocess(df, sensor_columns, scaler=None, fit=False):
    df = df.copy()
    # Moving-average denoising
    for col in sensor_columns:
        df[col] = df.groupby("unit")[col].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
    if fit:
        scaler = MinMaxScaler()
        df[sensor_columns] = scaler.fit_transform(df[sensor_columns])
    else:
        df[sensor_columns] = scaler.transform(df[sensor_columns])
    return df, scaler

def make_sequences(df, sensor_columns):
    X, y_class, y_rul, unit_ids = [], [], [], []
    for unit, group in df.groupby("unit"):
        values = group[sensor_columns].values
        labels = group["failure_label"].values
        ruls = group["RUL"].values
        for i in range(WINDOW_SIZE - 1, len(group), STRIDE):
            X.append(values[i - WINDOW_SIZE + 1:i + 1])
            y_class.append(labels[i])
            y_rul.append(ruls[i])
            unit_ids.append(unit)
    return np.array(X), np.array(y_class), np.array(y_rul), np.array(unit_ids)

def statistical_features(sequence):
    return np.array([
        sequence.mean(axis=0),
        sequence.std(axis=0),
        sequence.min(axis=0),
        sequence.max(axis=0),
        np.sqrt((sequence ** 2).mean(axis=0))
    ]).flatten()