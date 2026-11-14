import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, confusion_matrix, classification_report
)

from data_utils import load_cmapss, add_rul_and_labels, preprocess, make_sequences, statistical_features
from model import train_lstm_encoder, extract_features, train_gradient_boosting_heads
from explain import global_shap_explanation, local_lime_explanation
from config import TEST_SIZE, RANDOM_STATE

TRAIN_PATH = "data/train_FD001.txt"
TEST_PATH = "data/test_FD001.txt"
RUL_PATH = "data/RUL_FD001.txt"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    train_raw, test_raw, test_rul = load_cmapss(TRAIN_PATH, TEST_PATH, RUL_PATH)

    sensor_columns = [col for col in train_raw.columns if col.startswith("sensor_")]

    train_df = add_rul_and_labels(train_raw, is_train=True)
    test_df = add_rul_and_labels(test_raw, rul_df=test_rul, is_train=False)

    train_df, scaler = preprocess(train_df, sensor_columns, fit=True)
    test_df, _ = preprocess(test_df, sensor_columns, scaler=scaler, fit=False)

    X_train_seq, y_train_class, y_train_rul, _ = make_sequences(train_df, sensor_columns)
    X_test_seq, y_test_class, y_test_rul, _ = make_sequences(test_df, sensor_columns)

    X_train_stat = np.array([statistical_features(seq) for seq in X_train_seq])
    X_test_stat = np.array([statistical_features(seq) for seq in X_test_seq])

    encoder, lstm_model = train_lstm_encoder(X_train_seq, y_train_class, y_train_rul)

    X_train_features = extract_features(encoder, X_train_seq, X_train_stat)
    X_test_features = extract_features(encoder, X_test_seq, X_test_stat)

    classifier, regressor = train_gradient_boosting_heads(X_train_features, y_train_class, y_train_rul)

    y_pred_class = classifier.predict(X_test_features)
    y_pred_rul = regressor.predict(X_test_features)

    classification_metrics = {
        "accuracy": accuracy_score(y_test_class, y_pred_class),
        "precision": precision_score(y_test_class, y_pred_class),
        "recall": recall_score(y_test_class, y_pred_class),
        "f1_score": f1_score(y_test_class, y_pred_class)
    }

    regression_metrics = {
        "rmse": mean_squared_error(y_test_rul, y_pred_rul) ** 0.5,
        "mae": mean_absolute_error(y_test_rul, y_pred_rul)
    }

    results = {
        "failure_classification": classification_metrics,
        "rul_regression": regression_metrics
    }

    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print(json.dumps(results, indent=4))
    print(classification_report(y_test_class, y_pred_class))

    # Confusion matrix - show all labels even if one is missing
    cm = confusion_matrix(y_test_class, y_pred_class, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Failure Prediction Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Healthy", "Failure risk"])
    plt.yticks([0, 1], ["Healthy", "Failure risk"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center",
                     color="red", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # RUL scatter plot
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test_rul, y_pred_rul, alpha=0.4, s=12)
    plt.plot(
        [y_test_rul.min(), y_test_rul.max()],
        [y_test_rul.min(), y_test_rul.max()],
        "r--"
    )
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("PredictXAI RUL Estimation")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rul_scatter.png", dpi=150)
    plt.close()

    # Global SHAP explanation
    feature_names = [
        f"LSTM_{i}" for i in range(encoder.output_shape[-1])
    ] + [f"Stat_{i}" for i in range(X_train_stat.shape[1])]

    global_shap_explanation(
        classifier,
        X_test_features[:500],
        feature_names,
        save_path=f"{OUTPUT_DIR}/shap_summary.png"
    )

    # Local LIME explanation for one high-risk prediction
    risk_index = int(np.argmax(classifier.predict_proba(X_test_features)[:, 1]))
    lime_explanation = local_lime_explanation(
        classifier,
        X_train_features,
        X_test_features[risk_index],
        feature_names
    )

    with open(f"{OUTPUT_DIR}/lime_explanation.txt", "w") as f:
        f.write(str(lime_explanation.as_list()))

    # Save artifacts
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
    joblib.dump(classifier, f"{OUTPUT_DIR}/failure_classifier.pkl")
    joblib.dump(regressor, f"{OUTPUT_DIR}/rul_regressor.pkl")
    encoder.save(f"{OUTPUT_DIR}/lstm_encoder.keras")

    pd.DataFrame({
        "true_failure": y_test_class,
        "predicted_failure": y_pred_class,
        "true_RUL": y_test_rul,
        "predicted_RUL": y_pred_rul
    }).to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)

if __name__ == "__main__":
    main()