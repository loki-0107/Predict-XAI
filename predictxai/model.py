import numpy as np
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from config import (LSTM_UNITS, DROPOUT, BATCH_SIZE, EPOCHS, PATIENCE, GB_ESTIMATORS, GB_MAX_DEPTH, GB_LEARNING_RATE, LOSS_WEIGHT_CLASSIFICATION, LOSS_WEIGHT_RUL)

def build_lstm_encoder(input_shape):
    inputs = Input(shape=input_shape)
    x = inputs
    for units in LSTM_UNITS:
        x = LSTM(units, return_sequences=(units != LSTM_UNITS[-1]))(x)
        x = Dropout(DROPOUT)(x)
    latent = Dense(32, activation="relu", name="latent_features")(x)
    model = Model(inputs, latent, name="lstm_encoder")
    return model

def train_lstm_encoder(X_train, y_class_train, y_rul_train):
    encoder = build_lstm_encoder(input_shape=(X_train.shape[1], X_train.shape[2]))
    class_output = Dense(1, activation="sigmoid", name="failure_output")(encoder.output)
    rul_output = Dense(1, activation="linear", name="rul_output")(encoder.output)
    model = Model(encoder.input, [class_output, rul_output])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "failure_output": "binary_crossentropy",
            "rul_output": "mse"
        },
        loss_weights={
            "failure_output": LOSS_WEIGHT_CLASSIFICATION,
            "rul_output": LOSS_WEIGHT_RUL
        },
        metrics={
            "failure_output": ["accuracy"],
            "rul_output": ["mae"]
        }
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True
    )
    model.fit(
        X_train,
        {"failure_output": y_class_train, "rul_output": y_rul_train},
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    return encoder, model

def extract_features(encoder, X, statistical_features):
    lstm_features = encoder.predict(X, verbose=0)
    return np.concatenate([lstm_features, statistical_features], axis=1)

def train_gradient_boosting_heads(X_features, y_class, y_rul):
    classifier = GradientBoostingClassifier(
        n_estimators=GB_ESTIMATORS,
        max_depth=GB_MAX_DEPTH,
        learning_rate=GB_LEARNING_RATE,
        random_state=42
    )
    regressor = GradientBoostingRegressor(
        n_estimators=GB_ESTIMATORS,
        max_depth=GB_MAX_DEPTH,
        learning_rate=GB_LEARNING_RATE,
        random_state=42
    )
    classifier.fit(X_features, y_class)
    regressor.fit(X_features, y_rul)
    return classifier, regressor