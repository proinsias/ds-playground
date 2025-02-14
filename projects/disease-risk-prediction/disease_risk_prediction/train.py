"""Model building and training for the Disease Risk Prediction project."""

from keras.layers import Dense, Dropout
from keras.models import Sequential
from tensorflow import keras


def build_model(input_shape: int) -> keras.Model:
    """Compile a deep learning model."""
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),  # Binary classification (disease risk)
        ],
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
