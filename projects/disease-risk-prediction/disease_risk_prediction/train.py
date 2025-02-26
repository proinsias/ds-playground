"""Model building and training for the Disease Risk Prediction project."""

from keras.layers import Dense, Dropout
from keras.models import Sequential
from tensorflow import keras

# from sklearn.utils.class_weight import compute_class_weight


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

    # FIXME: Add extra metrics?
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            "auc",
            "binary_crossentropy",
            "f1_score",
        ],
    )
    return model


# FIXME: How choose optimizer?
# FIXME: Visualize features after pre-processing pipeline?
# FIXME: Ask GPT how I can visualize NN to figure out how to improve things.
# FIXME: What parameters of the build_model and model.compile and model.fit functions should I optimize? How best optimize? Need Google Colab?
# FIXME: Compare to baselines: dummy classifiers, xgboost?
# FIXME: How visualize (fitted with weights) keras NN?
# FIXME: Equivalent of training performance plots from scikit-learn?
# FIXME: yellowbrick useful for keras?


# FIXME: - train all models from here? only keep those that have ?? metric value above ???
# asthms1 - nope
# drdxar2 - a little better than xgboost
# michd -
# addepev3 -
# diabete4
# rfhype6
# rfchol3
# chckdny2
# chccopd3
# cvdstrk3
# cancer
