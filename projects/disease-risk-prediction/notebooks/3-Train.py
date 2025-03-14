# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
"""Training."""

# %% [markdown]
# ## Imports

# %%
import gc
import warnings
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import sklearn.preprocessing
import sklearn.compose
import sklearn.dummy
import xgboost as xgb
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from IPython.core.interactiveshell import InteractiveShell
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest, VarianceThreshold, SelectorMixin
from sklearn.metrics import adjusted_rand_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

import disease_risk_prediction.constants as c
from disease_risk_prediction.data import (
    fetch_health_data,
)
from disease_risk_prediction.preprocess import (
    get_training_df,
)
from disease_risk_prediction.train import build_model, get_X_y_df


# %%
matplotlib.use("nbagg")
# prettier plots
plt.style.use("ggplot")
# larger plots - two different ways.
matplotlib.rc("figure", figsize=(15, 10))
plt.rcParams["figure.dpi"] = 90

# larger fonts
sns.set_context("notebook", font_scale=1.5)


# %%
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
pd.set_option("display.max_rows", 500)
pd.options.display.max_columns = 50
pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 80
# Adjust the number of columns profiled and displayed by the `info()` method.
pd.options.display.max_info_columns = 150
# Adjust the number of decimals to be displayed in a DataFrame.
pd.options.display.precision = 15
# Adjust the display format in a DataFrame.
# pd.options.display.float_format = '{:.2f}%'.format
# Prints and parses dates with the year first.
pd.options.display.date_yearfirst = True

InteractiveShell.ast_node_interactivity = "all"


# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Load training datasets

# %%
training_df = get_training_df(None, None)  # Load landed file.

# %%
training_df.shape
training_df.head()

# %%
disease = 'asthms1'
X, y, disease_df = get_X_y_df(training_df, disease)


# %% [markdown]
# ## Drop features with low signal with target

# %%
class MutualInfoThresholdSelector(SelectorMixin, BaseEstimator):
    """
    Feature selector that removes features with mutual information (MI) below a given threshold.

    Parameters:
    - threshold (float): Minimum MI score a feature must have to be kept.
    - discrete_features (bool or 'auto'): Whether features are discrete.
    - random_state (int, optional): Random state for reproducibility.

    Attributes:
    - mi_scores_ (np.ndarray): Mutual information scores for each feature.
    """

    # FIXME: threshold is very low, but I don't want to throw away too many features just yet.
    def __init__(self, threshold: float = 0.001, discrete_features: str | bool = 'auto', random_state: int | None = None):
        self.threshold = threshold
        self.discrete_features = discrete_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MutualInfoThresholdSelector':
        """
        Compute mutual information scores and determine which features to keep.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target array.

        Returns:
        - self: Fitted selector.
        """
        self.mi_scores_ = mutual_info_classif(X, y, discrete_features=self.discrete_features, random_state=self.random_state)
        return self

    def _get_support_mask(self) -> np.ndarray:
        """
        Generate a boolean mask indicating which features to keep.

        Returns:
        - np.ndarray: Boolean array where True indicates a feature is kept.
        """
        return self.mi_scores_ >= self.threshold

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce feature matrix to selected features.

        Parameters:
        - X (np.ndarray): Feature matrix.

        Returns:
        - np.ndarray: Reduced feature matrix.
        """
        return X[:, self._get_support_mask()]



# %%
mi = MutualInfoThresholdSelector()
X_mi = mi.fit_transform(X_vif.to_numpy(), y)

# %%
X_vif.shape, X_mi.shape

# %% [markdown]
# ## Sample targets

# %% [markdown]
# ### Undersampling

# %%
X_rus, y_rus = RandomUnderSampler(
    sampling_strategy='auto',
    random_state=c.RANDOM_STATE,
).fit_resample(X, y)

print(f"Class distribution before RandomUnderSampler: {Counter(y)}")
print(f"Class distribution after RandomUnderSampler: {Counter(y_rus)}")

# %% [markdown]
# ### Oversampling

# %%
X_smote, y_smote = SMOTE(
    sampling_strategy='auto',
    random_state=c.RANDOM_STATE,
).fit_resample(X, y)

print(f"Class distribution before SMOTE: {Counter(y)}")
print(f"Class distribution after SMOTE: {Counter(y_smote)}")

# %% [markdown]
# ### Under- and over-sampling

# %%
# %%time
X_smt, y_smt = SMOTETomek(
    sampling_strategy='auto',
    random_state=c.RANDOM_STATE,
    n_jobs=-1,
).fit_resample(X, y)

print(f"Class distribution before SMT: {Counter(y)}")
print(f"Class distribution after SMT: {Counter(y_smt)}")

# %%

# %%

# %%
pipeline = Pipeline([
    ('scaler', StandardScaler()),            # Standardize the features
    ('smote_tomek', SMOTETomek(
    sampling_strategy='auto',
    random_state=c.RANDOM_STATE,
    n_jobs=-1,
)),  # Apply SMOTETomek
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Train the model
])


# %% [markdown]
# •	Try a shallow network first (1–3 hidden layers) and evaluate performance.
# •	If underfitting occurs (high training and validation error), experiment with adding more layers or increasing neurons per layer.
# •	Use techniques like cross-validation and early stopping to monitor generalization.

# %%
# use xgboost's feature importance! but first do hyper-parameter tuning of xgboost?

import xgboost as xgb
model = xgb.XGBClassifier(tree_method='gpu_hist')
model.fit(X_train, y_train)

# Try an auto ML model

# UPDATE README FLOWCHART!

# SAVE MODELS TO DISK!

# Try training with different numbers of features - see below for keras code

# add weights and biases?

# what metrics should I minimize when training for xgboost and keras? f1-score, or auc?
# try over and under sampling.
# also provide both balanced and unbalanced test set for final metrics.


# 1. Load your data (assuming X_asthma and y_asthma are defined)
X_train, X_test, y_train, y_test = train_test_split(
    X_asthma, y_asthma, test_size=0.2, random_state=42, stratify=y_asthma
)


# 4. Feature importance from Random Forest
def get_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Trains a Random Forest and returns feature importances."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

feature_importance = get_feature_importance(X_train, y_train)
print("\nTop 10 features by Random Forest importance:\n", feature_importance.head(10))


# 5. Select top features based on mutual information & feature importance -> maybe f_classify as well!
top_features = list(mi_scores.head(10).index) + list(feature_importance.head(10).index)
top_features = list(set(top_features))  # Remove duplicates
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

print(f"\nSelected {len(top_features)} top features:\n", top_features)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


# 3. Build and train a Keras model
def build_model(input_dim: int) -> Sequential:
    """Creates a simple Keras binary classification model."""
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model


# 4. Experiment with different feature set sizes
feature_set_sizes = [10, 20, 50, 100]

results = []

for size in feature_set_sizes:
    selected_features = list(set(mi_scores.head(size).index) | set(rf_importance.head(size).index))

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    model = build_model(input_dim=X_train_selected.shape[1])

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_selected,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    # Evaluate model
    y_pred = model.predict(X_test_selected).ravel()
    y_pred_labels = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred_labels)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred_labels)

    results.append({'Feature Count': size, 'Accuracy': acc, 'AUC': auc, 'F1-score': f1})
    print(f"Feature Count: {size} | Accuracy: {acc:.4f} | AUC: {auc:.4f} | F1-score: {f1:.4f}")


# 5. Summarize results
results_df = pd.DataFrame(results)
print("\nPerformance Summary:")
print(results_df.sort_values(by='AUC', ascending=False))

# %%

# %%

# %%
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from tensorflow.keras import layers

# Create an imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2,
                           weights=[0.95, 0.05], flip_y=0, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple Keras model
def create_model(learning_rate: float = 0.001) -> keras.models.Sequential:
    """Creates a simple fully-connected neural network.

    Args:
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        keras.models.Sequential: Compiled Keras model.
    """
    model = keras.models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Wrap the Keras model using KerasClassifier
keras_clf = KerasClassifier(model=create_model, learning_rate=0.001, epochs=10, batch_size=32, verbose=0)

# Build the sklearn pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),         # Standardize the features
    ('classifier', keras_clf)             # Keras model wrapped as sklearn classifier
])

# Cross-validation scores
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')

print(f"Cross-validated F1 scores: {cv_scores}")
print(f"Mean F1 score: {cv_scores.mean():.4f}")

# Fit the pipeline and evaluate on the test set
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\nClassification Report on Test Data:")
print(classification_report(y_test, y_pred))

# %%

# %%

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

model = Sequential(
    [
        # Input layer with batch normalization and L2 regularization
        Dense(64, activation="relu", input_shape=(input_shape,), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),

        # Hidden layer 1
        Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),

        # Hidden layer 2
        Dense(16, activation="relu", kernel_regularizer=l2(0.01)),
        BatchNormalization(),

        # Output layer
        Dense(1, activation="sigmoid"),  # Binary classification
    ]
)

# %%
# Adjust learning rate
learning_rate = 0.001  # Start with a common default, tune if needed
optimizer = Adam(learning_rate=learning_rate) # Use instead of optimizer="adam" in compile.

# Adjust batch size when fitting
batch_size = 64  # Try 32, 64, 128 and see what works best
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=batch_size,
    callbacks=[early_stopping],
    verbose=1
)

# %%
import wandb
from wandb.keras import WandbCallback

wandb.init(project="disease-risk-prediction")

model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[WandbCallback()]
)

# %%
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load sample data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Initialize W&B
wandb.init(project="sklearn-xgboost-tracking", name="random-forest-experiment")

# Define model and hyperparameters
params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}
wandb.config.update(params)

# Train model
model = RandomForestClassifier(**params)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log metrics
wandb.log({
    "accuracy": accuracy,
    "classification_report": classification_report(y_test, y_pred, output_dict=True)
})

# Finish the run
wandb.finish()

# %%
import wandb
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load sample data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Initialize W&B
wandb.init(project="sklearn-xgboost-tracking", name="xgboost-experiment")

# Define model and hyperparameters
params = {
    "objective": "multi:softmax",
    "num_class": 3,
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100
}
wandb.config.update(params)

# Train model
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log metrics
wandb.log({"accuracy": accuracy})

# Optionally log feature importance
wandb.log({"feature_importance": wandb.plot.bar(
    dict(zip(iris.feature_names, model.feature_importances_)),
    title="Feature Importance"
)})

# Finish the run
wandb.finish()

# %%
import shap
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap

# Load sample data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Initialize W&B
wandb.init(project="sklearn-xgboost-tracking", name="random-forest-with-shap")

# Define model and hyperparameters
params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}
wandb.config.update(params)

# Train model
model = RandomForestClassifier(**params)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log metrics
wandb.log({
    "accuracy": accuracy,
})

# SHAP explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary plot and log it in W&B
shap.summary_plot(shap_values, X_test, feature_names=iris.feature_names)
wandb.log({"shap_summary_plot": wandb.Image(shap.summary_plot(shap_values, X_test, feature_names=iris.feature_names))})

# Plot SHAP dependence plot for a particular feature and log it in W&B
shap.dependence_plot(0, shap_values[0], X_test, feature_names=iris.feature_names)
wandb.log({"shap_dependence_plot": wandb.Image(shap.dependence_plot(0, shap_values[0], X_test, feature_names=iris.feature_names))})

# Finish the run
wandb.finish()

# %%
import shap
import wandb
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load sample data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Initialize W&B
wandb.init(project="sklearn-xgboost-tracking", name="xgboost-with-shap")

# Define model and hyperparameters
params = {
    "objective": "multi:softmax",
    "num_class": 3,
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100
}
wandb.config.update(params)

# Train model
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log metrics
wandb.log({"accuracy": accuracy})

# SHAP explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary plot and log it in W&B
shap.summary_plot(shap_values, X_test, feature_names=iris.feature_names)
wandb.log({"shap_summary_plot": wandb.Image(shap.summary_plot(shap_values, X_test, feature_names=iris.feature_names))})

# Plot SHAP dependence plot for a particular feature and log it in W&B
shap.dependence_plot(0, shap_values[0], X_test, feature_names=iris.feature_names)
wandb.log({"shap_dependence_plot": wandb.Image(shap.dependence_plot(0, shap_values[0], X_test, feature_names=iris.feature_names))})

# Finish the run
wandb.finish()

# %%
import shap
import wandb
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Load and preprocess data
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Initialize W&B
wandb.init(project="keras-with-shap", name="keras-model-with-shap")

# Define and compile the Keras model
model = keras.Sequential([
    keras.layers.Dense(64, input_dim=X.shape[1], activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # 3 classes for Iris
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define model hyperparameters
params = {
    "epochs": 50,
    "batch_size": 32
}
wandb.config.update(params)

# Train the model
history = model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"], validation_data=(X_test, y_test))

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))

# Log metrics to W&B
wandb.log({"accuracy": accuracy})

# SHAP explainability
explainer = shap.KernelExplainer(model.predict, X_train)  # Using KernelExplainer for neural networks
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary plot and log it in W&B
shap.summary_plot(shap_values, X_test, feature_names=iris.feature_names)
wandb.log({"shap_summary_plot": wandb.Image(shap.summary_plot(shap_values, X_test, feature_names=iris.feature_names))})

# Plot SHAP dependence plot for a particular feature and log it in W&B
shap.dependence_plot(0, shap_values[0], X_test, feature_names=iris.feature_names)
wandb.log({"shap_dependence_plot": wandb.Image(shap.dependence_plot(0, shap_values[0], X_test, feature_names=iris.feature_names))})

# Finish the run
wandb.finish()

# %%
# https://docs.wandb.ai/guides/sweeps/sweep-config-keys/ -> can use bayesian, but have to specify max number of runs.

# %%
# https://wandb.ai/wandb_fc/articles/reports/What-Is-Bayesian-Hyperparameter-Optimization-With-Tutorial---Vmlldzo1NDQyNzcw

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from typing import Tuple, List


def create_keras_model(input_shape: int) -> Sequential:
    """
    Creates and compiles a simple Keras model.

    Args:
        input_shape (int): Number of features in the input data.

    Returns:
        Sequential: A compiled Keras model.
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def learning_curve_keras(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: List[float],
    epochs: int = 50,
    batch_size: int = 32
) -> Tuple[List[float], List[float]]:
    """
    Computes training and validation accuracy for different training set sizes.

    Args:
        model_fn (callable): Function to create a fresh Keras model.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        train_sizes (List[float]): Proportions of training data to use.
        epochs (int): Number of epochs to train each model.
        batch_size (int): Batch size for training.

    Returns:
        Tuple[List[float], List[float]]: Training and validation accuracies for each train size.
    """
    train_accuracies = []
    val_accuracies = []

    # Split into initial train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    for size in train_sizes:
        # Create a smaller training set based on the current size
        X_subtrain, _, y_subtrain, _ = train_test_split(X_train, y_train, train_size=size, random_state=42)

        model = model_fn(X.shape[1])

        # Fit the model
        history = model.fit(
            X_subtrain, y_subtrain,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
        )

        # Evaluate on training and validation sets
        train_preds = (model.predict(X_subtrain) > 0.5).astype(int)
        val_preds = (model.predict(X_val) > 0.5).astype(int)

        train_acc = accuracy_score(y_subtrain, train_preds)
        val_acc = accuracy_score(y_val, val_preds)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Train size: {size}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return train_accuracies, val_accuracies


def plot_learning_curve(train_sizes: List[float], train_acc: List[float], val_acc: List[float]) -> None:
    """
    Plots the learning curve.

    Args:
        train_sizes (List[float]): Proportions of training data used.
        train_acc (List[float]): Training accuracy scores.
        val_acc (List[float]): Validation accuracy scores.
    """
    plt.plot(train_sizes, train_acc, marker='o', label='Training Accuracy')
    plt.plot(train_sizes, val_acc, marker='o', label='Validation Accuracy')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Dummy dataset
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    train_sizes = np.linspace(0.1, 1.0, 10)
    train_acc, val_acc = learning_curve_keras(create_keras_model, X, y, train_sizes)

    plot_learning_curve(train_sizes, train_acc, val_acc)  # Use LearningCurveDisplay for xgboost


# %%
def compare_training(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=c.RANDOM_STATE,
        stratify=y,
    )

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

    class_weight_dict = dict(enumerate(class_weights))

    dummy = sklearn.dummy.DummyClassifier(strategy='prior', random_state=c.RANDOM_STATE)

    _ = dummy.fit(
        X_train,
        y_train,
    )

    print(classification_report(y_test, dummy.predict(X_test)))

    xmodel = xgb.XGBClassifier(
        random_state=c.RANDOM_STATE,
        scale_pos_weight=len(y_train[y_train == 0])/len(y_train[y_train == 1])
    )

    _ = xmodel.fit(
        X_train,
        y_train,
    )

    print(classification_report(y_test, xmodel.predict(X_test)))

    model = build_model(X_train.shape[1])

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
    )

    print(classification_report(y_test, pd.DataFrame(model.predict(X_test)).astype(int)))


# %%
If your classes aren’t well separated, you’ll likely see poor model performance — so increasing class separability can make a huge difference. Here are some practical strategies:

Feature Engineering:
	1.	Create Interaction Terms:
Combine features to capture relationships that single features miss.

X['new_feature'] = X['feature1'] * X['feature2']


https://levelup.gitconnected.com/4-python-libraries-for-automated-feature-engineering-that-you-should-use-in-2023-54bccecb1683
https://github.com/feature-engine/feature_engine
https://github.com/cod3licious/autofeat
https://github.com/alteryx/featuretools/


	2.	Polynomial Features:
Capture non-linear relationships by adding squared or cubic terms.

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)


	7.	Class Weights:
Penalize misclassification of the minority class more heavily.

model.fit(X, y, class_weight='balanced')

# %% [markdown]
# ## Train asthma

# %% jupyter={"source_hidden": true}
compare_training(X_asthma, y_asthma)

# %% [markdown]
# ## Train arthritis

# %%
ys["drdxar2"].value_counts()

# %%
y_arthritis = ys["drdxar2"].astype(float).astype(int)
mask = y_arthritis.isin([1, 2])

X_arthritis = X[mask]

y_arthritis = y_arthritis[mask]
# y_arthritis.loc[y_arthritis.isin([1])] = 1
y_arthritis.loc[y_arthritis == 2] = 0

# %%
y_arthritis.value_counts()

# %%
compare_training(X_arthritis, y_arthritis)

# %% [markdown]
# ## Train mi/chd

# %%
ys["michd"].value_counts()

# %%
y_michd = ys["michd"].astype(float).astype(int)
mask = y_michd.isin([1, 2])

X_michd = X[mask]

y_michd = y_michd[mask]
# y_michd.loc[y_michd.isin([1])] = 1
y_michd.loc[y_michd == 2] = 0

# %%
y_michd.value_counts()

# %%
compare_training(X_michd, y_michd)

# %% [markdown]
# ## Train depressive disorder

# %%
ys["addepev3"].value_counts()

# %%
y_addepev3 = ys["addepev3"].astype(float).astype(int)
mask = y_addepev3.isin([1, 2])

X_addepev3 = X[mask]

y_addepev3 = y_addepev3[mask]
# y_addepev3.loc[y_addepev3.isin([1])] = 1
y_addepev3.loc[y_addepev3 == 2] = 0

# %% jupyter={"source_hidden": true}
y_addepev3.value_counts()

# %%
compare_training(X_addepev3, y_addepev3)

# %% [markdown]
# ## Train diabetes

# %%
ys["diabete4"].value_counts()

# %%
y_diabete4 = ys["diabete4"].astype(float).astype(int)
mask = y_diabete4.isin([1, 2, 3, 4])

X_diabete4 = X[mask]

y_diabete4 = y_diabete4[mask]
y_diabete4.loc[y_diabete4.isin([1, 2])] = 1
y_diabete4.loc[y_diabete4.isin([3, 4])] = 0

# %%
y_diabete4.value_counts()

# %%
compare_training(X_diabete4, y_diabete4)

# %% [markdown]
# ## Train high blood pressure

# %%
ys["rfhype6"].value_counts()

# %%
y_rfhype6 = ys["rfhype6"].astype(float).astype(int)
mask = y_rfhype6.isin([1, 2])

X_rfhype6 = X[mask]

y_rfhype6 = y_rfhype6[mask]
# y_rfhype6.loc[y_rfhype6.isin([1])] = 1
y_rfhype6.loc[y_rfhype6.isin([2])] = 0

# %%
y_rfhype6.value_counts()

# %%
compare_training(X_rfhype6, y_rfhype6)

# %% [markdown]
# ## Train high cholestrol

# %%
ys["rfchol3"].value_counts()

# %%
y_rfchol3 = ys["rfchol3"].astype(float).astype(int)
mask = y_rfchol3.isin([1, 2])

X_rfchol3 = X[mask]

y_rfchol3 = y_rfchol3[mask]
# y_rfchol3.loc[y_rfchol3.isin([1])] = 1
y_rfchol3.loc[y_rfchol3.isin([2])] = 0

# %%
y_rfchol3.value_counts()

# %%
compare_training(X_rfchol3, y_rfchol3)

# %% [markdown]
# ## Train kidney disease

# %%
ys["chckdny2"].value_counts()

# %%
y_chckdny2 = ys["chckdny2"].astype(float).astype(int)
mask = y_chckdny2.isin([1, 2])

X_chckdny2 = X[mask]

y_chckdny2 = y_chckdny2[mask]
# y_chckdny2.loc[y_chckdny2.isin([1])] = 1
y_chckdny2.loc[y_chckdny2.isin([2])] = 0

# %%
y_chckdny2.value_counts()

# %%
compare_training(X_chckdny2, y_chckdny2)

# %% [markdown]
# ## Train lung disease

# %%
ys["chccopd3"].value_counts()

# %%
y_chccopd3 = ys["chccopd3"].astype(float).astype(int)
mask = y_chccopd3.isin([1, 2])

X_chccopd3 = X[mask]

y_chccopd3 = y_chccopd3[mask]
# y_chccopd3.loc[y_chccopd3.isin([1])] = 1
y_chccopd3.loc[y_chccopd3.isin([2])] = 0

# %%
y_chccopd3.value_counts()

# %%
compare_training(X_chccopd3, y_chccopd3)

# %% [markdown]
# ## Train stroke

# %%
ys["cvdstrk3"].value_counts()

# %%
y_cvdstrk3 = ys["cvdstrk3"].astype(float).astype(int)
mask = y_cvdstrk3.isin([1, 2])

X_cvdstrk3 = X[mask]

y_cvdstrk3 = y_cvdstrk3[mask]
# y_cvdstrk3.loc[y_cvdstrk3.isin([1])] = 1
y_cvdstrk3.loc[y_cvdstrk3.isin([2])] = 0

# %%
y_cvdstrk3.value_counts()

# %%
compare_training(X_cvdstrk3, y_cvdstrk3)

# %% [markdown]
# ## Train cancer

# %%
ys["cancer"].value_counts()

# %%
y_cancer = ys["cancer"].astype(float).astype(int)
mask = y_cancer.isin([1, 2])

X_cancer = X[mask]

y_cancer = y_cancer[mask]
# y_cancer.loc[y_cancer.isin([1])] = 1
y_cancer.loc[y_cancer.isin([2])] = 0

# %%
y_cancer.value_counts()

# %%
compare_training(X_cancer, y_cancer)

# %%

# %%

# %%

# %% [markdown]
# ## Notes

# %% [markdown]
# Best keras:
#
# - addepev3 - 0.71 -> is this also where i have plenty of 1s?
# - rfhype6
# - arthritis
# - X_diabete4
