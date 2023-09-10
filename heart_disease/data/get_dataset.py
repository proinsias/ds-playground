import os
import zipfile

import numpy as np
import pandas as pd
import requests

import heart_disease.constants as hdc


def get_dataset() -> None:
    # FIXME: Add docstring.
    if all(file.is_file() for file in hdc.DATA_PATHS.values()):
        print(
            "Data files already downloaded.",
        )  # FIXME: Switch to logging or loguru or?
        return

    # Ensure the output directory exists, create it if it doesn't.
    os.makedirs(hdc.RAW_DATA_DIR_PATH, exist_ok=True)

    # Construct the full path to save the zip file.
    zip_file_path = hdc.RAW_DATA_DIR_PATH / "heart_disease.zip"

    # Send an HTTP GET request to download the zip file
    response = requests.get(
        url="https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
        timeout=60,
    )

    # Check if the request was successful (status code 200).
    if response.status_code == 200:
        # Save the content of the response to the zip file.
        with open(zip_file_path, "wb") as zip_file:
            zip_file.write(response.content)
        print(f"Downloaded the zip file to {zip_file_path} .")

        # Extract the contents of the zip file.
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(hdc.RAW_DATA_DIR_PATH)
        print(f"Extracted files to {hdc.RAW_DATA_DIR_PATH} .")
    else:
        print(f"Failed to download the zip file. Status code: {response.status_code}.")


def validate_and_combine_dataset() -> pd.DataFrame:
    """Combine  and validate data from different sites.

    This validation is based on the metadata provided in the
    `heart-disease-names` file extracted from the original zip file.
    """
    cleveland_df = pd.read_csv(
        filepath_or_buffer=hdc.DATA_PATHS[hdc.CLEVELAND_STR],
        na_values=hdc.NA_VALUES,
        names=hdc.DATA_COLUMNS,
    )
    cleveland_df["source"] = "cleveland"
    if cleveland_df.shape != (303, 15):
        raise  # FIXME: Add custom exception.

    hungarian_df = pd.read_csv(
        filepath_or_buffer=hdc.DATA_PATHS[hdc.HUNGARIAN_STR],
        na_values=hdc.NA_VALUES,
        names=hdc.DATA_COLUMNS,
        sep=" ",
    )
    hungarian_df["source"] = "hungarian"
    # Note that the re-processed hungarian data has one more row
    # than the processed version.
    if hungarian_df.shape != (295, 15):
        raise  # FIXME: Add custom exception.
    # This row has a null value for 'num', so we drop it.
    mask = hungarian_df["num"].isna()
    hungarian_df = hungarian_df[~mask]
    if hungarian_df.shape != (294, 15):
        raise  # FIXME: Add custom exception.

    switzerland_df = pd.read_csv(
        filepath_or_buffer=hdc.DATA_PATHS[hdc.SWITZERLAND_STR],
        na_values=hdc.NA_VALUES,
        names=hdc.DATA_COLUMNS,
    )
    switzerland_df["source"] = "switzerland"
    if switzerland_df.shape != (123, 15):
        raise  # FIXME: Add custom exception.

    va_df = pd.read_csv(
        filepath_or_buffer=hdc.DATA_PATHS[hdc.VA_STR],
        na_values=hdc.NA_VALUES,
        names=hdc.DATA_COLUMNS,
    )
    va_df["source"] = "va"
    if va_df.shape != (200, 15):
        raise  # FIXME: Add custom exception.

    combined_df = pd.concat(
        objs=(
            cleveland_df,
            hungarian_df,
            switzerland_df,
            va_df,
        ),
        ignore_index=True,
    )

    if combined_df.shape != (
        len(cleveland_df) + len(hungarian_df) + len(switzerland_df) + len(va_df),
        15,
    ):
        raise  # FIXME: Add custom exception.

    # Verify that there are no values representing NaN left in DataFrame.
    s = pd.Series(combined_df.values.flatten())
    if sum(len(s[s == val]) for val in hdc.NA_VALUES) > 0:
        raise  # FIXME: Add custom exception.

    # Ensure class distribution is as expected.
    s = combined_df[["source", "num"]].value_counts(dropna=False)

    if not all(
        (
            s.loc[("cleveland", 0.0)] == 164,
            s.loc[("cleveland", 1.0)] == 55,
            s.loc[("cleveland", 2.0)] == 36,
            s.loc[("cleveland", 3.0)] == 35,
            s.loc[("cleveland", 4.0)] == 13,
            s.loc[("hungarian", 0.0)] == 188,
            s.loc[("hungarian", 1.0)] == 37,
            s.loc[("hungarian", 2.0)] == 26,
            s.loc[("hungarian", 3.0)] == 28,
            s.loc[("hungarian", 4.0)] == 15,
            s.loc[("switzerland", 0.0)] == 8,
            s.loc[("switzerland", 1.0)] == 48,
            s.loc[("switzerland", 2.0)] == 32,
            s.loc[("switzerland", 3.0)] == 30,
            s.loc[("switzerland", 4.0)] == 5,
            s.loc[("va", 0.0)] == 51,
            s.loc[("va", 1.0)] == 56,
            s.loc[("va", 2.0)] == 41,
            s.loc[("va", 3.0)] == 42,
            s.loc[("va", 4.0)] == 10,
        ),
    ):
        raise  # FIXME: Add custom exception.

    # Convert integer columns.

    integer_cols = [
        # Age in years.
        "age",
        # Resting blood pressure in mm Hg on admission to the hospital.
        "trestbps",
        # Serum cholestoral in mg/dl.
        "chol",
        # Maximum heart rate achieved.
        "thalach",
        # Number of major vessels (0-3) colored by flourosopy.
        "ca",
    ]

    if not all(
        # Verify that all values for these columns are integers represented as floats.
        combined_df[col].apply(lambda f: f.is_integer() or np.isnan(f)).all()
        for col in integer_cols
    ):
        raise  # FIXME: Add custom exception.

    for col in integer_cols:
        # Convert these columns to Int64Dtype that allows for nulls.
        combined_df[col] = combined_df[col].astype(pd.Int64Dtype())

    # Convert categoricals.

    combined_df["sex"] = (
        combined_df["sex"]
        .astype(
            dtype="category",
        )
        .cat.rename_categories(
            new_categories={
                0.0: "female",
                1.0: "male",
            },
        )
    )

    # Chest pain type.
    combined_df["cp"] = (
        combined_df["cp"]
        .astype(
            dtype="category",
        )
        .cat.rename_categories(
            new_categories={
                1.0: "typical angina",
                2.0: "atypical angina",
                3.0: "non-anginal pain",
                4.0: "asymptomatic",
            },
        )
    )

    # Exercise induced angina.
    combined_df["exang"] = (
        combined_df["exang"]
        .fillna(
            value=-1.0,
        )
        .astype(
            dtype="category",
        )
        .cat.rename_categories(
            new_categories={
                -1.0: "missing value",
                0.0: "no exercise induced angina",
                1.0: "exercise induced angina",
            },
        )
    )

    combined_df["thal"] = (
        combined_df["thal"]
        .fillna(
            value=-1.0,
        )
        .astype(
            dtype="category",
        )
        .cat.rename_categories(
            new_categories={
                -1.0: "missing value",
                3.0: "normal",
                6.0: "fixed defect",
                7.0: "reversable defect",
            },
        )
    )

    # Fasting blood sugar.
    combined_df["fbs"] = (
        combined_df["fbs"]
        .fillna(
            value=-1.0,
        )
        .astype(
            dtype="category",
        )
        .cat.rename_categories(
            new_categories={
                -1.0: "missing value",
                0.0: "low fasting blood sugar",
                1.0: "high fasting blood sugar",
            },
        )
    )

    # Resting electrocardiographic results.
    combined_df["restecg"] = (
        combined_df["restecg"]
        .fillna(
            value=-1.0,
        )
        .astype(
            dtype="category",
        )
        .cat.rename_categories(
            new_categories={
                -1.0: "missing value",
                0.0: "normal",
                1.0: "ST-T wave abnormality",
                2.0: "left ventricular hypertrophy",
            },
        )
    )

    # Slope of the peak exercise ST segment.
    combined_df["slope"] = (
        combined_df["slope"]
        .fillna(
            value=-1.0,
        )
        .astype(
            dtype="category",
        )
        .cat.rename_categories(
            new_categories={
                -1.0: "missing value",
                1.0: "upsloping",
                2.0: "flat",
                3.0: "downsloping",
            },
        )
    )

    # Create boolean target variable, whether or not heart disease is present.
    combined_df["target"] = (combined_df["num"] > 0).astype(int)
    return combined_df.drop(
        columns=[
            "num",
        ],
    )


if __name__ == "__main__":
    get_dataset()
    _ = validate_and_combine_dataset()
