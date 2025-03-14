"""Fetch data for the Disease Risk Prediction project."""

from io import BytesIO, StringIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pandera as pa
import requests
import tqdm
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin

import disease_risk_prediction.constants as c


class PatientDataSchema(pa.DataFrameModel):
    """Data validation schema using pandera.

    See this url for codebook:
    https://www.cdc.gov/brfss/annual_data/2023/zip/codebook23_llcp-v2-508.zip
    """

    #
    # Features / flags.
    # Apply all filters together.
    #
    # Overall.
    #
    # Keep only completed surveys.
    dispcode: pa.typing.Series[int] = pa.Field(isin=[1100])
    #
    # Demographics.
    #
    state: pa.typing.Series[int] = pa.Field(
        isin=list(c.US_STATES_FIPS.keys()),
    )
    sex: pa.typing.Series[int] = pa.Field(isin=[1, 2])
    educa: pa.typing.Series[float] = pa.Field(ge=1, le=6)
    marital: pa.typing.Series[float] = pa.Field(ge=1, le=6)
    veteran3: pa.typing.Series[float] = pa.Field(isin=[1, 2])
    income3: pa.typing.Series[float] = pa.Field(ge=1, le=11)
    employ1: pa.typing.Series[float] = pa.Field(ge=1, le=8)
    children: pa.typing.Series[int] = pa.Field(ge=0, le=30)
    #
    # Medical.
    #
    wtkg3: pa.typing.Series[float] = pa.Field(ge=2300, le=29500)
    htm4: pa.typing.Series[float] = pa.Field(ge=91, le=244)
    physhlth: pa.typing.Series[int] = pa.Field(ge=0, le=30)
    menthlth: pa.typing.Series[float] = pa.Field(ge=0, le=30)
    genhlth: pa.typing.Series[float] = pa.Field(ge=1, le=5)
    smoke100: pa.typing.Series[float] = pa.Field(isin=[1, 2])
    checkup1: pa.typing.Series[int] = pa.Field(ge=1, le=4)
    flushot7: pa.typing.Series[float] = pa.Field(isin=[1, 2])
    pneuvac4: pa.typing.Series[float] = pa.Field(isin=[1, 2])

    class Config:  # dead: disable
        """PatientDataSchema configuration."""

        coerce = True  # dead: disable
        drop_invalid_rows = True  # dead: disable
        strict = False  # dead: disable


def fetch_health_data() -> pd.DataFrame:
    """
    Fetch health data from CDC API.

    Returns: CDC health data as a DataFrame.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
    """
    data_path = c.DATA_DIR / "health.feather"
    if data_path.exists():
        return pd.read_feather(data_path)

    with tqdm.tqdm(total=6, desc="Fetching health data") as pbar:
        url = "https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023ASC.zip"
        response = requests.get(url, timeout=30)
        pbar.update(1)  # 1

        if response.status_code == 200:
            # Extract data from zip as string.
            with ZipFile(BytesIO(response.content)).open("LLCP2023.ASC ") as file:
                data = file.read().decode("utf-8")
            pbar.update(1)  # 2

        else:
            raise requests.exceptions.RequestException(
                f"API request failed with status {response.status_code}",
            )

        # Get fixed-width format table.
        table_url = "https://www.cdc.gov/brfss/annual_data/2023/llcp_varlayout_23_onecolumn.html"
        table_response = requests.get(table_url, timeout=30)
        pbar.update(1)  # 3

        # Extract fixed-width format table.
        if table_response.status_code == 200:
            soup = BeautifulSoup(table_response.text, "html.parser")
            table = soup.find("table")
            rows = table.find_all("tr")

            # Extract table data
            table_data = []
            headers = []
            for i, row in enumerate(rows):
                if i == 0:
                    headers = [ele.text.strip() for ele in row.find_all("th")]
                else:
                    cols = [ele.text.strip() for ele in row.find_all("td")]
                    table_data.append([ele for ele in cols if ele])

            fwf_df = pd.DataFrame(table_data, columns=headers)
            fwf_df["Starting Column"] = (
                fwf_df["Starting Column"].astype(int) - 1
            )  # Make it 0-indexed.
            fwf_df["End column"] = fwf_df["Starting Column"] + fwf_df[
                "Field Length"
            ].astype(int)
            names = fwf_df["Variable Name"].str.lower().str.replace("_", "")
            colspecs = fwf_df[["Starting Column", "End column"]].values.tolist()

            pbar.update(1)  # 4
        else:
            raise requests.exceptions.RequestException(
                f"Table extraction failed with status {table_response.status_code}",
            )

        df = pd.read_fwf(
            StringIO(data),  # here data is a string.
            colspecs=colspecs,
            names=names,
        )
        pbar.update(1)  # 5

        if len(df) != c.NUM_RECORDS_2023:
            logger.error(f"DataFrame has {len(df)} (not {c.NUM_RECORDS_2023}) rows!!!")

        df.to_feather(data_path)
        pbar.update(1)  # 6

        return df


class HealthDataValidator(BaseEstimator, TransformerMixin):
    """
    Custom transformer to validate health data, with parallel processing support.
    """

    def __init__(
        self,
        n_jobs: int | None = -1,
    ):
        """
        Initialize the transformer.

        Args:
            n_jobs: Number of CPU cores to use. -1 uses all available cores.
        """
        self.n_jobs = n_jobs

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "HealthDataValidator":
        return self  # No fitting needed

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate health data.

        Args:
            X: DataFrame containing health data.

        Returns:
            Validated health data as a DataFrame.
        """
        col_conversions = {
            "children": {
                "replace_val": 88,
                "new_val": 0,
            },
            "menthlth": {
                "replace_val": 88,
                "new_val": 0,
            },
            "physhlth": {
                "replace_val": 88,
                "new_val": 0,
            },
            "checkup1": {
                "replace_val": 8,
                "new_val": 4,
            },
        }
        col_conversions = {k: v for k, v in col_conversions.items() if k in X.columns}
        ss = Parallel(
            n_jobs=self.n_jobs,
        )(
            delayed(self._convert_column)(
                X=X,
                col=k,
                dtype="Int64",
                replace_val=v["replace_val"],
                new_val=v["new_val"],
            )
            for k, v in col_conversions.items()
        )
        X[list(col_conversions.keys())] = pd.concat(
            objs=ss,
            axis="columns",
        )

        # FIXME: Update PatientDataSchema to keep only the data we ask the user in predictions.
        cols = list(PatientDataSchema.__annotations__.keys())
        valid_health_df = PatientDataSchema.validate(X[cols], lazy=True)

        valid_health_df[["state_latitude", "state_longitude"]] = Parallel(
            n_jobs=self.n_jobs,
        )(
            delayed(self._get_state_coordinates)(state)
            for state in valid_health_df["state"]
        )

        valid_health_df = valid_health_df.drop(
            columns=[
                "dispcode",
                "state",
            ],
            errors="ignore",
        ).convert_dtypes()

        valid_health_df[c.CATEGORICAL_COLS] = valid_health_df[
            c.CATEGORICAL_COLS
        ].astype("category")

        return valid_health_df

    @staticmethod
    def _convert_column(
        X: pd.DataFrame,
        col: str,
        dtype: str,
        replace_val: int,
        new_val: int,
    ) -> pd.Series:
        """
        Convert column to a specific type and replace a value.

        Args:
            X: DataFrame.
            col: Column name.
            dtype: Target data type.
            replace_val: Value to replace.
            new_val: New value.

        Returns:
            Transformed column.
        """
        return X[col].astype(dtype).replace(replace_val, new_val)

    @staticmethod
    def _get_state_coordinates(
        state: str,
    ) -> pd.Series:
        """
        Get state coordinates from mappings.

        Args:
            state: State code.

        Returns:
            Latitude and longitude as a Series.
        """
        return pd.Series(
            [
                c.US_STATES_COORDINATES[c.US_STATES_FIPS[state]][0],
                c.US_STATES_COORDINATES[c.US_STATES_FIPS[state]][1],
            ]
        )


class HealthTrainingDataValidator(BaseEstimator, TransformerMixin):
    """
    Custom transformer to validate health target data.
    """

    def __init__(
        self,
        n_jobs: int | None = -1,
    ):
        """
        Initialize the transformer.

        Args:
            n_jobs: Number of CPU cores to use. -1 uses all available cores.
        """
        self.n_jobs = n_jobs

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "HealthTrainingDataValidator":
        return self  # No fitting needed

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate health target data.

        Args:
            X: DataFrame containing health target data.

        Returns:
            Validated health target data as a DataFrame.
        """
        cols = [
            "asthms1",  # Asthma.
            "drdxar2",  # Arthritis.
            "chcscnc1",  # Cancer.
            "chcocnc1",  # Cancer.
            "michd",  # Coronary heart disease (CHD) or myocardial infarction (MI).
            "addepev3",  # Depression.
            "diabete4",  # Diabetes.
            "rfhype6",  # High blood pressure.
            "rfchol3",  # High cholesterol.
            "chckdny2",  # Kidney disease.
            "chccopd3",  # Lung disease.
            "cvdstrk3",  # Stroke.
        ]

        training_df = (
            X[cols]
            .assign(cancer=X.apply(self._determine_cancer_status, axis="columns"))
            .drop(
                columns=[
                    "chcscnc1",  # Cancer.
                    "chcocnc1",  # Cancer.
                ],
            )
            .convert_dtypes()
        )

        ss = Parallel(
            n_jobs=self.n_jobs,
        )(
            delayed(self._binarize_targets)(
                X=training_df,
                disease_col=col,
                keep_values=values["keep_values"],
                ones=values["ones"],
                zeros=values["zeros"],
            )
            for col, values in c.TRAINING_PREP_DATA.items()
        )

        training_df[list(c.TRAINING_PREP_DATA.keys())] = pd.concat(
            objs=ss,
            axis="columns",
        )

        if sorted(list(training_df.columns)) != sorted(c.Y_COLS):
            logger.error(
                "DataFrame has incorrect columns!!!",
            )

        return training_df

    @staticmethod
    def _determine_cancer_status(
        row: pd.Series,
    ) -> float:
        """
        Determine the cancer status of a row.

        Args:
            row: A row of the DataFrame.

        Returns:
            Cancer status as a string.
        """
        if pd.notna(row["chcscnc1"]) and np.isclose(row["chcscnc1"], 1.0):
            return 1.0
        elif pd.notna(row["chcocnc1"]) and np.isclose(row["chcocnc1"], 1.0):
            return 1.0
        elif (
            pd.notna(row["chcscnc1"])
            and np.isclose(row["chcscnc1"], 2.0)
            and pd.notna(row["chcocnc1"])
            and np.isclose(row["chcocnc1"], 2.0)
        ):
            return 2.0
        return 7.0

    @staticmethod
    def _binarize_targets(
        X: pd.DataFrame,
        disease_col: str,
        keep_values: list[int],
        ones: list[int],
        zeros: list[int],
    ) -> pd.Series:
        """Binarize target values for input disease."""
        y = X[disease_col].astype(float).astype(pd.Int8Dtype())

        logger.info(f"Initial {disease_col} value counts:")
        logger.info(y.value_counts(dropna=False).to_string())

        y.loc[~y.isin(keep_values)] = pd.NA
        y.loc[y.isin(ones)] = 1  # Has or has had disease.
        y.loc[y.isin(zeros)] = 0  # Never has had disease.

        logger.info(f"\nUpdated {disease_col} value counts:")
        logger.info(y.value_counts(dropna=False).to_string())

        return y


# FIXME: import us, us.states.lookup('24'), us.states.lookup('MD') -> convert to fips
# FIXME: See codebook for how to deriv _sex from inputs.
# FIXME: Just as for 'cancer', not chcocnc1 and chcscnc1
# FIXME: Ask for weight in lbs and convert.
# FIXME: Ask for height in feet and inches and convert.

# FIXME: diabete4 Filter out 7, 9, BLANK.
# FIXME: _rfhype6 Filter out 9.
# FIXME: _rfchol3 Filter out 9, BLANK.
# FIXME: chckdny2 Filter out 7, 9, BLANK.
# FIXME: chccopd3 Filter out 7, 9, BLANK.
# FIXME: cvdstrk3 Filter out 7, 9, BLANK.
# FIXME: chcscnc1 Filter out 7, 9, BLANK.
# FIXME: chcocnc1 Filter out 7, 9, BLANK.

# FIXME: Add extra features and target variables - below:

# alcday4
# avedrnk3
# drnk3ge5
# maxdrnks

# exeroft1 or
# exerhmm1
# _hlthpl1
# _totinda
# padur1_
# pafreq1_
# _minac12
# pamin13_
# pa3min_ -> probably just this one!
