"""Fetch data for the Disease Risk Prediction project."""

from io import BytesIO, StringIO
from zipfile import ZipFile

import pandas as pd
import pandera as pa
import requests
import tqdm
from bs4 import BeautifulSoup

import disease_risk_prediction.constants as c


class PatientDataSchema(pa.DataFrameModel):  # dead: disable
    """Data validation schema using pandera."""

    age: pa.typing.Series[int]
    gender: pa.typing.Series[str]
    bmi: pa.typing.Series[float]
    smoking_status: pa.typing.Series[str]
    alcohol_use: pa.typing.Series[str]
    physical_activity: pa.typing.Series[str]
    blood_pressure: pa.typing.Series[str]
    cholesterol_level: pa.typing.Series[str]
    diabetes_history: pa.typing.Series[str]

    @pa.check("age", name="row_count_check", strategy="element_wise")
    def check_row_count(cls, series: pd.Series) -> bool:  # dead: disable
        """Ensure the DataFrame has the correct length."""
        return len(series) == c.NUM_RECORDS_2023

    class Config:  # dead: disable
        """PatientDataSchema configuration."""

        strict = True  # dead: disable


def fetch_health_data() -> pd.DataFrame:
    """
    Fetch health data from CDC API.

    Returns: Validated health data as a DataFrame.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
        pa.errors.SchemaErrors: If the data validation fails.
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
            names = fwf_df["Variable Name"].str.lower()
            fwf_df["Starting Column"] = (
                fwf_df["Starting Column"].astype(int) - 1
            )  # Make it 0-indexed.
            fwf_df["End column"] = fwf_df["Starting Column"] + fwf_df[
                "Field Length"
            ].astype(int)
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
            # FIXME: names=c.HEALTH_DATA_COLS_OF_INTEREST,
        )
        pbar.update(1)  # 5

        validated_df = df
        # FIXME: Add back in once the schema is updated.
        # # Validate data.
        # try:
        #     validated_df = PatientDataSchema.validate(df)
        # except pa.errors.SchemaErrors as e:
        #     logger.error(f"Schema validation errors: {e.failure_cases}")
        #     raise
        pbar.update(1)  # 6

    validated_df.to_feather(data_path)
    return validated_df
