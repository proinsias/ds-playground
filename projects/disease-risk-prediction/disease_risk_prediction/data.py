"""Fetch data for the Disease Risk Prediction project."""

from io import BytesIO, StringIO
from zipfile import ZipFile

import pandas as pd
import pandera as pa
import requests
import tqdm
from bs4 import BeautifulSoup
from loguru import logger

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
    #
    # Target variables.
    # Filter out these individually for different models.
    #
    # Asthma.
    asthms1: pa.typing.Series[str]
    # Arthritis.
    drdxar2: pa.typing.Series[str]
    # Cancer.
    chcscnc1: pa.typing.Series[str]
    chcocnc1: pa.typing.Series[str]
    # Coronary heart disease (CHD) or myocardial infarction (MI).
    michd: pa.typing.Series[str]
    # Depression.
    addepev3: pa.typing.Series[str]
    # Diabetes.
    diabete4: pa.typing.Series[str]
    # High blood pressure.
    rfhype6: pa.typing.Series[str]
    # High cholesterol.
    rfchol3: pa.typing.Series[str]
    # Kidney disease.
    chckdny2: pa.typing.Series[str]
    # Lung disease.
    chccopd3: pa.typing.Series[str]
    # Stroke.
    cvdstrk3: pa.typing.Series[str]

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


def validate_health_data(
    health_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate health data.

    Returns: Validated health data as a DataFrame.
    """
    cols = list(PatientDataSchema.__annotations__.keys())

    for col in [
        "children",
        "menthlth",
        "physhlth",
    ]:
        health_df[col] = health_df[col].astype("Int64").replace(88, 0)

    health_df["checkup1"] = health_df["checkup1"].astype("Int64").replace(8, 4)

    valid_health_df = PatientDataSchema.validate(health_df[cols], lazy=True)

    mask1 = (valid_health_df["chcscnc1"] == "1.0") | (
        valid_health_df["chcocnc1"] == "1.0"
    )
    valid_health_df.loc[
        mask1,
        "cancer",
    ] = "1.0"
    mask2 = (valid_health_df["chcscnc1"] == "2.0") & (
        valid_health_df["chcocnc1"] == "2.0"
    )
    valid_health_df.loc[
        mask2,
        "cancer",
    ] = "2.0"
    valid_health_df.loc[
        ~(mask1 | mask2),
        "cancer",
    ] = "7.0"

    valid_health_df["state_latitude"] = valid_health_df["state"].map(
        lambda x: c.US_STATES_COORDINATES[c.US_STATES_FIPS[x]][0],
    )
    valid_health_df["state_longitude"] = valid_health_df["state"].map(
        lambda x: c.US_STATES_COORDINATES[c.US_STATES_FIPS[x]][1],
    )

    valid_health_df = valid_health_df.drop(
        columns=[
            "dispcode",
            "chcscnc1",
            "chcocnc1",
            "state",
        ],
    )

    if len(valid_health_df) != c.NUM_VALID_RECORDS_2023:
        logger.error(
            f"DataFrame has {len(valid_health_df)} (not {c.NUM_VALID_RECORDS_2023}) rows!!!",
        )

    # Drop duplicates!

    valid_health_df = valid_health_df.drop_duplicates()

    return valid_health_df.convert_dtypes()


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
