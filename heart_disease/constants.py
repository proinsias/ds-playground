"""Constants for heart-disease project."""

import pathlib

DATA_DIR_PATH = pathlib.Path(__file__).parents[1] / "data"

EXTERNAL_DATA_DIR_PATH = DATA_DIR_PATH / "external"
INTERIM_DATA_DIR_PATH = DATA_DIR_PATH / "interim"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
RAW_DATA_DIR_PATH = DATA_DIR_PATH / "raw"

# Cleveland Clinic Foundation.
CLEVELAND_DATA_PATH = RAW_DATA_DIR_PATH / "processed.cleveland.data"
# Hungarian Institute of Cardiology, Budapest.
HUNGARIAN_DATA_PATH = RAW_DATA_DIR_PATH / "reprocessed.hungarian.data"
# University Hospital, Zurich, Switzerland.
SWITZERLAND_DATA_PATH = RAW_DATA_DIR_PATH / "processed.switzerland.data"
# V.A. Medical Center, Long Beach, CA.
VA_DATA_PATH = RAW_DATA_DIR_PATH / "processed.va.data"

DATA_PATHS = (
    CLEVELAND_DATA_PATH,
    HUNGARIAN_DATA_PATH,
    SWITZERLAND_DATA_PATH,
    VA_DATA_PATH,
)
