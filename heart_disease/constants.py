"""Constants for heart-disease project."""

import pathlib

DATA_DIR_PATH = pathlib.Path(__file__).parents[1] / "data"

EXTERNAL_DATA_DIR_PATH = DATA_DIR_PATH / "external"
INTERIM_DATA_DIR_PATH = DATA_DIR_PATH / "interim"
PROCESSED_DATA_DIR_PATH = DATA_DIR_PATH / "processed"
RAW_DATA_DIR_PATH = DATA_DIR_PATH / "raw"

# Cleveland Clinic Foundation.
CLEVELAND_STR = "cleveland"
# Hungarian Institute of Cardiology, Budapest.
HUNGARIAN_STR = "hungarian"
# University Hospital, Zurich, Switzerland.
SWITZERLAND_STR = "switzerland"
# V.A. Medical Center, Long Beach, CA.
VA_STR = "va"

DATA_PATHS = {
    CLEVELAND_STR: RAW_DATA_DIR_PATH / f"processed.{CLEVELAND_STR}.data",
    # Note this is the reprocessed data.
    HUNGARIAN_STR: RAW_DATA_DIR_PATH / f"reprocessed.{HUNGARIAN_STR}.data",
    SWITZERLAND_STR: RAW_DATA_DIR_PATH / f"processed.{SWITZERLAND_STR}.data",
    VA_STR: RAW_DATA_DIR_PATH / f"processed.{VA_STR}.data",
}
