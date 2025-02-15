"""Constants used across the Disease Risk Prediction project."""

import pathlib

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

# FIXME: Replace HEALTH_DATA_COLS_OF_INTEREST with list of PatientDataSchema variables.
HEALTH_DATA_COLS_OF_INTEREST = [  # dead: disable
    # See for codebook:
    # https://www.cdc.gov/brfss/annual_data/2023/zip/codebook23_llcp-v2-508.zip
    #
    # FIXME: Can I do this filtering via Pandera without showing error?
    #
    # Features / flags.
    # Apply all filters together.
    #
    # Overall.
    "dispcode",  # FIXME: Filter out incomplete surveys. Keep only values '1100'. Then drop.
    # Demographics.
    "_state",  # FIXME: import us, us.states.lookup('24'), us.states.lookup('MD')
    "_sex",  # FIXME: See codebook for how to derive from inputs.
    "educa",  # FIXME: Filter out 9, BLANK.
    "marital",  # FIXME: Filter out 9, BLANK.
    "veteran3",  # FIXME: Filter out 7, 9, BLANK.
    "income3",  # FIXME: Filter out 77, 99, BLANK.
    "employ1",  # FIXME: Filter out 9, BLANK.
    "children",  # FIXME: Filter out 99, BLANK. Set 88 to 0.
    "firearm5",  # FIXME: Filter out 7, 9, BLANK.
    # Medical.
    "wtkg3",  # FIXME: Filter out BLANK.  # FIXME: Ask for weight in lbs and convert.
    "htm4",  # FIXME: Filter out BLANK.  # FIXME: Ask for height in feet and inches and convert.
    "physhlth",  # FIXME: Filter out 77, 99, BLANK. Set 88 to 0.
    "menthlth",  # FIXME: Filter out 77, 99, BLANK. Set 88 to 0.
    "genhlth",  # FIXME: Filter out 7, 9, BLANK.
    "smoke100",  # FIXME: Filter out 7, 9, BLANK.
    "smokday2",  # FIXME: Filter out 7, 9, BLANK.
    "usenow3",  # FIXME: Filter out 7, 9, BLANK.
    "checkup1",  # FIXMEL: Filter out 7, 9, BLANK. Set 8 to 4.
    #
    # Target variables.
    # Filter out these individually for different models.
    #
    # Asthma.
    "asthma3",  # FIXME: Filter out 7, 9, BLANK.
    "asthnow",  # FIXME: Filter out 7, 9, BLANK.
    # Arthritis.
    "_drdxar2",  # FIXME: Filter out BLANK.
    # Cancer.
    "chcscnc1",  # FIXME: Filter out 7, 9, BLANK.
    "chcocnc1",  # FIXME: Filter out 7, 9, BLANK.
    # Coronary heart disease (CHD) or myocardial infarction (MI).
    "_michd",  # FIXME: Filter out BLANK.
    # Diabetes.
    "diabete4",  # FIXME: Filter out 7, 9, BLANK.
    # High blood pressure.
    "_rfhype6",  # FIXME: Filter out 9.
    # High cholesterol.
    "_rfchol3",  # FIXME: Filter out 9, BLANK.
    # Lung disease.
    "chccopd3",  # FIXME: Filter out 7, 9, BLANK.
    # Stroke.
    "cvdstrk3",  # FIXME: Filter out 7, 9, BLANK.
]
NUM_RECORDS_2023 = 433323  # 433,323 records for 2023.
RANDOM_STATE = 42

# addepev3
# chckdny2
# colncncr

# exeroft1 or
# exerhmm1
# deaf
# blind
# alcday4
# avedrnk3
# drnk3ge5
# maxdrnks
# flushot7
# flshtmy3
# pneuvac4
# shingle2
# hivtst7
# seatbelt
# drnkdri2
# covidpo1
# lcsfirst
# lcslast
# lcsnumcg
# lastsmk2
# stopsmk2
# somale
# sofemale
# trnsgndr
# marijan1
# lsatisfy
# emtsuprt
# sdlonely
# sdhemply
# foodstmp
# _hlthpl1
# _totinda
# maxvo21_
# padur1_
# pafreq1_
# _minac12
# pamin13_
# pa3min_ -> probably just this one!
# pa3vigm_
