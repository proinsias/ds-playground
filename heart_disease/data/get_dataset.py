import os
import zipfile

import requests

import heart_disease.constants as hdc  # FIXME: Is this right?


def get_dataset() -> None:
    if all((file.is_file() for file in hdc.DATA_PATHS.values())):
        print("Data files already downloaded.")  # FIXME: Switch to logging or loguru or?
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
        with open(zip_file_path, 'wb') as zip_file:
            zip_file.write(response.content)
        print(f"Downloaded the zip file to {zip_file_path} .")
        
        # Extract the contents of the zip file.
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(hdc.RAW_DATA_DIR_PATH)
        print(f"Extracted files to {hdc.RAW_DATA_DIR_PATH} .")
    else:
        print(f"Failed to download the zip file. Status code: {response.status_code}.")


if __name__ == '__main__':
    get_dataset()
