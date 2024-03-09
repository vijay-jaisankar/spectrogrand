"""
    Save the URLs of the album covers pertaining to songs of a particular genre 
"""

import argparse
import os
import pandas as pd
import json
from typing import List
import requests
from tqdm import tqdm

# Check if URL exists
# @ref https://stackoverflow.com/a/69016995
def check_url_exists(url: str):
    """
    Checks if a url exists
    :param url: url to check
    :return: True if the url exists, false otherwise.
    """
    return requests.head(url, allow_redirects=True).status_code == 200


# Read the MUMU Amazon file and get the IDs of all songs in the selected genre
# Note: The cases must match as explicit string matching is performed for the column names
def get_ids_genre(csv_file_path:str, exact_genre_name:str) -> List[str]:
    # Load the csv file and limit the processed columns to `amazon_id` and `genre`
    df = pd.read_csv(
        csv_file_path,
        usecols = ["amazon_id", "genres"]
    )

    # Filter out the rows of the selected genre
    df = df[df["genres"] == exact_genre_name]
    return list(df["amazon_id"])

# Process the MUMU JSON file sequentially and extract the album cover URLs using the amazon_id values as fk
def get_album_urls(json_file_path:str, target_ids:List[str]) -> List[str]:
    # Load the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)
    assert data is not None or len(data) > 0

    # Remove duplicate target IDs
    target_ids = list(set(target_ids))

    # Create a map for all target_ids: O(1) access
    TARGET_ID_MAP = {}
    for _id in target_ids:
        TARGET_ID_MAP[_id] = True

    # Store the selected album cover URLs
    N = len(data)
    selected_urls = []
    # Process the JSON file and check if the ID is a target
    for i in tqdm(range(N)):
        current_id = data[i]["amazon_id"]
        if current_id in TARGET_ID_MAP:
            _url = data[i]["imUrl"]
            # Check if the URL is valid
            if check_url_exists(_url) is True:
                selected_urls.append(_url)
    return selected_urls

# Main pipeline
def main(csv_file_path:str, json_file_path:str, exact_genre_name:str, write_file_path:str) -> None:
    # Get list of selected IDs
    selected_ids = get_ids_genre(csv_file_path=csv_file_path, exact_genre_name=exact_genre_name)
    assert len(selected_ids) > 0

    # Get validated URLs
    validated_urls = get_album_urls(json_file_path=json_file_path, target_ids=selected_ids)

    # Write the URLs to a file
    with open(write_file_path, "w") as f:
        for _url in validated_urls:
            f.write(f"{_url}\n")

if __name__ == "__main__":
    # Set up argument parsing routine
    parser = argparse.ArgumentParser()
    parser.add_argument("csvpath", help="mumu amazon csv path")
    parser.add_argument("jsonpath", help="mumu amazon json path")
    parser.add_argument("genre", help="target genre string")
    parser.add_argument("outputpath", help="output file write full path")

    args = parser.parse_args()
    
    # Call the pipeline
    main(csv_file_path=args.csvpath, json_file_path=args.jsonpath, exact_genre_name=args.genre, write_file_path=args.outputpath)
    