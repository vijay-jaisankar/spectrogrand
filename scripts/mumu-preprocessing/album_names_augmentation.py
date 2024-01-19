"""
    Save the albums' names in the MuMu descriptor .csv file
"""

import argparse
import pandas as pd
import json
from tqdm import tqdm
from typing import Dict, List

# Read the MUMU Amazon file and create a consolidated mapping : {ID:Name} for a given list of IDS
def create_id_name_mapping(json_file_path:str, given_ids_list:List[str]) -> Dict:
    # Load the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)
    assert data is not None or len(data) > 0

    # Define and populate the mapping
    ID_NAME_MAPPING = {}

    N = len(data)
    for i in tqdm(range(N)):
        current_id = data[i]["imUrl"]
        if "title" not in data[i]:
            current_title = "undefined"
        else:
            current_title = data[i]["title"]
        if current_id in given_ids_list:
            if current_id not in ID_NAME_MAPPING:
                ID_NAME_MAPPING[current_id] = current_title

    assert len(list(ID_NAME_MAPPING.keys())) <= len(given_ids_list)
    return ID_NAME_MAPPING

# Main pipeline
def main(csv_file_path:str, json_file_path:str, write_file_csv_path:str) -> None:
    # Get list of selected URLs
    df = pd.read_csv(csv_file_path)
    
    # Parse the image names and remove the ".jpg" to form the IDs
    selected_ids = df["image_file_path"].apply(lambda x: f"http://ecx.images-amazon.com/images/I/{x}")
    selected_ids = list(selected_ids)
    # Get the ID:Name mapping
    global_mapping = create_id_name_mapping(json_file_path=json_file_path, given_ids_list=selected_ids)

    # Populate a new dataframe and save it to a new csv file
    all_album_names = []
    for _id in selected_ids:
        if _id not in global_mapping:
            all_album_names.append("undefined")
        else:
            all_album_names.append(global_mapping[_id])

    df["album_name"] = all_album_names
    df.to_csv(
        write_file_csv_path,
        index = False
    )

if __name__ == "__main__":
    # Set up argument parsing routine
    parser = argparse.ArgumentParser()
    parser.add_argument("csvpath", help="intermediate processing csv path")
    parser.add_argument("jsonpath", help="mumu amazon json path")
    parser.add_argument("outputpath", help="output csv file write full path")

    args = parser.parse_args()
    
    # Call the pipeline
    main(csv_file_path=args.csvpath, json_file_path=args.jsonpath, write_file_csv_path=args.outputpath)
    