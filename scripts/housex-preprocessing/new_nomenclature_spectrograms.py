"""
    Save the spectrograms of the HouseX dataset in the following format
    {train_test_val}_{bass|future|melodic|progressive}_{identifier}.jpg
"""

import argparse
import os
import cv2
from glob import glob
from typing import List
from tqdm import tqdm

# Maintain a global count
GLOBAL_COUNT_DICT = {
    "train": {
        "bass" : 0, "future" : 0, "melodic" : 0, "progressive" : 0
    },
    "test" : {
        "bass" : 0, "future" : 0, "melodic" : 0, "progressive" : 0
    },
    "val": {
        "bass" : 0, "future" : 0, "melodic" : 0, "progressive" : 0
    }
}

# Get image descriptors for full file path
def get_image_descriptors(full_file_path:str) -> dict:
    # Get the parent folder name
    split_path = full_file_path.split("/")
    parent_folder_name = split_path[-2]

    # Get the genre
    raw_song_name = split_path[-1].replace(".jpg", "")
    genre = raw_song_name.split("-")[0].replace(" house", "")

    return {
        "parent" : parent_folder_name,
        "genre" : genre
    }

# Recursively get all full paths containing .jpg files
def get_all_candidate_images(input_dir_path:str) -> List[str]:
    # Recursively scrape all jpg files
    files = glob(f"{input_dir_path}/**/*.jpg", recursive = True)
    return list(files)

# Load image from given path
def load_image(image_path:str):
    try:
        img = cv2.imread(image_path)
        return img
    except Exception as e:
        print(f"Error {e} obtained while reading file {image_path}")
        return None

# Write image into given path
def save_image(new_image_path:str, img):
    try:
        cv2.imwrite(new_image_path, img)
    except Exception as e:
        print(f"Error {e} obtained while writing file {new_image_path}")
        pass

# Main saving and loading pipeline
def main(input_dir_path:str, parent_output_dir_path:str) -> None:
    global GLOBAL_COUNT_DICT

    # Get all jpg files
    all_jpg_files = get_all_candidate_images(input_dir_path=input_dir_path)
    N = len(all_jpg_files)
    assert N>0

    for i in tqdm(range(N)):
        # Get image descriptors
        descriptors = get_image_descriptors(all_jpg_files[i])
        parent = descriptors["parent"]
        genre = descriptors["genre"]

        # Update the count in the main dict
        GLOBAL_COUNT_DICT[parent][genre] += 1
        file_index = GLOBAL_COUNT_DICT[parent][genre]

        # Construct new file name
        save_path = f"{parent_output_dir_path}/{parent}/{parent}_{genre}_{file_index}.jpg"

        # Load and save image
        img = load_image(all_jpg_files[i])
        if img is None:
            continue
        save_image(save_path, img)

if __name__ == "__main__":
    # Set up argument parsing routine
    parser = argparse.ArgumentParser()
    parser.add_argument("inputpath", help="parent input path")
    parser.add_argument("outputpath", help="parent output path")
    args = parser.parse_args()
    
    # Call the pipeline
    main(input_dir_path=args.inputpath, parent_output_dir_path=args.outputpath)
    
