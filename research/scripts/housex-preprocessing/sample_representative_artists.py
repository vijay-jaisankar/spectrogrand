"""
    Get a unique sample of `k` artists per subgenre of house music.
"""

import argparse
import os
from typing import Optional, List
import random

# Set random seed
random.seed(42)

# Maintain a global artist dict
GLOBAL_ARTIST_DICT = {
    "bass" : [],
    "future" : [],
    "melodic" : [],
    "progressive" : []
}
SUB_GENRE_NAMES = ["bass", "progressive", "future", "melodic"]

# Get artist names for subgenre; Accepted values for "sub_genre_name" : ["bass", "house", "future", "melodic"]
def get_artist_names(parent_dir_path:str, sub_genre_name:str) -> List[str]:
    global SUB_GENRE_NAMES
    # Sanity check
    if sub_genre_name not in SUB_GENRE_NAMES:
        print(f"{sub_genre_name} is an invalid name as it is not in {SUB_GENRE_NAMES}")
        return []
    try:
        with open(f"{parent_dir_path}/{sub_genre_name}_house/{sub_genre_name}_all_artists.txt", "r") as f:
            all_artists = f.readlines()
        # Remove duplicates and convert names to lowercase
        all_artists = list(set(all_artists))
        all_artists = [x.lower() for x in all_artists]
        return all_artists
    except Exception as e:
        print(f"Error {e} while retrieving artist names")
        return []

# Get random sample of size k from list
def get_random_sample_list(candidate_elements:List, k:int) -> Optional[List]:
    # Sanity check
    if k > len(candidate_elements):
        return None
    # Generate random sample
    sample = random.sample(candidate_elements, k)
    return list(sample)

# Main artist selection pipeline
def main(input_parent_dir_path:str, output_file_path:str, k:int = 4) -> None:
    global GLOBAL_ARTIST_DICT
    # Read all the files and store artists' names
    bass_artist_names = get_artist_names(parent_dir_path=input_parent_dir_path, sub_genre_name="bass")
    future_artist_names = get_artist_names(parent_dir_path=input_parent_dir_path, sub_genre_name="future")
    melodic_artist_names = get_artist_names(parent_dir_path=input_parent_dir_path, sub_genre_name="melodic") 
    progressive_artist_names = get_artist_names(parent_dir_path=input_parent_dir_path, sub_genre_name="progressive")

    GLOBAL_ARTIST_DICT["bass"] = bass_artist_names
    GLOBAL_ARTIST_DICT["future"] = future_artist_names
    GLOBAL_ARTIST_DICT["melodic"] = melodic_artist_names
    GLOBAL_ARTIST_DICT["progressive"] = progressive_artist_names

    for _k in GLOBAL_ARTIST_DICT:
        assert len(GLOBAL_ARTIST_DICT[_k]) > 0

    # Progressively remove duplicate artists - keep candidate as bass
    GLOBAL_ARTIST_DICT["future"] = [x for x in GLOBAL_ARTIST_DICT["future"] if x not in GLOBAL_ARTIST_DICT["bass"]]
    GLOBAL_ARTIST_DICT["future"] = [x for x in GLOBAL_ARTIST_DICT["future"] if x not in GLOBAL_ARTIST_DICT["melodic"]]
    GLOBAL_ARTIST_DICT["future"] = [x for x in GLOBAL_ARTIST_DICT["future"] if x not in GLOBAL_ARTIST_DICT["progressive"]]

    GLOBAL_ARTIST_DICT["melodic"] = [x for x in GLOBAL_ARTIST_DICT["melodic"] if x not in GLOBAL_ARTIST_DICT["bass"]]
    GLOBAL_ARTIST_DICT["melodic"] = [x for x in GLOBAL_ARTIST_DICT["melodic"] if x not in GLOBAL_ARTIST_DICT["progressive"]]
    GLOBAL_ARTIST_DICT["melodic"] = [x for x in GLOBAL_ARTIST_DICT["melodic"] if x not in GLOBAL_ARTIST_DICT["future"]]

    GLOBAL_ARTIST_DICT["progressive"] = [x for x in GLOBAL_ARTIST_DICT["progressive"] if x not in GLOBAL_ARTIST_DICT["bass"]]
    GLOBAL_ARTIST_DICT["progressive"] = [x for x in GLOBAL_ARTIST_DICT["progressive"] if x not in GLOBAL_ARTIST_DICT["future"]]
    GLOBAL_ARTIST_DICT["progressive"] = [x for x in GLOBAL_ARTIST_DICT["progressive"] if x not in GLOBAL_ARTIST_DICT["melodic"]]

    # Choose k samples per sub-genre
    selected_bass = get_random_sample_list(candidate_elements=GLOBAL_ARTIST_DICT["bass"],k=int(k))
    selected_future = get_random_sample_list(candidate_elements=GLOBAL_ARTIST_DICT["future"],k=int(k))
    selected_melodic = get_random_sample_list(candidate_elements=GLOBAL_ARTIST_DICT["melodic"],k=int(k))
    selected_progressive = get_random_sample_list(candidate_elements=GLOBAL_ARTIST_DICT["progressive"],k=int(k))

    SELECTED_DICT = {
        "bass" : selected_bass,
        "future" : selected_future,
        "melodic" : selected_melodic,
        "progressive" : selected_progressive
    }

    # Write the selections into a file
    if ".txt" not in output_file_path:
        output_file_path = output_file_path + ".txt"
    with open(output_file_path, "w") as f:
        for _k in SELECTED_DICT:
            f.write(f'{_k}\n')
            f.write('-'*40)
            f.write('\n')
            for artist_name in SELECTED_DICT[_k]:
                f.write(f'{artist_name}')
            f.write('\n')

if __name__ == "__main__":
    # Set up argument parsing routine
    parser = argparse.ArgumentParser()
    parser.add_argument("inputpath", help="parent input path")
    parser.add_argument("outputfilepath", help="output txt file path")
    args = parser.parse_args()
    
    # Call the pipeline
    main(input_parent_dir_path=args.inputpath, output_file_path=args.outputfilepath)
    