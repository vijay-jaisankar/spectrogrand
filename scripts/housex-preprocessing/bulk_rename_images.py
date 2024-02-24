"""
    Bulk rename images into {genre}_{idx}.jpg
"""

import argparse
import os

# Bulk rename directory
# @note For HouseX spectrogram processing, `full_dir_path` should be /path/to/housex/{train|test|val}
def bulk_rename_dir(full_dir_path:str, verbose:bool = False):
    # Get files in the dir
    # @ref https://stackoverflow.com/a/37467669
    files = os.listdir(full_dir_path)

    # Print number of files in the pipeline
    if verbose:
        print(f"Processing {len(files)} files")

    # Store mapping of genre:last_seen_index
    genre_idx_mapping = {}

    # Enumerate and rename in place
    for idx, f in enumerate(files):

        # Get selected index
        song_type = f.split("-")[0]
        if song_type not in genre_idx_mapping:
            genre_idx_mapping[song_type] = 0
        else:
            genre_idx_mapping[song_type] += 1

        chosen_index = genre_idx_mapping[song_type]
        try:
            os.rename(
                os.path.join(full_dir_path, f),
                os.path.join(full_dir_path, f"{song_type}_{str(chosen_index)}.jpg")
            )
        except Exception as e:
            print(f"Error obtained while renaming {f}")
            continue

# Main Loop
def main(input_dir_path:str, verbose:bool) -> None:
    bulk_rename_dir(full_dir_path=input_dir_path, verbose=verbose)

if __name__ == "__main__":
    # Set up argument parsing routing
    parser = argparse.ArgumentParser()
    parser.add_argument("inputpath", help="input path containing jpg images to be renamed")
    parser.add_argument("verbose", help="toggle verbosity (default: True)", default=True)
    args = parser.parse_args()

    # Call the pipeline
    main(input_dir_path=args.inputpath, verbose=args.verbose)