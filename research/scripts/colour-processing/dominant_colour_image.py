"""
    Helper script to identify the dominant colour in an image and get its name:str using https://www.thecolorapi.com/ 
"""

import argparse
import requests
from colorthief import ColorThief
from typing import Optional
import pandas as pd
from tqdm import tqdm


# Get dominant HEX colour from an image
def get_dominant_colour(image_file_path:str) -> str:
    # Use `colorthief` to get dominant RGB colour
    color_thief_obj = ColorThief(
        file=image_file_path
    )
    dominant_rgb = color_thief_obj.get_color(
        quality=1
    )
    # Convert RGB to HEX
    # @ref https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string
    r, g, b = dominant_rgb
    hex_str = "{:02x}{:02x}{:02x}".format(r,g,b)
    return hex_str

# Get the colour name for an image
# @ref https://www.thecolorapi.com/id?hex=F93F6B
def get_colour_name(hex_code:str) -> Optional[str]:
    try:
        # Construct query
        hex_code.replace("#","")
        query_string = f"https://www.thecolorapi.com/id?hex={hex_code}"

        # Make a request to the API
        r = requests.get(
            url = query_string
        )

        # Extract the colour name
        res = r.json()
        colour_name = str(res["name"]["value"])
        return colour_name

    except Exception as e:
        print(f"Error {e}")
        return None

# Main pipeline - read images from csv files and write colour names into another csv file
def main(input_csv_file_path:str, output_csv_file_path:str, col_name:str="image_file_path") -> None:
    # Read input dataframe
    df = pd.read_csv(input_csv_file_path)

    # Extract the file names
    file_names = df[col_name]
    
    # Extract the colour names
    all_colour_names = []
    for i in tqdm(range(len(file_names))):
        _file = file_names[i]
        hex_dom = get_dominant_colour(_file)
        col_name = get_colour_name(hex_dom)
        if col_name is None:
            col_name = "undefined"
        all_colour_names.append(col_name)

    # Save combined dataframe to output
    df["dominant_colour_name"] = all_colour_names
    
    if ".csv" not in output_csv_file_path:
        output_csv_file_path.replace(".","")
        output_csv_file_path += ".csv"
    df.to_csv(
        output_csv_file_path,
        index = False
    )

if __name__ == "__main__":
    # Set up argument parsing routine
    parser = argparse.ArgumentParser()
    parser.add_argument("inputcsvpath", help="Path to the input .csv file")
    parser.add_argument("outputcsvpath", help = 'Path to the output .csv file')
    parser.add_argument("filecolname", help="Column name of the input csv file containing the filenames")

    args = parser.parse_args()

    # Call the pipeline
    main(input_csv_file_path=args.inputcsvpath, output_csv_file_path=args.outputcsvpath, col_name=args.filecolname)
