"""
    Parsing the markdown files generated using LLM outputs
"""

import pickle
import argparse
from typing import Optional, List

# Parse MD files
# @note is_external_model is true only for GPT 3,5 outputs
def parse_md(input_file_path:str, is_external_model:bool = False) -> Optional[dict]:
    try:
        # Read the file
        with open(input_file_path, "r") as f:
            all_lines = f.readlines()
        
        # Preprocess the formatting of the lines
        all_lines = [x for x in all_lines if x != '\n']

        if is_external_model is True:
            all_lines = all_lines[1:]

        all_lines = [x.replace('\n', '') for x in all_lines]
        all_lines = [x.replace("- ", "") for x in all_lines]
        all_lines = [x.replace("-", "") for x in all_lines]
        all_lines = [x.replace(": ", "") for x in all_lines]
        all_lines = [x.replace(":", "") for x in all_lines]

        # Find out the indices where *_house is present
        parent_house_indices = []
        N = len(all_lines)
        for i in range(N):
            if "_house" in all_lines[i]:
                parent_house_indices.append(i)

        # Store sub-genre outputs
        sub_genre_nested_outputs = []
        M = len(parent_house_indices)
        assert M == 4 

        # Append `N` to process the parent_house_indices for all 4 genres
        parent_house_indices_extended = parent_house_indices.copy()
        parent_house_indices_extended.append(N)
        for i in range(0, len(parent_house_indices)):
            src_index = parent_house_indices_extended[i] + 1
            target_index = parent_house_indices_extended[i+1]
            current_outputs = ""
            for j in range(src_index, target_index):
                current_outputs += all_lines[j]
                current_outputs += ","
            sub_genre_nested_outputs.append(current_outputs)

        assert len(sub_genre_nested_outputs) == M
        # Generate the dictionary of type:descriptors
        descriptor_dict = {}
        for i in range(M):
            # Remove the trailing comma
            descriptor_dict[all_lines[parent_house_indices[i]]] = sub_genre_nested_outputs[i][:-1].lower()

        return descriptor_dict

    except Exception as e:
        print(f"Error: {e.with_traceback()}")
        return None

# Main Loop
def main(input_md_file:str, is_gpt_file:bool, output_path_pkl:str) -> None:
    # Call the parser to get the dictionary
    parsed_dict = parse_md(input_md_file, is_gpt_file)

    # Check if the output is None and if not, save to pkl file
    if parsed_dict is None:
        print(f"Empty parsing output, aborting...")
        return

    if ".pkl" not in output_path_pkl:
        output_path_pkl.replace(".", "")
        output_path_pkl += ".pkl"
    
    with open(output_path_pkl, "wb") as f:
        pickle.dump(parsed_dict, f)
  
if __name__ == "__main__":
    # Set up argument parsing routine
    parser = argparse.ArgumentParser()
    parser.add_argument("inputpathmd", help="Input path to the markdown file")
    parser.add_argument("outputpathpkl", help="Output path to the pickle fle where parsed outputs are to the stored")
    parser.add_argument("isexternal", help="True if the MD file contains GPT 3.5 outputs (default: False)", default=False)
    args = parser.parse_args()

    # Call the pipeline
    main(input_md_file=args.inputpathmd, is_gpt_file=args.isexternal, output_path_pkl=args.outputpathpkl)
