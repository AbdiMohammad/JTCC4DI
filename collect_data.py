import sys
import json
import os
import pathlib

if __name__ == '__main__':

    json_args_file_path = sys.argv[1]
    with open(json_args_file_path) as infile:
        json_data = json.load(infile)
    
    experiment_id = 0
    for PSNR in [20, 16, 12, 8, 4]:
        json_data["PSNR"] = PSNR
        for codebook_prune_value in [60, 20, 40]:
            for codebook in json_data["codebooks"]:
                codebook["prune_value"] = codebook_prune_value
            for funnel_prune_value in [60, 20, 40]:
                json_data["funnel"]["prune_value"] = funnel_prune_value
                output_folder = f"output/experiment_{experiment_id}"
                try:
                    pathlib.Path(output_folder).mkdir(parents=True)
                except FileExistsError:
                    print(f'PSNR: {PSNR}, Codebook Prune Value: {codebook_prune_value}, Funnel Prune Value: {funnel_prune_value} skipped')
                    experiment_id += 1
                    continue
                json_data["output_folder"] = output_folder
                with open(f"configs/experiment_{experiment_id}.json", 'w') as outfile:
                    json.dump(json_data, outfile, indent=4)
                os.system(f"python JTCC4DI.py configs/experiment_{experiment_id}.json > {output_folder}/report.txt  2>&1")
                experiment_id += 1