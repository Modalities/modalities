import json
import os
import sys

from benchmarks.dataformats.utils import get_common_opts, profile
import webdataset as wds
from src.modalities.dataloader.dataset import PackedMemMapDataset
from pathlib import Path

@profile(runs=1)
def prepare_webdataset(json_path, output_dir):

    shard_writer = wds.ShardWriter(f"{output_dir}/%06d.tar", maxcount=10000)

    with open(json_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            name = data['img_path']
            basename, _ = os.path.splitext(os.path.basename(name))
            
            with open(f"{json_path.parent}/{name}", "rb") as stream:
                image = stream.read()
            
            # get the captions
            data_copy = data.copy()
            data_copy.pop('img_path')
            
            text = json.dumps(data_copy).encode('utf-8')    
            sample = {
                "__key__": basename,
                "jpg": image,
                "json": text,
            }
            shard_writer.write(sample)
        shard_writer.close()

    

if __name__ == "__main__":

    opts = get_common_opts(sys.argv[1:])

    if not opts.json_file:
        raise Exception("Json file not provided")

    json_file = Path(opts.json_file)
    
    # check if the file exists
    if not json_file.exists():
        raise Exception("Json file does not exist")
    
    # get the split name from the json file
    split_name = json_file.stem

    if "train" in split_name:
        split = "train"
    elif "val" in split_name:
        split = "val"
    elif "test" in split_name:
        split = "test"
    else:
        raise Exception("Split name not recognized")

    ########################################################
    # Webdataset preparation
    ########################################################

    # create the output directory
    web_out_dir = Path(opts.webout)/split

    # check if the dir  exists
    if not web_out_dir.exists():
        os.makedirs(web_out_dir)
    else:
        # prompt the user to delete the directory
        print(f"Directory {web_out_dir} already exists. Do you want to delete it? [y/n]")
        response = input()
        if response == "y":
            os.system(f"rm -rf {web_out_dir}")
            os.makedirs(web_out_dir)
        else:
            raise Exception("Directory already exists")

    web_data = prepare_webdataset(json_path=json_file, output_dir=web_out_dir)
        
     ########################################################
    # Memmap preparation
    #########################################################