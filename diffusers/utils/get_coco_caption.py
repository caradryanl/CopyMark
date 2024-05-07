import os, shutil
import argparse
import pandas as pd
import json

def main(args):
    source = args.source
    target = args.target

    with open(source, 'r') as json_file:
        caption = json.load(json_file)

    data = {}
    for _, metadata in enumerate(caption['images']):
        img_path = metadata["file_name"]
        height, width, id = metadata['height'], metadata['width'], metadata['id']
        data[id] = {
            "path": img_path,
            "height": height,
            "width": width,
            "caption": []
        }

    for annotation in caption["annotations"]:
        id = annotation["image_id"]
        data[id]["caption"].append(annotation["caption"])

    with open(target, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="datasets/coco2017-val/captions_val2017.json")
    parser.add_argument("--target", type=str, default="datasets/coco2017-val/caption.json")
    args = parser.parse_args()
    main(args)