import os, shutil
import argparse
import pandas as pd
import json

def main(args):
    dataset = args.dataset
    target = args.target
    num_images = args.num_images

    with open(dataset + 'caption.json', 'r') as json_file:
        caption = json.load(json_file)

    data = {}
    cnt = 0
    for id, metadata in caption.items():
        data[id] = metadata
        img_path = metadata["path"]

        source = dataset + 'images/' + img_path
        dest = target + 'images/' + img_path
        shutil.copy(source, dest)

        cnt += 1
        if cnt >= num_images:
            break

    with open(target + 'caption.json', 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=2500)
    parser.add_argument("--dataset", type=str, default="datasets/laion-aesthetic-50k/")
    parser.add_argument("--target", type=str, default="datasets/laion-aesthetic-2-5k/")
    args = parser.parse_args()
    main(args)