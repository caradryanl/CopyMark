import argparse
import pandas as pd
import json
import os

import requests
from PIL import Image
from io import BytesIO
import base64



def main(args):
    dataset = args.dataset
    target = args.target
    num_images = args.num_images
    # Load Parquet file into a pandas DataFrame

    img_dir = target + 'images/'
    caption = {}

    for img in os.listdir(img_dir):
        caption[img[:-4]] = {
            "path": img,
            "height": 256,
            "width": 256,
            "caption": []
        }
            
    with open(target + 'caption.json', 'w') as file:
        json.dump(caption, file, indent=4)
    # with open(target + 'failure_download.json', 'w') as file:
    #     json.dump(failure, file, indent=4)

    # print("Available image number: {}".format(num_images - len(failure)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=2500)
    parser.add_argument("--dataset", type=str, default="datasets/celeba-hq-2-5k/train-00000-of-00006.parquet")
    parser.add_argument("--target", type=str, default="datasets/celeba-hq-2-5k/")
    args = parser.parse_args()
    main(args)