import argparse
import pandas as pd
import json, os

import requests
from PIL import Image
from io import BytesIO
import base64



def main(args):
    dataset = args.dataset
    target = args.target
    num_images = args.num_images
    # Load Parquet file into a pandas DataFrame
    df = pd.read_parquet(dataset)
    print(df.info())

    if not os.path.exists(target + 'images/'):
        os.mkdir(target + 'images/')

    # 32600
    caption = {}
    for idx in range(num_images):
        # print(df.iloc[0, 0])
        # image_bytes = base64.b64decode(df.iloc[0, 0]['bytes'])
        image_bytes = df.iloc[idx, 0]['bytes']
        image = Image.open(BytesIO(image_bytes))
        image = image.convert('RGB')  # Ensure it's in RGB mode for saving as JPEG
        image.save(target + 'images/' + f'{idx}.png', 'PNG')

        caption[idx] = {
            "path": target + 'images/' + f'{idx}.png',
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
    parser.add_argument("--dataset", type=str, default="datasets/celeba-hq-2-5k/train-00001-of-00006.parquet")
    parser.add_argument("--target", type=str, default="datasets/celeba-hq-2-5k-eval/")
    args = parser.parse_args()
    main(args)