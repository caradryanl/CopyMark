import os
import argparse
import pandas as pd
import json

import requests
from PIL import Image
from io import BytesIO
import base64



def main(args):
    dataset = args.dataset
    target = args.target
    num_images = args.num_images

    dfs = []
    parquets = ['0000.parquet', '0001.parquet', '0002.parquet', '0003.parquet', '0004.parquet']
    for parquet in parquets:
        # Load Parquet file into a pandas DataFrame
        df = pd.read_parquet(dataset + parquet)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    print(df.info())
    print(df.head())
    # print(type(df.iloc[0, 2]))

    os.makedirs(target + 'train/', exist_ok=True)
    os.makedirs(target + 'val/', exist_ok=True)
    
    sampled_df = df.sample(n=num_images*3, random_state=42)

    # # 32600
    caption, flag = {}, False
    for idx in range(num_images * 2):
        # print(df.iloc[0, 0]['bytes'])
        # image_bytes = base64.b64decode(df.iloc[0, 0]['bytes'])
        image_bytes = sampled_df.iloc[idx, 2]
        image = Image.open(BytesIO(image_bytes))
        image = image.convert('RGB')  # Ensure it's in RGB mode for saving as JPEG

        dir = target + 'train/' + sampled_df.iloc[idx, 3] + '.png' if flag == False else target + 'val/' + sampled_df.iloc[idx, 3] + '.png'
        image.save(dir, 'PNG')

        caption[idx] = {
            "path": sampled_df.iloc[idx, 3],
            "height": image.size[1],
            "width": image.size[0],
            "caption": ['']
        }
        if len(caption) >= num_images and flag == False:
            flag = True
            with open(target + 'caption_train.json', 'w') as file:
                json.dump(caption, file, indent=4)
            caption = {}
        elif len(caption) >= num_images:
            with open(target + 'caption_val.json', 'w') as file:
                json.dump(caption, file, indent=4)
            break

    # print("Available image number: {}".format(num_images - len(failure)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=2500)
    parser.add_argument("--dataset", type=str, default="datasets/yfcc100m/")
    parser.add_argument("--target", type=str, default="datasets/yfcc100m/")
    args = parser.parse_args()
    main(args)