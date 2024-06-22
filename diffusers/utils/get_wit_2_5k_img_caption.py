import pandas as pd
import os
import argparse
import gzip
import json

import requests
from PIL import Image
from io import BytesIO

# https://github.com/google-research-datasets/wit/blob/main/DATA.md
'''
    https://storage.googleapis.com/gresearch/wit/wit_v1.val.all-00000-of-00005.tsv.gz
    https://storage.googleapis.com/gresearch/wit/wit_v1.val.all-00001-of-00005.tsv.gz
    https://storage.googleapis.com/gresearch/wit/wit_v1.val.all-00002-of-00005.tsv.gz
    https://storage.googleapis.com/gresearch/wit/wit_v1.val.all-00003-of-00005.tsv.gz
    https://storage.googleapis.com/gresearch/wit/wit_v1.val.all-00004-of-00005.tsv.gz
'''

def download_image(url, idx, dir):
    # Fetch the image content using requests
    try:
        response = requests.get(url, timeout = 25)
        if response.status_code == 200:
            # Create an image object with PIL from the raw content
            
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')  # Ensure it's in RGB mode for saving as JPEG
                width, height = img.size
                img.save(dir+'{}.png'.format(idx), 'PNG')
                print('Image downloaded and saved as {}.png'.format(idx))
                return True, (width, height)
        else:
            print('Failed to download the image index {}'.format(idx))
            return False, None
    except:
        print('Failed to download the image index {}'.format(idx))
        return False, None

def main(args):
    dataset = args.dataset
    target = args.target
    num_images = args.num_images

    dfs = []
    indices = [0, 1, 2, 3, 4]
    # indices = [0,]
    for idx in indices:
        # Load Parquet file into a pandas DataFrame
        with gzip.open(dataset + f'wit_v1.val.all-0000{idx}-of-00005.tsv.gz', 'rt') as f:
            df = pd.read_csv(f, sep='\t')
            dfs.append(df)
    df = pd.concat(dfs, axis=0)
    sampled_df = df.sample(n=num_images*10, random_state=42)
    print(sampled_df.info())
    print(sampled_df.head())

    os.makedirs(target + 'eval/', exist_ok=True)
    os.makedirs(target + 'test/', exist_ok=True)

    failure, caption, flag = [], {}, False
    for idx in range(num_images*10):
        # print(df.iloc[0, 0], type(df.iloc[0, 0]), df.iloc[0, 1], type(df.iloc[0, 1]))
        dir = target + 'eval/' if flag == False else  target + 'test/'
        success, size = download_image(url=sampled_df.iloc[idx, 2], idx=idx, dir=dir)
        if not success:
            failure.append(idx)
        else:
            caption[idx] = {
                'path': '{}.png'.format(idx),
                'caption': [''],
                'width': int(sampled_df.iloc[idx, 11]),
                'height': int(sampled_df.iloc[idx, 10])
            }
            if len(caption) >= num_images and flag == False:
                flag = True
                with open(target + 'caption_eval.json', 'w') as file:
                    json.dump(caption, file, indent=4)
                caption = {}
            elif len(caption) >= num_images:
                with open(target + 'caption_test.json', 'w') as file:
                    json.dump(caption, file, indent=4)
                break

    with open(target + 'failure_download.json', 'w') as file:
        json.dump(failure, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=2500)
    parser.add_argument("--dataset", type=str, default="datasets/wit/")
    parser.add_argument("--target", type=str, default="datasets/wit/")
    args = parser.parse_args()
    main(args)