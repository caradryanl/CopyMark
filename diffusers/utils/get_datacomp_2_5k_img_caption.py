# https://huggingface.co/datasets/mlfoundations/datacomp_pools/tree/main/datacomp_1b
'''
    https://huggingface.co/datasets/mlfoundations/datacomp_pools/resolve/main/datacomp_1b/0035af9f90f581816acf269df5eb37ad.parquet?download=true
    https://huggingface.co/datasets/mlfoundations/datacomp_pools/resolve/main/datacomp_1b/003da708d909c8cab24c7dcf4d04c371.parquet?download=true
    https://huggingface.co/datasets/mlfoundations/datacomp_pools/resolve/main/datacomp_1b/00818e301428c0573aac33fb4c1b5f02.parquet?download=true
    https://huggingface.co/datasets/mlfoundations/datacomp_pools/resolve/main/datacomp_1b/00aa8e74b038faf4d69ac89e84a318ba.parquet?download=true
    https://huggingface.co/datasets/mlfoundations/datacomp_pools/resolve/main/datacomp_1b/00f02df9e64ea4f4617886d397a13efb.parquet?download=true
'''


import argparse
import pandas as pd
import json
import os
import requests
from PIL import Image
from io import BytesIO

# https://huggingface.co/datasets/MuhammadHanif/Laion_aesthetics_5plus_1024_33M/

def download_image(url, path, dir):
    # Fetch the image content using requests
    try:
        response = requests.get(url, timeout = 2)
        if response.status_code == 200:
            # Create an image object with PIL from the raw content
            
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')  # Ensure it's in RGB mode for saving as JPEG
                img.save(dir+path, 'PNG')
                print('Image downloaded and saved as {}'.format(path))
                return True
        else:
            print('Failed to download the image path {}'.format(path))
            return False
    except:
        print('Failed to download the image path {}'.format(path))
        return False

def main(args):
    dataset = args.dataset
    target = args.target
    num_images = args.num_images
    dfs = []
    parquets = [
        '0035af9f90f581816acf269df5eb37ad.parquet',
        '003da708d909c8cab24c7dcf4d04c371.parquet',
        '00818e301428c0573aac33fb4c1b5f02.parquet',
        '00aa8e74b038faf4d69ac89e84a318ba.parquet',
        '00f02df9e64ea4f4617886d397a13efb.parquet',
    ]
    for parquet in parquets:
        # Load Parquet file into a pandas DataFrame
        df = pd.read_parquet(dataset + parquet)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    print(df.info())
    print(df.head())
    # print(type(df.iloc[0, 2]))

    os.makedirs(target + 'eval/', exist_ok=True)
    os.makedirs(target + 'test/', exist_ok=True)
    
    sampled_df = df.sample(n=num_images*3, random_state=42)

    caption, flag, failure = {}, False, []
    for idx in range(num_images * 2):
        # print(df.iloc[0, 0]['bytes'])
        # image_bytes = base64.b64decode(df.iloc[0, 0]['bytes'])
        dir = target + 'eval/' if flag == False else  target + 'test/'
        success = download_image(url=sampled_df.iloc[idx, 1], path=sampled_df.iloc[idx, 0] + '.png', dir=dir)
        if not success:
            failure.append(idx)
        else:
            caption[idx] = {
                'path': sampled_df.iloc[idx, 0] + '.png',
                'caption': [sampled_df.iloc[idx, 2]],
                'width': sampled_df.iloc[idx, 3],
                'height': sampled_df.iloc[idx, 4]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=2500)
    parser.add_argument("--dataset", type=str, default="datasets/datacomp/")
    parser.add_argument("--target", type=str, default="datasets/datacomp/")
    args = parser.parse_args()
    main(args)