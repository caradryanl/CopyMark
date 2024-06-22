import argparse
import pandas as pd
import json
import os
import requests
from PIL import Image
from io import BytesIO

# https://huggingface.co/datasets/MuhammadHanif/Laion_aesthetics_5plus_1024_33M/
'''
    https://huggingface.co/datasets/MuhammadHanif/Laion_aesthetics_5plus_1024_33M/resolve/main/laion_aesthetics_1024_33M_1.parquet?download=true
    https://huggingface.co/datasets/MuhammadHanif/Laion_aesthetics_5plus_1024_33M/resolve/main/laion_aesthetics_1024_33M_2.parquet?download=true
    https://huggingface.co/datasets/MuhammadHanif/Laion_aesthetics_5plus_1024_33M/resolve/main/laion_aesthetics_1024_33M_3.parquet?download=true
    https://huggingface.co/datasets/MuhammadHanif/Laion_aesthetics_5plus_1024_33M/resolve/main/laion_aesthetics_1024_33M_4.parquet?download=true
    https://huggingface.co/datasets/MuhammadHanif/Laion_aesthetics_5plus_1024_33M/resolve/main/laion_aesthetics_1024_33M_5.parquet?download=true
    https://huggingface.co/datasets/MuhammadHanif/Laion_aesthetics_5plus_1024_33M/resolve/main/laion_aesthetics_1024_33M_6.parquet?download=true
    https://huggingface.co/datasets/MuhammadHanif/Laion_aesthetics_5plus_1024_33M/resolve/main/laion_aesthetics_1024_33M_7.parquet?download=true
    https://huggingface.co/datasets/MuhammadHanif/Laion_aesthetics_5plus_1024_33M/resolve/main/laion_aesthetics_1024_33M_8.parquet?download=true
    https://huggingface.co/datasets/MuhammadHanif/Laion_aesthetics_5plus_1024_33M/resolve/main/laion_aesthetics_1024_33M_9.parquet?download=true
    https://huggingface.co/datasets/MuhammadHanif/Laion_aesthetics_5plus_1024_33M/resolve/main/laion_aesthetics_1024_33M_10.parquet?download=true
'''

def download_image(url, idx, dir):
    # Fetch the image content using requests
    try:
        response = requests.get(url, timeout = 2)
        if response.status_code == 200:
            # Create an image object with PIL from the raw content
            
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')  # Ensure it's in RGB mode for saving as JPEG
                img.save(dir+'{}.png'.format(idx), 'PNG')
                print('Image downloaded and saved as {}.png'.format(idx))
                return True
        else:
            print('Failed to download the image index {}'.format(idx))
            return False
    except:
        print('Failed to download the image index {}'.format(idx))
        return False

def main(args):
    dataset = args.dataset
    target = args.target
    num_images = args.num_images
    dfs = []
    indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for idx in indices:
        parquet = f'laion_aesthetics_1024_33M_{idx}.parquet'
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

        dir = target + 'eval/' if flag == False else  target + 'test/'
        success, size = download_image(url=sampled_df.iloc[idx, 0], idx=idx, dir=dir)
        if not success:
            failure.append(idx)
        else:
            caption[idx] = {
                'path': f'{idx}.png',
                'caption': [sampled_df.iloc[idx, 1]],
                'width': int(sampled_df.iloc[idx, 2]),
                'height': int(sampled_df.iloc[idx, 3])
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
    parser.add_argument("--num-images", type=int, default=10000)
    parser.add_argument("--dataset", type=str, default="datasets/laion-aesthetic/")
    parser.add_argument("--target", type=str, default="datasets/laion-aesthetic/")
    args = parser.parse_args()
    main(args)