# import pandas as pd

# # Load the TSV file
# file_path = 'datasets/cc12m/cc12m.tsv'  # Replace with your file path
# df = pd.read_csv(file_path, sep='\t')

# # Display the first few rows of the dataframe
# print(df.iloc[0].iloc[0], df.iloc[0])
# print(df.iloc[1].iloc[0], df.iloc[1])
# print(df.iloc[2].iloc[0], df.iloc[2])

import os
import argparse
import pandas as pd
import json

import requests
from PIL import Image
from io import BytesIO

def download_image(url, idx, dir):
    # Fetch the image content using requests
    try:
        response = requests.get(url, timeout = 5)
        if response.status_code == 200:
            # Create an image object with PIL from the raw content
            
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')  # Ensure it's in RGB mode for saving as JPEG
                width, height = img.size
                img.save(dir+'{}.jpg'.format(idx), 'JPEG')
                print('Image downloaded and saved as {}.jpg'.format(idx))
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
    df = pd.read_csv(dataset, sep='\t')
    print(df.info())

    sampled_df = df.sample(n=num_images*50, random_state=42)

    os.makedirs(target + 'train/', exist_ok=True)
    os.makedirs(target + 'val/', exist_ok=True)

    failure, caption, flag = [], {}, False
    for idx in range(num_images*50):
        # print(df.iloc[0, 0], type(df.iloc[0, 0]), df.iloc[0, 1], type(df.iloc[0, 1]))
        dir = target + 'train/' if flag == False else  target + 'val/'
        success, size = download_image(url=sampled_df.iloc[idx, 0], idx=idx, dir=dir)
        if not success:
            failure.append(idx)
        else:
            caption[idx] = {
                'text': sampled_df.iloc[idx, 1],
                'width': size[0],
                'height': size[1]
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

    with open(target + 'failure_download.json', 'w') as file:
        json.dump(failure, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=2500)
    parser.add_argument("--dataset", type=str, default="datasets/cc12m/cc12m.tsv")
    parser.add_argument("--target", type=str, default="datasets/cc12m/")
    args = parser.parse_args()
    main(args)