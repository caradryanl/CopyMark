import argparse
import pandas as pd
import json

import requests
from PIL import Image
from io import BytesIO

def download_image(url, idx, dir):
    # Fetch the image content using requests
    try:
        response = requests.get(url, timeout = 2)
        if response.status_code == 200:
            # Create an image object with PIL from the raw content
            
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')  # Ensure it's in RGB mode for saving as JPEG
                img.save(dir+'{}.jpg'.format(idx), 'JPEG')
                print('Image downloaded and saved as {}.jpg'.format(idx))
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
    # Load Parquet file into a pandas DataFrame
    df = pd.read_parquet(dataset)
    print(df.info())

    # 32600
    failure, caption = [], {}
    for idx in range(32600, num_images):
        # print(df.iloc[0, 0], type(df.iloc[0, 0]), df.iloc[0, 1], type(df.iloc[0, 1]))
        success = download_image(url=df.iloc[idx, 0], idx=idx, dir=target + 'images/')
        if not success:
            failure.append(idx)
        else:
            caption[idx] = {
                'text': df.iloc[idx, 1],
                'width': df.iloc[idx, 2],
                'height': df.iloc[idx, 3]
            }
    with open(target + 'caption.json', 'w') as file:
        json.dump(caption, file, indent=4)
    with open(target + 'failure_download.json', 'w') as file:
        json.dump(failure, file, indent=4)

    print("Available image number: {}".format(num_images - len(failure)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--dataset", type=str, default="datasets/laion-aesthetic-info/laion_aesthetics_1024_33M_1.parquet")
    parser.add_argument("--target", type=str, default="datasets/laion-aesthetic-50k/")
    args = parser.parse_args()
    main(args)