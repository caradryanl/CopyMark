import argparse
import pandas as pd
import json, os
import numpy as np

import requests
from PIL import Image
from io import BytesIO
import base64

def download_image(url, idx, dir):
    # Fetch the image content using requests
    try:
        response = requests.get(url, timeout = 5)
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
    target_eval, target_test = args.target_eval, args.target_test
    num_images = args.num_images
    # Load Parquet file into a pandas DataFrame
    df = pd.read_parquet(dataset)
    length = len(df)

    sequence = np.arange(0, length)
    np.random.shuffle(sequence)

    print(df.head())

    if not os.path.exists(target_eval + 'images/'):
        os.mkdir(target_eval + 'images/')
    if not os.path.exists(target_test + 'images/'):
        os.mkdir(target_test + 'images/')

    # # 32600
    caption, failure = {}, []
    test_flag = False
    for idx in sequence:
        idx = int(idx)
        dir = target_eval + 'images/' if test_flag == False else target_test + 'images/'
        success = download_image(url=df.iloc[idx, 0], idx=int(df.iloc[idx, 2]), dir=dir)
        if not success:
            failure.append(idx)
        else:
            caption[idx] = {
                'path': f'{int(df.iloc[idx, 2])}.png',
                'caption': [df.iloc[idx, 1]],
                'width': 1024,
                'height': 1024
            }
            if len(caption) >= num_images and test_flag == False:
                test_flag = True
                with open(target_eval + 'caption_eval.json', 'w') as file:
                    json.dump(caption, file, indent=4)
                with open(target_eval + 'failure_eval.json', 'w') as file:
                    json.dump(failure, file, indent=4)
                caption, failure = {}, []
            elif len(caption) >= num_images:
                with open(target_test + 'caption_test.json', 'w') as file:
                    json.dump(caption, file, indent=4)
                with open(target_test + 'failure_test.json', 'w') as file:
                    json.dump(failure, file, indent=4)
                break
    

    # print("Available image number: {}".format(num_images - len(failure)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=2500)
    parser.add_argument("--dataset", type=str, default="datasets/laion-mi-info/laion_mi_members_metadata.parquet")
    parser.add_argument("--target-eval", type=str, default="datasets/laion-mi-member-eval/")
    parser.add_argument("--target-test", type=str, default="datasets/laion-mi-member-test/")
    args = parser.parse_args()
    main(args)