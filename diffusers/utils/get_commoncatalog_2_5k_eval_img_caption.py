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
    # Load Parquet file into a pandas DataFrame
    df = pd.read_parquet(dataset)
    print(df.info())

    # # 32600
    caption = {}
    for idx in range(num_images):
        idx = idx + num_images
        # print(df.iloc[0, 0]['bytes'])
        # image_bytes = base64.b64decode(df.iloc[0, 0]['bytes'])
        image_bytes = df.iloc[idx]['jpg']
        image = Image.open(BytesIO(image_bytes))
        image = image.convert('RGB')  # Ensure it's in RGB mode for saving as JPEG
        image.save(target + 'images/' +f'{df.iloc[idx,0]}.jpg', 'JPEG')

        caption[idx] = {
            "path": f'{df.iloc[idx,0]}.jpg',
            "height": int(df.iloc[idx]['height']),
            "width": int(df.iloc[idx]['width']),
            "caption": [df.iloc[idx]['blip2_caption']]
        }
    with open(target + 'caption.json', 'w') as file:
        json.dump(caption, file, indent=4)
    # with open(target + 'failure_download.json', 'w') as file:
    #     json.dump(failure, file, indent=4)

    # print("Available image number: {}".format(num_images - len(failure)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=2500)
    parser.add_argument("--dataset", type=str, default="datasets/commoncatalog-2-5k/part-00000-tid-6788610333251177415-200b26fd-7ec7-4ac9-857a-cdfc7823dce5-56235-1-c000.parquet")
    parser.add_argument("--target", type=str, default="datasets/commoncatalog-2-5k-eval/")
    args = parser.parse_args()
    main(args)