import os
import argparse
import pandas as pd
import json

def main(args):
    dataset = args.dataset
    target = args.target
    num_images = args.num_images
    # Load Parquet file into a pandas DataFrame
    df = pd.read_parquet(dataset)
    print(df.info())

    # 32600
    caption = {}
    for idx in range(num_images):
        if '{}.jpg'.format(idx) in os.listdir(target + 'images/'):
            caption[idx] = {
                'path':f'{idx}.jpg',
                'height': df.iloc[idx, 3],
                'width': df.iloc[idx, 2],
                'caption': [df.iloc[idx, 1]],
            }
            print(f'add caption for {idx}.jpg')
    with open(target + 'caption.json', 'w') as file:
        json.dump(caption, file, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--dataset", type=str, default="datasets/laion-aesthetic-info/laion_aesthetics_1024_33M_1.parquet")
    parser.add_argument("--target", type=str, default="datasets/laion-aesthetic-50k/")
    args = parser.parse_args()
    main(args)