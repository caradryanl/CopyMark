import os, shutil
import argparse
import pandas as pd
import json

def main(args):
    dataset = args.dataset
    target = args.target

    if not os.path.exists(target + 'images/'):
        os.mkdir(target + 'images/')
    caption = {}
    training_list = os.listdir(dataset+ 'images/')
    for idx, path in enumerate(training_list):
        caption[idx] = {
            "path": path,
            "height": None,
            "width": None,
            "caption": [""]
        }

    with open(target + 'caption.json', 'w') as file:
        json.dump(caption, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="assets/examples/")
    parser.add_argument("--target", type=str, default="assets/examples/")
    args = parser.parse_args()
    main(args)