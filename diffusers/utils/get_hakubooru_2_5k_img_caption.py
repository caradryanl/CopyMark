import logging
import glob, os, shutil, json
from pytorch_lightning import seed_everything

from hakubooru.logging import logger


def list_files_with_extension(directory, extension):
    # Ensure the extension starts with a dot
    if not extension.startswith("."):
        extension = "." + extension
    
    # Get a list of all files with the given extension
    files = glob.glob(os.path.join(directory, '**', '*' + extension), recursive=True)
    
    return files


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    num_samples = 2500
    seed_everything(1)

    # select hakubooru-member
    member_dataset = 'datasets/danbooru2023/images_member/'
    target_datasets = ['datasets/hakubooru-2-5k-member/', 'datasets/hakubooru-2-5k-eval-member/']
    member_files = list_files_with_extension(member_dataset, '.webp')
    print(f"number of members: {len(member_files)}")
    cnt = 0
    caption = {}
    for member_file in member_files:
        if cnt < num_samples:
            target_dataset = target_datasets[0]
        elif cnt < num_samples * 2:
            target_dataset = target_datasets[1]
        else:
            break

        if cnt == num_samples:
            with open(target_datasets[0] + 'caption.json', 'w') as file:
                json.dump(caption, file, indent=4)

        id = member_file[:-5]
        txt_file = id + '.txt'
        with open(os.path.join(member_dataset, txt_file), 'r') as file:
            prompt = file.read()

        caption[id]={
            "path": member_file,
            "height": None,
            "width": None,
            "caption": [prompt]
        }
        shutil.copy(os.path.join(member_dataset, member_file), target_dataset + 'images/' + member_file)
        cnt += 1

    with open(target_datasets[1] + 'caption.json', 'w') as file:
        json.dump(caption, file, indent=4)
        
    # select hakubooru-member
    nonmember_dataset = 'datasets/danbooru2023/images_nonmember/'
    target_datasets = ['datasets/hakubooru-2-5k-nonmember/', 'datasets/hakubooru-2-5k-eval-nonmember/']
    nonmember_files = list_files_with_extension(nonmember_dataset, '.webp')
    print(f"number of members: {len(member_files)}")
    cnt = 0
    caption = {}
    for nonmember_file in nonmember_files:
        if cnt < num_samples:
            target_dataset = target_datasets[0]
        elif cnt < num_samples * 2:
            target_dataset = target_datasets[1]
        else:
            break

        if cnt == num_samples:
            with open(target_datasets[0] + 'caption.json', 'w') as file:
                json.dump(caption, file, indent=4)

        id = nonmember_file[:-5]
        txt_file = id + '.txt'
        with open(os.path.join(nonmember_dataset, txt_file), 'r') as file:
            prompt = file.read()

        caption[id]={
            "path": nonmember_file,
            "height": None,
            "width": None,
            "caption": [prompt]
        }
        shutil.copy(os.path.join(nonmember_dataset, nonmember_file), target_dataset + 'images/' + nonmember_file)
        cnt += 1

    with open(target_datasets[1] + 'caption.json', 'w') as file:
        json.dump(caption, file, indent=4)



   

    