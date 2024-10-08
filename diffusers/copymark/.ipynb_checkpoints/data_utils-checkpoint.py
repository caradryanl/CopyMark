import torch
from torchvision import transforms
from transformers import CLIPTokenizer

from sklearn import metrics
import torch
import numpy as np
import json
from PIL import Image
import os
from typing import Callable, Optional, Any, Tuple, List

def test(member_scores, nonmember_scores, experiment, output_path, threshold_path):
    with open(threshold_path + experiment + '_result.json', 'r') as file:
        result = json.load(file)

    best_threshold_at_1_FPR = result['best_threshold_at_1_FPR']
    best_threshold_at_01_FPR = result['best_threshold_at_01_FPR']

    min_score = min(member_scores.min(), nonmember_scores.min())
    max_score = max(member_scores.max(), nonmember_scores.max())

    TPR_list = []
    FPR_list = []
    threshold_list = []
    output = {}

    total = member_scores.size(0) + nonmember_scores.size(0)
    for threshold in torch.arange(min_score, max_score, (max_score - min_score) / 10000):
        acc = ((member_scores <= threshold).sum() + (nonmember_scores > threshold).sum()) / total

        TP = (member_scores <= threshold).sum()
        TN = (nonmember_scores > threshold).sum()
        FP = (nonmember_scores <= threshold).sum()
        FN = (member_scores > threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())
        threshold_list.append(threshold.item())

    TP = (member_scores <= best_threshold_at_1_FPR).sum()
    TN = (nonmember_scores > best_threshold_at_1_FPR).sum()
    FP = (nonmember_scores <= best_threshold_at_1_FPR).sum()
    FN = (member_scores > best_threshold_at_1_FPR).sum()
    TPR_at_1_threshold = TP / (TP + FN)
    FPR_at_1_threshold = FP / (FP + TN)
    TPR_at_1_threshold, FPR_at_1_threshold = TPR_at_1_threshold.item(), FPR_at_1_threshold.item()

    TP = (member_scores <= best_threshold_at_01_FPR).sum()
    TN = (nonmember_scores > best_threshold_at_01_FPR).sum()
    FP = (nonmember_scores <= best_threshold_at_01_FPR).sum()
    FN = (member_scores > best_threshold_at_01_FPR).sum()
    TPR_at_01_threshold = TP / (TP + FN)
    FPR_at_01_threshold = FP / (FP + TN)
    TPR_at_01_threshold, FPR_at_01_threshold = TPR_at_01_threshold.item(), FPR_at_01_threshold.item()

    # print(f'Score threshold = {threshold:.16f} \t ASR: {acc:.8f} \t TPR: {TPR:.8f} \t FPR: {FPR:.8f}')
    auc = metrics.auc(np.asarray(FPR_list), np.asarray(TPR_list))
    print(f'AUROC: {auc}')
    print(f'TPR_at_1_threshold: {TPR_at_1_threshold}, FPR_at_1_threshold: {FPR_at_1_threshold}')
    print(f'TPR_at_01_threshold: {TPR_at_01_threshold}, FPR_at_01_threshold: {FPR_at_01_threshold}')

    output['TPR_at_1_threshold'] = TPR_at_1_threshold
    output['FPR_at_1_threshold'] = FPR_at_1_threshold
    output['TPR_at_01_threshold'] = TPR_at_01_threshold
    output['FPR_at_01_threshold'] = FPR_at_01_threshold
    output['AUROC'] = auc
    output['TPR'] = TPR_list
    output['FPR'] = FPR_list
    output['threshold'] = threshold_list


    with open(output_path + experiment + '_result_test.json', 'w') as file:
        json.dump(output, file, indent=4)

def benchmark(member_scores, nonmember_scores, experiment, output_path):
    # print(member_scores, nonmember_scores)
    min_score = min(member_scores.min(), nonmember_scores.min())
    max_score = max(member_scores.max(), nonmember_scores.max())

    TPR_list = []
    FPR_list = []
    threshold_list = []
    output = {}

    best_TPR_at_1_FPR, best_TPR_at_01_FPR = 0.0, 0.0
    best_threshold_at_1_FPR, best_threshold_at_01_FPR = 0.0, 0.0

    total = member_scores.size(0) + nonmember_scores.size(0)
    for threshold in torch.arange(min_score, max_score, (max_score - min_score) / 10000):
        acc = ((member_scores <= threshold).sum() + (nonmember_scores > threshold).sum()) / total

        TP = (member_scores <= threshold).sum()
        TN = (nonmember_scores > threshold).sum()
        FP = (nonmember_scores <= threshold).sum()
        FN = (member_scores > threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())
        threshold_list.append(threshold.item())

        if FPR <= 0.01 and TPR > best_TPR_at_1_FPR:
            best_TPR_at_1_FPR = TPR.item()
            best_threshold_at_1_FPR = threshold.item()
        if FPR <= 0.001 and TPR > best_TPR_at_01_FPR:
            best_TPR_at_01_FPR = TPR.item()
            best_threshold_at_01_FPR = threshold.item()

        # print(f'Score threshold = {threshold:.16f} \t ASR: {acc:.8f} \t TPR: {TPR:.8f} \t FPR: {FPR:.8f}')
    auc = metrics.auc(np.asarray(FPR_list), np.asarray(TPR_list))
    print(f'AUROC: {auc}')
    print(f'best_TPR_at_1_FPR: {best_TPR_at_1_FPR}, best_threshold_at_1_FPR: {best_threshold_at_1_FPR}')
    print(f'best_TPR_at_01_FPR: {best_TPR_at_01_FPR}, best_threshold_at_01_FPR: {best_threshold_at_01_FPR}')

    output['AUROC'] = auc
    output['best_TPR_at_1_FPR'] = best_TPR_at_1_FPR
    output['best_threshold_at_1_FPR'] = best_threshold_at_1_FPR
    output['best_TPR_at_01_FPR'] = best_TPR_at_01_FPR
    output['best_threshold_at_01_FPR'] = best_threshold_at_01_FPR
    output['TPR'] = TPR_list
    output['FPR'] = FPR_list
    output['threshold'] = threshold_list

    with open(output_path + experiment + '_result.json', 'w') as file:
        json.dump(output, file, indent=4)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    if examples[0]["input_ids"] == None:
        input_ids = None
    else:
        input_ids = torch.stack([example["input_ids"] for example in examples])
    path = [example["path"] for example in examples]
    prompts = [example["prompt"] for example in examples]
    return {"pixel_values": pixel_values, "input_ids": input_ids, "path": path, "prompts": prompts}

class StandardTransform: 
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")

        return "\n".join(body)


class Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset: str,
            img_root: str,
            transforms: Optional[Callable] = None,
            tokenizer=None,
    ) -> None:
        self.dataset = dataset
        self.img_root = img_root
        self.tokenizer = tokenizer
        self.transforms = transforms
        caption_path = os.path.join(img_root, dataset, 'caption.json')
        # load list file
        self.img_info = []
        with open(caption_path, 'r') as json_file:
            img_info = json.load(json_file)
        for value in img_info.values():
            self.img_info.append(value)

        self._init_tokenize_captions()


    def __len__(self):
        return len(self.img_info)


    def _init_tokenize_captions(self):
        captions = []
        flag = False
        for metadata in self.img_info:
            if len(metadata['caption'])==0:
                captions.append(None)
            else:
                caption = metadata['caption'][0]
                captions.append(caption)
                flag=True
        
        if flag == False:
            self.input_ids = None
        else:
            inputs = self.tokenizer(
                captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
                return_tensors="pt"
            )
            self.input_ids = inputs.input_ids

    def _load_input_id(self, id: int):
        return self.input_ids[id] if self.input_ids is not None else None

    def __getitem__(self, index: int):
        img_name = self.img_info[index]['path']

        # image
        img_path = os.path.join(self.img_root, self.dataset, 'images', img_name)
        image = Image.open(img_path).convert("RGB")

        input_id = self._load_input_id(index)
        if len(self.img_info[index]['caption']) == 0:
            caption = None
        else:
            caption = self.img_info[index]['caption'][0]

        if self.transforms is not None:
            image, input_id = StandardTransform(self.transforms, None)(image, input_id)

        # return image, target
        return {"pixel_values": image, "input_ids": input_id, 'prompt': caption, 'path': img_name}


def load_dataset(dataset_root, ckpt_path, dataset: str='laion-aesthetic-2-5k', batch_size: int=6, model_type='sd'):
    
    if model_type != 'ldm':
        resolution = 512
        transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            ckpt_path, subfolder="tokenizer", revision=None
        )
    else:
        resolution = 256
        transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        tokenizer = None

    train_dataset = Dataset(
        dataset=dataset,
        img_root=dataset_root,
        transforms=transform, tokenizer=tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, pin_memory=True, shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    return train_dataset, train_dataloader