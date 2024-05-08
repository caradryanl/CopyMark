import torch
from torchvision import transforms
from transformers import CLIPTokenizer

import json
from PIL import Image
import os
from typing import Callable, Optional, Any, Tuple, List


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

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
        for metadata in self.img_info:
            caption = metadata['caption'][0]
            captions.append(caption)

        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        self.input_ids = inputs.input_ids

    def _load_input_id(self, id: int):
        return self.input_ids[id]

    def __getitem__(self, index: int):
        path = os.path.join(self.img_root, self.dataset, 'images', self.img_info[index]['path'])
        image = Image.open(path).convert("RGB")

        input_id = self._load_input_id(index)
        caption = self.img_info[index]['caption'][0]

        if self.transforms is not None:
            image, input_id = StandardTransform(self.transforms, None)(image, input_id)

        # return image, target
        return {"pixel_values": image, "input_ids": input_id, 'caption': caption}


def load_dataset(dataset_root, ckpt_path, dataset: str='laion-aesthetic-2-5k', batch_size: int=6):
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
    train_dataset = Dataset(
        dataset=dataset,
        img_root=dataset_root,
        transforms=transform, tokenizer=tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    return train_dataset, train_dataloader