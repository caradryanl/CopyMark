import sys
sys.path.append('..')
sys.path.append('.')
import tqdm
from sklearn import metrics
from torchvision import transforms
import torch
import numpy as np
import json
import random
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import os
from typing import Iterable, Callable, Optional, Any, Tuple, List
import argparse

from stable_copyright import SecMIStableDiffusionPipeline, SecMIDDIMScheduler


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


def load_dataset(dataset_root, dataset: str='laion-aesthetic-2-5k'):
    resolution = 512
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_dataset = Dataset(
        dataset=dataset,
        img_root=dataset_root,
        transforms=transform, tokenizer=tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1
    )
    return train_dataset, train_dataloader


def load_pipeline(ckpt_path, device='cuda:0'):
    pipe = SecMIStableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
    pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

def get_reverse_denoise_results(pipe, dataloader, prefix='member'):

    weight_dtype = torch.float32
    mean_l2 = 0
    scores = []
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        # Convert images to latent space
        pixel_values = batch["pixel_values"].to(weight_dtype)
        pixel_values = pixel_values.cuda()
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215
        # Get the text embedding for conditioning
        input_ids = batch["input_ids"].cuda()
        encoder_hidden_states = text_encoder(input_ids)[0]

        out = pipe(prompt=None, latents=latents, prompt_embeds=encoder_hidden_states, guidance_scale=1.0, num_inference_steps=100)
        image, posterior_results, denoising_results = out.images, out.posterior_results, out.denoising_results

        score = ((denoising_results[14] - posterior_results[14]) ** 2).sum()
        scores.append(score.reshape(-1, 1))
        mean_l2 += score
        print(f'[{batch_idx}/{len(dataloader)}] mean l2-sum: {mean_l2 / (batch_idx + 1):.8f}')

    return torch.concat(scores).reshape(-1)


def main(args):
    _, holdout_loader = load_dataset(args.dataset_root, args.holdout_dataset)
    _, member_loader = load_dataset(args.dataset_root, args.member_dataset)

    pipe = load_pipeline(args.ckpt_path, args.device)

    member_scores = get_reverse_denoise_results(pipe, member_loader)
    nonmember_scores = get_reverse_denoise_results(pipe, holdout_loader)

    min_score = min(member_scores.min(), nonmember_scores.min())
    max_score = max(member_scores.max(), nonmember_scores.max())

    TPR_list = []
    FPR_list = []
    threshold_list = []
    output = {}

    total = member_scores.size(0) + nonmember_scores.size(0)
    for threshold in torch.range(min_score, max_score, (max_score - min_score) / 10000):
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

        print(f'Score threshold = {threshold:.16f} \t ASR: {acc:.8f} \t TPR: {TPR:.8f} \t FPR: {FPR:.8f}')
    auc = metrics.auc(np.asarray(FPR_list), np.asarray(TPR_list))
    print(f'AUROC: {auc}')

    output['TPR'] = TPR_list
    output['FPR'] = FPR_list
    output['threshold'] = threshold_list

    with open(args.output, 'w') as file:
        json.dump(output, file, indent=4)


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--member-dataset', default='laion-aesthetic-2-5k', choices=['laion-aesthetic-2-5k'])
    parser.add_argument('--holdout-dataset', default='coco2017-val-2-5k', choices=['coco2017-val-2-5k'])
    parser.add_argument('--dataset-root', default='datasets/', type=str)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--ckpt-path', type=str, default='../models/diffusers/stable-diffusion-v1-5/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='outputs/secmi.json')
    args = parser.parse_args()

    tokenizer = CLIPTokenizer.from_pretrained(
        args.ckpt_path, subfolder="tokenizer", revision=None
    )
    # tokenizer = tokenizer.cuda()

    text_encoder = CLIPTextModel.from_pretrained(
        args.ckpt_path, subfolder="text_encoder", revision=None
    )
    text_encoder = text_encoder.to(args.device)

    vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    vae = vae.to(args.device)

    unet = UNet2DConditionModel.from_pretrained(
        args.ckpt_path, subfolder="unet", revision=None
    )
    unet = unet.to(args.device)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    fix_seed(args.seed)

    main(args)
