import sys
sys.path.append('..')
sys.path.append('.')
import tqdm
import torch
import numpy as np
import random
import os
import argparse
import json,time
from accelerate import Accelerator
import pickle

from sklearn import preprocessing
from xgboost import XGBRegressor

from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from transformers import CLIPTokenizer

from copymark import SecMIDDIMScheduler, GSAStableDiffusionPipeline
from copymark import benchmark, test, Dataset, collate_fn


def load_dataset(dataset_root, ckpt_path, dataset: str | list[str], batch_size: int=6, model_type='sd'):
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

    if isinstance(dataset, str):
        train_dataset = Dataset(
            dataset=dataset,
            img_root=dataset_root,
            transforms=transform, 
            tokenizer=tokenizer)
    else:
        train_datasets = []
        for item in dataset:
            train_datasets.append(
                Dataset(
                    dataset=item,
                    img_root=dataset_root,
                    transforms=transform, 
                    tokenizer=tokenizer
                )
            )
        train_dataset = ConcatDataset(train_datasets)

    train_dataloader = DataLoader(
        train_dataset, pin_memory=True, shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    return train_dataset, train_dataloader

def load_pipeline(ckpt_path, device='cuda:0'):
    pipe = GSAStableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16)
    pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

# difference from secmi: we return the sum of intermediate differences here
def get_reverse_denoise_results(pipe, dataloader, device, demo):
    accelerator = Accelerator()
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=0
    )
    for param in pipe.unet.parameters():
        param.requires_grad = True
    pipe, optimizer, dataloader = accelerator.prepare(
        pipe, optimizer, dataloader
    )
    
    weight_dtype = torch.float16
    features, path_log = [], []
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        path_log.extend(batch['path'])
        latents, encoder_hidden_states, prompts = pipe.prepare_inputs(batch, weight_dtype, device)
        out = pipe(\
            accelerator=accelerator, optimizer=optimizer, prompt=prompts, latents=latents, \
            prompt_embeds=encoder_hidden_states, guidance_scale=1.0, num_inference_steps=20, gsa_mode=2)
        gsa_features = out.gsa_features # # [bsz x Tensor(num_p)]
        for feature in gsa_features:
            features.append(feature.detach().clone().cpu())
        if demo and batch_idx > 0:
            break

    return torch.stack(features, dim=0), path_log

def preprocess(member, non_member):
    member = member[0:non_member.shape[0]]
    member_y_np = np.zeros(member.shape[0])
    nonmember_y_np = np.ones(non_member.shape[0])
    x = np.vstack((member, non_member))
    x_max, x_min, x_avg = x[np.isfinite(x)].max(), x[np.isfinite(x)].min(), x[np.isfinite(x)].mean()
    x = np.nan_to_num(x, nan=x_avg, posinf=x_max, neginf=x_min) # deal with exploding gradients

    x = normalize(x)
    x = preprocessing.scale(x)
    y = np.concatenate((member_y_np, nonmember_y_np))

    def has_nan_inf(array):
        has_nan = np.isnan(array).any()
        has_inf = np.isinf(array).any()
        return has_nan or has_inf
    
    print("X includes nan or inf? {}".format(has_nan_inf(x)))
    return x, y

def train_xgboost(member_features, nonmember_features):
    x, y = preprocess(member_features, nonmember_features)
    model = XGBRegressor(n_estimators=50, max_depth=2)
    model.fit(x, y)
    y_pred = model.predict(x)
    member_scores = torch.tensor(y_pred[y <= 0.5])
    nonmember_scores = torch.tensor(y_pred[y > 0.5])
    return model, member_scores, nonmember_scores

def test_xgboost(xgb_save_path, member_features, nonmember_features):
    x, y = preprocess(member_features, nonmember_features)
    with open(xgb_save_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(x)
    member_scores = torch.tensor(y_pred[y <= 0.5])
    nonmember_scores = torch.tensor(y_pred[y > 0.5])

    return member_scores, nonmember_scores

def normalize(array):
    eps = 1e-5
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val + eps)
    return normalized_array

def main(args):
    start_time = time.time()
    _, holdout_loader = load_dataset(args.dataset_root, args.ckpt_path, args.holdout_dataset, args.batch_size)
    _, member_loader = load_dataset(args.dataset_root, args.ckpt_path, args.member_dataset, args.batch_size)
    pipe = load_pipeline(args.ckpt_path, args.device)

    if not args.use_ddp:
        os.makedirs(args.output, exist_ok=True)

        # phase 1: get the training feature
        member_features, member_path_log = get_reverse_denoise_results(pipe, member_loader, args.device, args.demo)
        nonmember_features, nonmember_path_log = get_reverse_denoise_results(pipe, holdout_loader, args.device, args.demo)
        member_features, nonmember_features = member_features.numpy(), nonmember_features.numpy()
        features = np.vstack((member_features, nonmember_features))
        if not args.eval:
            with open(args.output + f'laion_ridar_features.npy', 'wb') as f:
                np.save(f, features)
            with open(args.output + f'laion_ridar_image_log.json', 'w') as file:
                json.dump(dict(member=member_path_log, nonmember=nonmember_path_log), file, indent=4)

            # train a xgboost
            xgb, member_scores, nonmember_scores = train_xgboost(member_features, nonmember_features)
            with open(args.output + f'xgboost_laion_ridar.bin', 'wb') as f:
                pickle.dump(xgb, f)

            benchmark(member_scores, nonmember_scores, f'laion_ridar_score', args.output)

            end_time = time.time()
            elapsed_time = end_time - start_time
            running_time = dict(running_time=elapsed_time)
            with open(args.output + f'laion_ridar_running_time.json', 'w') as file:
                json.dump(running_time, file, indent=4)
        else:
            with open(args.output + f'laion_ridar_features_test.npy', 'wb') as f:
                np.save(f, features)
            with open(args.output + f'laion_ridar_image_log_test.json', 'w') as file:
                json.dump(dict(member=member_path_log, nonmember=nonmember_path_log), file, indent=4)

            # test the trained xgboost
            xgb_save_path = args.threshold_root + f'laion_ridar/xgboost_laion_ridar.bin'
            member_scores, nonmember_scores = test_xgboost(xgb_save_path, member_features, nonmember_features)

            threshold_path = args.threshold_root + f'laion_ridar/'
            test(member_scores, nonmember_scores, f'laion_ridar_score', args.output, threshold_path)

    else:
        raise NotImplementedError('DDP not implemented')


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--member-dataset', default='laion-aesthetic/eval')
    parser.add_argument('--holdout-dataset', nargs='+', default=['cc12m/eval', 'yfcc100m/eval', 'datacomp/eval', 'coco2017-val/eval'])
    parser.add_argument('--dataset-root', default='datasets/', type=str)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--ckpt-path', type=str, default='models/diffusers/stable-diffusion-v1-5/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='outputs/')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--use-ddp', type=bool, default=False)
    parser.add_argument('--demo', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--threshold-root', type=str, default='experiments/')
    args = parser.parse_args()

    fix_seed(args.seed)

    main(args)