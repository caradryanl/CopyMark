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
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from xgboost import XGBClassifier, XGBRegressor


from copymark import GSALatentDiffusionPipeline, SecMIDDIMScheduler, GSAStableDiffusionPipeline, GSAStableDiffusionXLPipeline
from copymark import load_dataset, benchmark, test


def load_pipeline(ckpt_path, device='cuda:0', model_type='sd'):
    if model_type == 'sd':
        pipe = GSAStableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16)
        pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    elif model_type == 'ldm':
        pipe = GSALatentDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16)
        # pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
    elif model_type == 'sdxl' or model_type == 'kohaku':
        pipe = GSAStableDiffusionXLPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16)
        pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    else:
        raise NotImplementedError(f'Unrecognized model type {model_type}')
    return pipe

# difference from secmi: we return the sum of intermediate differences here
def get_reverse_denoise_results(pipe, dataloader, device, gsa_mode, demo):
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
            prompt_embeds=encoder_hidden_states, guidance_scale=1.0, num_inference_steps=10, gsa_mode=gsa_mode)
        gsa_features = out.gsa_features # # [bsz x Tensor(num_p)]
        # print(f"gsa: {gsa_features}")

        # print(f'posterior {posterior_results[0].shape}')

        for feature in gsa_features:
            features.append(feature.detach().clone().cpu())
        
        if demo and batch_idx > 0:
            break

    return torch.stack(features, dim=0), path_log

def normalize(array, min=None, max=None):
    eps = 1e-5
    if min is not None and max is not None:
        min_val = min
        max_val = max
    else:
        min_val = np.min(array)
        max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val + eps)
    normalized_array = (normalized_array - 0.5) / 0.5
    return normalized_array, min_val, max_val

def preprocess(member, non_member, min=None, max=None):
    member = member[0:non_member.shape[0]]
    member_y_np = np.zeros(member.shape[0])
    nonmember_y_np = np.ones(non_member.shape[0])
    x = np.vstack((member, non_member))
    x_max, x_min, x_avg = x[np.isfinite(x)].max(), x[np.isfinite(x)].min(), x[np.isfinite(x)].mean()
    x = np.nan_to_num(x, nan=x_avg, posinf=x_max, neginf=x_min) # deal with exploding gradients

    x, min, max = normalize(x, min, max)
    y = np.concatenate((member_y_np, nonmember_y_np))

    def has_nan_inf(array):
        has_nan = np.isnan(array).any()
        has_inf = np.isinf(array).any()
        return has_nan or has_inf
    
    print(has_nan_inf(x))
    return x, y, min, max

def train_model(x, y):

    model = XGBRegressor(n_estimators=50, max_depth=2)
    model.fit(x, y)

    def get_batches(X, y, batch_size):
        n_samples = X.shape[0]
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            yield X[start:end], y[start:end]

    # y_pred = batch_predict(xgb, x, batch_size=100)
    y_pred = model.predict(x)

    member_scores = torch.tensor(y_pred[y <= 0.5])
    nonmember_scores = torch.tensor(y_pred[y > 0.5])
    # print(member_scores[0:10], nonmember_scores[0:10])
    return model, member_scores, nonmember_scores

def batch_predict(model, X, batch_size):
    n_samples = X.shape[0]
    predictions = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        X_batch = X[start:end]
        batch_predictions = model.predict(X_batch)
        predictions.extend(batch_predictions)
    return np.array(predictions)

def test_model(model, x, y):

    # y_pred = batch_predict(xgb, x, batch_size=100)
    y_pred = model.predict(x)
    member_scores = torch.tensor(y_pred[y <= 0.5])
    nonmember_scores = torch.tensor(y_pred[y > 0.5])

    # print(member_scores[0:10], nonmember_scores[0:10])

    return member_scores, nonmember_scores

def infer_model(model, x):
    # y_pred = batch_predict(xgb, x, batch_size=100)
    y_pred = model.predict(x)
    # print(member_scores[0:10], nonmember_scores[0:10])

    return torch.tensor(y_pred)


def main(args):
    start_time = time.time()

    _, case_loader = load_dataset(args.dataset_root, args.ckpt_path, args.dataset, args.batch_size, args.model_type)

    pipe = load_pipeline(args.ckpt_path, args.device, args.model_type)


    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # phase 1: get the training feature
    case_features, case_img_log = get_reverse_denoise_results(pipe, case_loader, args.device, args.gsa_mode, args.demo)
    case_features = case_features.numpy()

    with open(f'experiments/{args.model_type}/gsa_{args.gsa_mode}/gsa_{args.gsa_mode}_{args.model_type}_features.npy', 'rb') as f:
        features = np.load(f)
    with open(f'experiments/{args.model_type}/gsa_{args.gsa_mode}/gsa_{args.gsa_mode}_{args.model_type}_sum_score_result.json', 'rb') as f:
                result = json.load(f)
    threshold = result['best_threshold_at_1_FPR']
    
    data_size = len(features)//2
    member_features, nonmember_features = features[0 :data_size], features[data_size:]
    # print(f"member: {len(member_features)}, nonmember: {len(nonmember_features)}")       

    # step 2: train the model
    x, y, x_min, x_max = preprocess(member_features, nonmember_features)
    model, member_scores, nonmember_scores = train_model(x, y)

    case_scores = infer_model(model, case_features)

    print(f'==>Results<==')
    for img_path, case_score in zip(case_img_log, case_scores):
        print(f'{img_path} {case_score <= 0.5}')

    data = pd.DataFrame({
            'member': member_scores,
            'non-member': nonmember_scores,
        })

    data_melted = data.melt(var_name='Category', value_name='Scores')
    data_melted['Method'] = f'gsa_{args.gsa_mode}'
    data_melted['Threshold'] = threshold

    fig = plt.figure(figsize=(108, 12))  # Set the overall figure siz
    mapping = {
        'gsa_1': 'GSA1',
        'gsa_2': 'GSA2',
    }
    colors = ['palegreen', 'gold']

    method = data_melted['Method'].iloc[0]
    threshold = data_melted['Threshold'].iloc[0]

    _, ax = plt.subplots(figsize=(20, 8))  # Create a new figure for each method

    categories = ['member', 'non-member']
    data_to_plot = [data_melted[data_melted['Category'] == cat]['Scores'].values for cat in categories]
    boxprops = dict(linewidth=3)
    medianprops = dict(linewidth=3, color='lightgray')
    whiskerprops = dict(linewidth=3)
    capprops = dict(linewidth=3)
    flierprops = dict(marker='o', color='green', markersize=12, alpha=1, markeredgewidth=2)

    ax.scatter(np.ones_like(case_scores), case_scores, marker='o', color='cyan', alpha=1.0, s=500, edgecolors="black", linewidths=2, zorder=3, label=f'Case scores')
    ax.scatter(2 * np.ones_like(case_scores), case_scores, marker='o', color='cyan', alpha=1.0, s=500, edgecolors="black", linewidths=2, zorder=3)
    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=categories, widths=0.3,
                    boxprops=boxprops, medianprops=medianprops,
                    whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
    

    # Change the color of each box
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    ax.set_title(f'{mapping[method]}', fontsize=43)
    ax.set_ylabel('Scores', fontsize=32)
    ax.tick_params(axis='y', labelsize=32)
    ax.tick_params(axis='x', labelsize=35)

    ax.yaxis.grid(True, linestyle='--', linewidth=1)
    ax.legend(fontsize=30)

    plt.subplots_adjust(left=0.15, right=0.99, top=0.88, bottom=0.12, wspace=0.4)  # Adjust subplot parameters
    plt.show()
    
    
    


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='examples')
    parser.add_argument('--dataset-root', default='assets/', type=str)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--ckpt-path', type=str, default='../models/diffusers/stable-diffusion-v1-5/')
    # parser.add_argument('--ckpt-path', type=str, default='../models/diffusers/ldm-celebahq-256/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='outputs/')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gsa-mode', type=int, default=2, choices=[1, 2])
    parser.add_argument('--model-type', type=str, choices=['sd', 'sdxl', 'ldm', 'kohaku'], default='sd')
    parser.add_argument('--demo', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--threshold-root', type=str, default='experiments/')
    args = parser.parse_args()

    fix_seed(args.seed)

    main(args)