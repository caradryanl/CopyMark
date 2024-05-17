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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing

from stable_copyright import GSALatentDiffusionPipeline, SecMIDDIMScheduler, GSAStableDiffusionPipeline
from stable_copyright import load_dataset, benchmark, test


def load_pipeline(ckpt_path, device='cuda:0', model_type='sd'):
    if model_type == 'sd':
        pipe = GSAStableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16)
        pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    elif model_type == 'ldm':
        pipe = GSALatentDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16)
        # pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
    elif model_type == 'sdxl':
        raise NotImplementedError('SDXL not implemented yet')
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
        latents, encoder_hidden_states = pipe.prepare_inputs(batch, weight_dtype, device)
        out = pipe(\
            accelerator=accelerator, optimizer=optimizer, prompt=None, latents=latents, \
            prompt_embeds=encoder_hidden_states, guidance_scale=1.0, num_inference_steps=20, gsa_mode=gsa_mode)
        gsa_features = out.gsa_features # # [bsz x Tensor(num_p)]
        # print(f"gsa: {gsa_features}")

        # print(f'posterior {posterior_results[0].shape}')

        for feature in gsa_features:
            features.append(feature.detach().clone().cpu())
        
        if demo and batch_idx > 0:
            break

    return torch.stack(features, dim=0), path_log

def get_reverse_denoise_results_ddp(pipe, dataloader):
    '''
        TODO:
        Implement the ddp sampling
    '''
    return None, None

def preprocess(member, non_member):
    member = member[0:non_member.shape[0]]
    member_y_np = np.zeros(member.shape[0])
    nonmember_y_np = np.ones(non_member.shape[0])
    x = np.vstack((member, non_member))
    y = np.concatenate((member_y_np, nonmember_y_np))
    x = preprocessing.scale(x)
    x = np.nan_to_num(x, nan=0)
    return x, y

def train_xgboost(member_features, nonmember_features):
    x, y = preprocess(member_features, nonmember_features)
    # print(x, y)
    xgb = SGDClassifier()

    def get_batches(X, y, batch_size):
        n_samples = X.shape[0]
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            yield X[start:end], y[start:end]
    
    # Training the model incrementally
    for X_batch, y_batch in get_batches(x, y, batch_size=100):
        xgb.partial_fit(X_batch, y_batch, classes=np.unique(y))
    # print(np.isnan(x).any())

    y_pred = batch_predict(xgb, x, batch_size=100)

    member_scores = torch.tensor(y_pred[y <= 0.5])
    nonmember_scores = torch.tensor(y_pred[y > 0.5])
    return xgb, member_scores, nonmember_scores

def batch_predict(model, X, batch_size):
    n_samples = X.shape[0]
    predictions = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        X_batch = X[start:end]
        batch_predictions = model.predict(X_batch)
        predictions.extend(batch_predictions)
    return np.array(predictions)

def test_xgboost(xgb_save_path, member_features, nonmember_features):
    x, y = preprocess(member_features, nonmember_features)
    
    with open(xgb_save_path, 'rb') as f:
        xgb = pickle.load(f)

    y_pred = batch_predict(xgb, x, batch_size=100)
    member_scores = torch.tensor(y_pred[y <= 0.5])
    nonmember_scores = torch.tensor(y_pred[y > 0.5])

    return member_scores, nonmember_scores

def main(args):
    start_time = time.time()

    _, holdout_loader = load_dataset(args.dataset_root, args.ckpt_path, args.holdout_dataset, args.batch_size, args.model_type)
    _, member_loader = load_dataset(args.dataset_root, args.ckpt_path, args.member_dataset, args.batch_size, args.model_type)

    pipe = load_pipeline(args.ckpt_path, args.device, args.model_type)

    if not args.use_ddp:

        if not os.path.exists(args.output):
            os.mkdir(args.output)

        # phase 1: get the training feature
        member_features, member_path_log = get_reverse_denoise_results(pipe, member_loader, args.device, args.gsa_mode, args.demo)
        nonmember_features, nonmember_path_log = get_reverse_denoise_results(pipe, holdout_loader, args.device, args.gsa_mode, args.demo)
         
        # save the features
        member_features, nonmember_features = member_features.numpy(), nonmember_features.numpy()
        member_features, nonmember_features = np.nan_to_num(member_features, nan=0), np.nan_to_num(nonmember_features, nan=0)

        if not args.eval:
            with open(args.output + f'gsa_{args.gsa_mode}_{args.model_type}_member_features.npy', 'wb') as f:
                np.save(f, member_features)
            with open(args.output + f'gsa_{args.gsa_mode}_{args.model_type}_nonmember_features.npy', 'wb') as f:
                np.save(f, nonmember_features)
            with open(args.output + f'gsa_{args.gsa_mode}_{args.model_type}_image_log.json', 'w') as file:
                json.dump(dict(member=member_path_log, nonmember=nonmember_path_log), file, indent=4)

            # train a xgboost
            xgb, member_scores, nonmember_scores = train_xgboost(member_features, nonmember_features)
            with open(args.output + f'xgboost_gsa_{args.gsa_mode}_{args.model_type}.bin', 'wb') as f:
                pickle.dump(xgb, f)

            benchmark(member_scores, nonmember_scores, f'gsa_{args.gsa_mode}_{args.model_type}_score', args.output)

            end_time = time.time()
            elapsed_time = end_time - start_time
            running_time = dict(running_time=elapsed_time)
            with open(args.output + f'gsa_{args.gsa_mode}_{args.model_type}_running_time.json', 'w') as file:
                json.dump(running_time, file, indent=4)
        else:
            with open(args.output + f'gsa_{args.gsa_mode}_{args.model_type}_member_features_test.npy', 'wb') as f:
                np.save(f, member_features)
            with open(args.output + f'gsa_{args.gsa_mode}_{args.model_type}_nonmember_features_test.npy', 'wb') as f:
                np.save(f, nonmember_features)
            with open(args.output + f'gsa_{args.gsa_mode}_{args.model_type}_image_log_test.json', 'w') as file:
                json.dump(dict(member=member_path_log, nonmember=nonmember_path_log), file, indent=4)

            # test the trained xgboost
            xgb_save_path = args.threshold_root + f'{args.model_type}/gsa_{args.gsa_mode}/xgboost_gsa_{args.gsa_mode}_{args.model_type}.bin'
            member_scores, nonmember_scores = test_xgboost(xgb_save_path, member_features, nonmember_features)

            threshold_path = args.threshold_root + f'{args.model_type}/gsa_{args.gsa_mode}/'
            test(member_scores, nonmember_scores, f'gsa_{args.gsa_mode}_{args.model_type}_score', args.output, threshold_path)

            TP = (member_scores <= 0.5).sum()
            TN = (nonmember_scores > 0.5).sum()
            FP = (nonmember_scores <= 0.5).sum()
            FN = (member_scores > 0.5).sum()
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)

            extra_output = dict(TPR=TPR.item(), FPR=FPR.item())
            with open(args.output + f'gsa_{args.gsa_mode}_{args.model_type}_score' + '_extra.json', 'w') as file:
                json.dump(extra_output, file, indent=4)

            
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
    parser.add_argument('--member-dataset', default='laion-aesthetic-2-5k')
    parser.add_argument('--holdout-dataset', default='coco2017-val-2-5k')
    parser.add_argument('--dataset-root', default='datasets/', type=str)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--ckpt-path', type=str, default='../models/diffusers/stable-diffusion-v1-5/')
    # parser.add_argument('--ckpt-path', type=str, default='../models/diffusers/ldm-celebahq-256/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='outputs/')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--use-ddp', type=bool, default=False)
    parser.add_argument('--gsa-mode', type=int, default=1, choices=[1, 2])
    parser.add_argument('--model-type', type=str, choices=['sd', 'sdxl', 'ldm'], default='sd')
    parser.add_argument('--demo', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--threshold-root', type=str, default='experiments/')
    args = parser.parse_args()

    fix_seed(args.seed)

    main(args)