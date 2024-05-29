import sys
sys.path.append('..')
sys.path.append('.')
import tqdm
import torch
from torchvision import transforms

import numpy as np
import random
import os
import argparse
from copy import deepcopy
import time, json

from stable_copyright import PFAMIStableDiffusionPipeline, SecMIDDIMScheduler, PFAMILatentDiffusionPipeline, PFAMIStableDiffusionXLPipeline
from stable_copyright import load_dataset, benchmark, test

def image_perturbation(image, strength, image_size=512):
    perturbation = transforms.Compose([
        transforms.CenterCrop(size=int(image_size * strength)),
        transforms.Resize(size=image_size, antialias=True),
    ])
    return perturbation(image)

def load_pipeline(ckpt_path, device='cuda:0', model_type='sd'):
    if model_type == 'sd':
        pipe = PFAMIStableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    elif model_type == 'ldm':
        pipe = PFAMILatentDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        # pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
    elif model_type == 'sdxl' or model_type == 'kohaku':
        pipe = PFAMIStableDiffusionXLPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    else:
        raise NotImplementedError(f'Unrecognized model type {model_type}')
    return pipe

# difference from secmi: we return the sum of intermediate differences here
def get_reverse_denoise_results(pipe, dataloader, device, strengths, demo):
    weight_dtype = torch.float32
    mean_l2 = 0
    scores_sum, scores_all_steps, path_log, = [], [], []
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        path_log.extend(batch['path'])
        original_batch = deepcopy(batch)
        # clean example
        latents, encoder_hidden_states, prompts = pipe.prepare_inputs(original_batch, weight_dtype, device)
        out = pipe(prompt=prompts, latents=latents, prompt_embeds=encoder_hidden_states, \
                guidance_scale=1.0, num_inference_steps=100)
        _, ori_posterior_results, ori_denoising_results = out.images, out.posterior_results, out.denoising_results
        # [len(attack_timesteps) x [B, 4, 64, 64]]

        # compute loss
        ori_losses = []
        for i in range(len(ori_posterior_results)):
            ori_losses_batch = []
            for idx in range(ori_posterior_results[0].shape[0]):
                ori_loss = ((ori_posterior_results[i][idx, ...] - ori_denoising_results[i][idx, ...]) ** 2).sum()
                ori_losses_batch.append(ori_loss)
            ori_losses_batch = torch.stack(ori_losses_batch, dim=0).reshape(-1)
            ori_losses.append(ori_losses_batch)
        ori_losses = torch.stack(ori_losses, dim=0) # [T, B]

        perturb_losses = []
        for strength in strengths:
            input_batch = deepcopy(batch)
            input_batch["pixel_values"] = image_perturbation(input_batch["pixel_values"], strength)
            latents, encoder_hidden_states, prompts = pipe.prepare_inputs(input_batch, weight_dtype, device)
            out = pipe(prompt=prompts, latents=latents, prompt_embeds=encoder_hidden_states, \
                    guidance_scale=1.0, num_inference_steps=100)
            _, posterior_results, denoising_results = out.images, out.posterior_results, out.denoising_results
            # [len(attack_timesteps) x [B, 4, 64, 64]]
            perturb_losses_strength = []
            for i in range(len(posterior_results)):
                losses_batch = []
                for idx in range(posterior_results[0].shape[0]):
                    loss = ((posterior_results[i][idx, ...] - denoising_results[i][idx, ...]) ** 2).sum()
                    losses_batch.append(loss)
                losses_batch = torch.stack(losses_batch, dim=0).reshape(-1)
                perturb_losses_strength.append(losses_batch)
            perturb_losses_strength = torch.stack(perturb_losses_strength, dim=0) # [T, B]
            perturb_losses.append(perturb_losses_strength)

        perturb_losses = torch.stack(perturb_losses, dim=0) # [M, T, B]

        # compute the probability flunctuation delta_prob
        eps = 1e-6
        ori_losses = ori_losses.unsqueeze(dim=0).repeat(len(strengths),1, 1) # [M, T, B]
        delta_prob = (ori_losses - perturb_losses) / (ori_losses + eps) # [M, T, B]
        delta_prob = delta_prob.mean(dim=0) # [T, B]
        delta_prob_sum = delta_prob.sum(dim=0)   # [B]

        for item in delta_prob_sum:
            scores_sum.append(item.detach().clone().cpu()) # List[tensor]

        for idx in range(delta_prob.shape[1]):
            scores_all_steps.append(delta_prob[:, idx].detach().clone().cpu())
            
        mean_l2 += scores_sum[-1].item()
        print(f'[{batch_idx}/{len(dataloader)}] mean l2-sum: {mean_l2 / (batch_idx + 1):.8f}')

        if demo and batch_idx > 0:
            break

    return torch.stack(scores_sum, dim=0), torch.stack(scores_all_steps, dim=0), path_log

def get_reverse_denoise_results_ddp(pipe, dataloader):
    '''
        TODO:
        Implement the ddp sampling
    '''
    return None, None

def compute_corr_score(member_scores, nonmember_scores):
    '''
        member_scores: [N, S]
        nonmember_scores: [N, S]
    '''
    all_scores = torch.cat([member_scores, nonmember_scores], dim=0)    # [2N, S]
    timestep_mean = all_scores.mean(dim=0)  # [S]

    member_scores = (member_scores / timestep_mean).mean(dim=1) # [N]
    nonmember_scores = (nonmember_scores / timestep_mean).mean(dim=1)

    return member_scores, nonmember_scores


def main(args):
    start_time = time.time()

    _, holdout_loader = load_dataset(args.dataset_root, args.ckpt_path, args.holdout_dataset, args.batch_size, args.model_type)
    _, member_loader = load_dataset(args.dataset_root, args.ckpt_path, args.member_dataset, args.batch_size, args.model_type)

    pipe = load_pipeline(args.ckpt_path, args.device, args.model_type)

    strengths = np.linspace(args.start_strength, args.end_strength, args.perturbation_number)

    if not args.use_ddp:

        if not os.path.exists(args.output):
            os.mkdir(args.output)

        member_scores_sum_step, member_scores_all_steps, member_path_log = get_reverse_denoise_results(pipe, member_loader, args.device, strengths, args.demo)
        nonmember_scores_sum_step, nonmember_scores_all_steps, nonmember_path_log = get_reverse_denoise_results(pipe, holdout_loader, args.device, strengths, args.demo)
        member_corr_scores, nonmember_corr_scores = compute_corr_score(member_scores_all_steps, nonmember_scores_all_steps)

        if not args.eval:
            torch.save(member_scores_all_steps, args.output + f'pfami_{args.model_type}_member_scores_all_steps.pth')
            torch.save(nonmember_scores_all_steps, args.output + f'pfami_{args.model_type}_nonmember_scores_all_steps.pth')

            benchmark(member_scores_sum_step, nonmember_scores_sum_step, f'pfami_{args.model_type}_sum_score', args.output)
            benchmark(member_corr_scores, nonmember_corr_scores, f'pfami_{args.model_type}_corr_score', args.output)

            with open(args.output + f'pfami_{args.model_type}_image_log.json', 'w') as file:
                json.dump(dict(member=member_path_log, nonmember=nonmember_path_log), file, indent=4)

            end_time = time.time()
            elapsed_time = end_time - start_time
            running_time = dict(running_time=elapsed_time)
            with open(args.output + f'pfami_{args.model_type}_running_time.json', 'w') as file:
                json.dump(running_time, file, indent=4)
        else:
            torch.save(member_scores_all_steps, args.output + f'pfami_{args.model_type}_member_scores_all_steps_test.pth')
            torch.save(nonmember_scores_all_steps, args.output + f'pfami_{args.model_type}_nonmember_scores_all_steps_test.pth')
            threshold_path = args.threshold_root + f'{args.model_type}/pfami/'

            test(member_scores_sum_step, nonmember_scores_sum_step, f'pfami_{args.model_type}_sum_score', args.output, threshold_path)
            test(member_corr_scores, nonmember_corr_scores, f'pfami_{args.model_type}_corr_score', args.output, threshold_path)

            with open(args.output + f'pfami_{args.model_type}_image_log_test.json', 'w') as file:
                json.dump(dict(member=member_path_log, nonmember=nonmember_path_log), file, indent=4)

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
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='outputs/')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--use-ddp', type=bool, default=False)
    parser.add_argument('--normalized', type=bool, default=False)
    parser.add_argument('--perturbation-number', type=int, default=10)
    parser.add_argument('--start-strength', type=float, default=0.95)
    parser.add_argument('--end-strength', type=float, default=0.7)
    parser.add_argument('--model-type', type=str, choices=['sd', 'sdxl', 'ldm', 'kohaku'], default='sd')
    parser.add_argument('--demo', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--threshold-root', type=str, default='experiments/')
    args = parser.parse_args()

    fix_seed(args.seed)

    main(args)