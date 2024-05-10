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

from stable_copyright import PIAStableDiffusionPipeline, SecMIDDIMScheduler
from stable_copyright import load_dataset, benchmark

def image_perturbation(image, strength):
    perturbation = transforms.Compose([
        transforms.CenterCrop(size=int(512 * strength)),
        transforms.Resize(size=512, antialias=True),
    ])
    return perturbation(image)

def load_pipeline(ckpt_path, device='cuda:0'):
    pipe = PIAStableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
    pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

# difference from secmi: we return the sum of intermediate differences here
def get_reverse_denoise_results(pipe, dataloader, device, strengths):
    weight_dtype = torch.float32
    mean_l2 = 0
    scores_sum, scores_all_steps = [], []
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        original_batch = deepcopy(batch)
        # clean example
        latents, encoder_hidden_states = pipe.prepare_inputs(original_batch, weight_dtype, device)
        out = pipe(prompt=None, latents=latents, prompt_embeds=encoder_hidden_states, \
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
            latents, encoder_hidden_states = pipe.prepare_inputs(input_batch, weight_dtype, device)
            out = pipe(prompt=None, latents=latents, prompt_embeds=encoder_hidden_states, \
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

        # if batch_idx > 0:
        #     break

    return torch.stack(scores_sum, dim=0), torch.stack(scores_all_steps, dim=0)

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
    _, holdout_loader = load_dataset(args.dataset_root, args.ckpt_path, args.holdout_dataset, args.batch_size)
    _, member_loader = load_dataset(args.dataset_root, args.ckpt_path, args.member_dataset, args.batch_size)

    pipe = load_pipeline(args.ckpt_path, args.device)

    strengths = np.linspace(args.start_strength, args.end_strength, args.perturbation_number)

    if not args.use_ddp:

        if not os.path.exists(args.output):
            os.mkdir(args.output)

        member_scores_sum_step, member_scores_all_steps = get_reverse_denoise_results(pipe, member_loader, args.device, strengths)
        torch.save(member_scores_all_steps, args.output + 'pfami_member_scores_all_steps.pth')

        nonmember_scores_sum_step, nonmember_scores_all_steps = get_reverse_denoise_results(pipe, holdout_loader, args.device, strengths)
        torch.save(nonmember_scores_all_steps, args.output + 'pfami_nonmember_scores_all_steps.pth')

        member_corr_scores, nonmember_corr_scores = compute_corr_score(member_scores_all_steps, nonmember_scores_all_steps)
        
        benchmark(member_scores_sum_step, nonmember_scores_sum_step, 'pfami_sum_score', args.output)
        benchmark(member_corr_scores, nonmember_corr_scores, 'pfami_corr_score', args.output)

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
    parser.add_argument('--member-dataset', default='laion-aesthetic-2-5k', choices=['laion-aesthetic-2-5k'])
    parser.add_argument('--holdout-dataset', default='coco2017-val-2-5k', choices=['coco2017-val-2-5k'])
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
    args = parser.parse_args()

    fix_seed(args.seed)

    main(args)