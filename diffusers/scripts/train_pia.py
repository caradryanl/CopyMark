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
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

from copymark import PIAStableDiffusionPipeline, SecMIDDIMScheduler, PIALatentDiffusionPipeline, PIAStableDiffusionXLPipeline
from copymark import load_dataset, benchmark, test


def load_pipeline(ckpt_path, device='cuda:0', model_type='sd'):
    if model_type == 'sd' or model_type == 'laion_mi':
        pipe = PIAStableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    elif model_type == 'ldm':
        pipe = PIALatentDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        # pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
    elif model_type == 'sdxl' or model_type == 'kohaku':
        pipe = PIAStableDiffusionXLPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    else:
        raise NotImplementedError(f'Unrecognized model type {model_type}')
    return pipe

# difference from secmi: we return the sum of intermediate differences here
def get_reverse_denoise_results(pipe, dataloader, device, normalized, demo):

    weight_dtype = torch.float32
    mean_l2 = 0
    scores_sum, scores_all_steps, path_log = [], [], []
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        path_log.extend(batch['path'])
        latents, encoder_hidden_states, prompts = pipe.prepare_inputs(batch, weight_dtype, device)
        out = pipe(\
            prompt=prompts, latents=latents, prompt_embeds=encoder_hidden_states, \
                guidance_scale=1.0, num_inference_steps=100, normalized=normalized, strength=0.5)
        _, posterior_results, denoising_results = out.images, out.posterior_results, out.denoising_results

        # print(f'posterior {posterior_results[0].shape}')

        for idx in range(posterior_results[0].shape[0]):
            
            score_all_step = []
            for i in range(len(denoising_results)):
                score_all_step.append(((denoising_results[i][idx, ...] - posterior_results[i][idx, ...]) ** 2).sum())
            score_all_step = torch.stack(score_all_step, dim=0) # torch.Size([50])
            # print(score_all_step.shape)
            score_sum = score_all_step.sum()

            scores_sum.append(score_sum.reshape(-1).detach().clone().cpu())    # List[torch.Size([1])]
            scores_all_steps.append(score_all_step.reshape(-1).detach().clone().cpu()) # List[torch.Size([50])]
        
        mean_l2 += score_sum.item()
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

    if args.normalized:
        pia_or_pian = 'pian'
    else:
        pia_or_pian = 'pia'

    if not args.use_ddp:

        if not os.path.exists(args.output):
            os.mkdir(args.output)

        member_scores_sum_step, member_scores_all_steps, member_path_log = get_reverse_denoise_results(pipe, member_loader, args.device, args.normalized, args.demo)
        nonmember_scores_sum_step, nonmember_scores_all_steps, nonmember_path_log = get_reverse_denoise_results(pipe, holdout_loader, args.device, args.normalized, args.demo)
        member_corr_scores, nonmember_corr_scores = compute_corr_score(member_scores_all_steps, nonmember_scores_all_steps)
        
        
        if not args.eval:
            torch.save(member_scores_all_steps, args.output + f'{pia_or_pian}_{args.model_type}_member_scores_all_steps.pth')
            torch.save(nonmember_scores_all_steps, args.output + f'{pia_or_pian}_{args.model_type}_nonmember_scores_all_steps.pth')

            benchmark(member_scores_sum_step, nonmember_scores_sum_step, f'{pia_or_pian}_{args.model_type}_sum_score', args.output)
            benchmark(member_corr_scores, nonmember_corr_scores, f'{pia_or_pian}_{args.model_type}_corr_score', args.output)

            with open(args.output + f'{pia_or_pian}_{args.model_type}_image_log.json', 'w') as file:
                json.dump(dict(member=member_path_log, nonmember=nonmember_path_log), file, indent=4)

            end_time = time.time()
            elapsed_time = end_time - start_time
            running_time = dict(running_time=elapsed_time)
            with open(args.output + f'{pia_or_pian}_{args.model_type}_running_time.json', 'w') as file:
                json.dump(running_time, file, indent=4)
        else:
            torch.save(member_scores_all_steps, args.output + f'{pia_or_pian}_{args.model_type}_member_scores_all_steps_test.pth')
            torch.save(nonmember_scores_all_steps, args.output + f'{pia_or_pian}_{args.model_type}_nonmember_scores_all_steps_test.pth')
            threshold_path = args.threshold_root + f'{args.model_type}/{pia_or_pian}/'
            test(member_scores_sum_step, nonmember_scores_sum_step, f'{pia_or_pian}_{args.model_type}_sum_score', args.output, threshold_path)
            test(member_corr_scores, nonmember_corr_scores, f'{pia_or_pian}_{args.model_type}_corr_score', args.output, threshold_path)

            with open(args.output + f'{pia_or_pian}_{args.model_type}_image_log_test.json', 'w') as file:
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
    # parser.add_argument('--ckpt-path', type=str, default='../models/diffusers/ldm-celebahq-256/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='outputs/')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--use-ddp', type=bool, default=False)
    parser.add_argument('--normalized', type=bool, default=False)
    parser.add_argument('--model-type', type=str, choices=['sd', 'sdxl', 'ldm', 'kohaku', 'laion_mi'], default='sd')
    parser.add_argument('--demo', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--threshold-root', type=str, default='experiments/')
    args = parser.parse_args()

    fix_seed(args.seed)

    main(args)