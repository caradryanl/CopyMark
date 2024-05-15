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

from stable_copyright import SecMILatentDiffusionPipeline, SecMIStableDiffusionPipeline, SecMIDDIMScheduler
from stable_copyright import load_dataset, benchmark, test


def load_pipeline(ckpt_path, device='cuda:0', model_type='sd'):
    if model_type == 'sd':
        pipe = SecMIStableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    elif model_type == 'ldm':
        pipe = SecMILatentDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        # pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
    elif model_type == 'sdxl':
        raise NotImplementedError('SDXL not implemented yet')
    else:
        raise NotImplementedError(f'Unrecognized model type {model_type}')
    return pipe

def get_reverse_denoise_results(pipe, dataloader, device, demo=False):

    weight_dtype = torch.float32
    mean_l2 = 0
    scores_50_step, scores_all_steps, path_log = [], [], []
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        path_log.extend(batch['path'])
        latents, encoder_hidden_states = pipe.prepare_inputs(batch, weight_dtype, device)
        out = pipe(prompt=None, latents=latents, prompt_embeds=encoder_hidden_states, guidance_scale=1.0, num_inference_steps=100)
        _, posterior_results, denoising_results = out.images, out.posterior_results, out.denoising_results

        # print(f'posterior {posterior_results[0].shape}')

        for idx in range(posterior_results[0].shape[0]):
            score_50_step = ((denoising_results[14][idx, ...] - posterior_results[14][idx, ...]) ** 2).sum()
            # score_50_step.shape: torch.Size([])
            score_all_step = []
            for i in range(len(denoising_results)):
                score_all_step.append(((denoising_results[i][idx, ...] - posterior_results[i][idx, ...]) ** 2).sum())
            score_all_step = torch.stack(score_all_step, dim=0) # torch.Size([20])

            scores_50_step.append(score_50_step.reshape(-1).detach().clone().cpu())    # List[torch.Size([1])]
            scores_all_steps.append(score_all_step.reshape(-1).detach().clone().cpu()) # List[torch.Size([20])]
        
        mean_l2 += score_50_step
        print(f'[{batch_idx}/{len(dataloader)}] mean l2-sum: {mean_l2 / (batch_idx + 1):.8f}')

        if demo and batch_idx > 0:
            break

    return torch.stack(scores_50_step, dim=0), torch.stack(scores_all_steps, dim=0), path_log

def get_reverse_denoise_results_ddp(pipe, dataloader, prefix='member'):
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

    if not args.use_ddp:

        if not os.path.exists(args.output):
            os.mkdir(args.output)

        member_scores_50th_step, member_scores_all_steps, member_path_log = get_reverse_denoise_results(pipe, member_loader, args.device, args.demo)
        torch.save(member_scores_all_steps, args.output + f'secmi_{args.model_type}_member_scores_all_steps.pth')

        nonmember_scores_50th_step, nonmember_scores_all_steps, nonmember_path_log = get_reverse_denoise_results(pipe, holdout_loader, args.device, args.demo)
        torch.save(nonmember_scores_all_steps, args.output + f'secmi_{args.model_type}_nonmember_scores_all_steps.pth')

        member_corr_scores, nonmember_corr_scores = compute_corr_score(member_scores_all_steps, nonmember_scores_all_steps)
        
        if not args.eval:
            benchmark(member_scores_50th_step, nonmember_scores_50th_step, f'secmi_{args.model_type}_50th_score', args.output)
            benchmark(member_corr_scores, nonmember_corr_scores, f'secmi_{args.model_type}_corr_score', args.output)

            with open(args.output + f'secmi_{args.model_type}_image_log.json', 'w') as file:
                json.dump(dict(member=member_path_log, nonmember=nonmember_path_log), file, indent=4)
            end_time = time.time()
            elapsed_time = end_time - start_time
            running_time = dict(running_time=elapsed_time)
            with open(args.output + f'secmi_{args.model_type}_running_time.json', 'w') as file:
                json.dump(running_time, file, indent=4)
        else:
            threshold_path = args.threshold_root + f'{args.model_type}/secmi/'

            test(member_scores_50th_step, nonmember_scores_50th_step, f'secmi_{args.model_type}_50th_score', args.output, threshold_path)
            test(member_corr_scores, nonmember_corr_scores, f'secmi_{args.model_type}_corr_score', args.output, threshold_path)

            with open(args.output + f'secmi_{args.model_type}_image_log_test.json', 'w') as file:
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
    parser.add_argument('--model-type', type=str, choices=['sd', 'sdxl', 'ldm'], default='sd')
    parser.add_argument('--demo', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--threshold-root', type=str, default='experiments/')
    args = parser.parse_args()

    fix_seed(args.seed)

    main(args)
