import sys
sys.path.append('..')
sys.path.append('.')
import tqdm
import torch
import numpy as np
import random
import os
import argparse

from stable_copyright import PIAStableDiffusionPipeline, SecMIDDIMScheduler
from stable_copyright import load_dataset, benchmark


def load_pipeline(ckpt_path, device='cuda:0'):
    pipe = PIAStableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
    pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

# difference from secmi: we return the sum of intermediate differences here
def get_reverse_denoise_results(pipe, dataloader, device, normalized):

    weight_dtype = torch.float32
    mean_l2 = 0
    scores_sum, scores_all_steps = [], []
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        latents, encoder_hidden_states = pipe.prepare_inputs(batch, weight_dtype, device)
        out = pipe(\
            prompt=None, latents=latents, prompt_embeds=encoder_hidden_states, \
                guidance_scale=1.0, num_inference_steps=100, normalized=normalized, strengh=0.5)
        _, posterior_results, denoising_results = out.images, out.posterior_results, out.denoising_results

        # print(f'posterior {posterior_results[0].shape}')

        for idx in range(posterior_results[0].shape[0]):
            
            score_all_step = []
            for i in range(len(denoising_results)):
                score_all_step.append(((denoising_results[i][idx, ...] - posterior_results[i][idx, ...]) ** 2).sum())
            score_all_step = torch.stack(score_all_step, dim=0) # torch.Size([50])
            score_sum = score_all_step.sum()

            scores_sum.append(score_sum.reshape(-1).detach().clone().cpu())    # List[torch.Size([1])]
            scores_all_steps.append(score_all_step.reshape(-1).detach().clone().cpu()) # List[torch.Size([50])]
        
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

    if args.normalized:
        pia_or_pian = 'pian'
    else:
        pia_or_pian = 'pia'

    if not args.use_ddp:

        if not os.path.exists(args.output):
            os.mkdir(args.output)

        member_scores_sum_step, member_scores_all_steps = get_reverse_denoise_results(pipe, member_loader, args.device, args.normalized)
        torch.save(member_scores_all_steps, args.output + f'{pia_or_pian}_member_scores_all_steps.pth')

        nonmember_scores_sum_step, nonmember_scores_all_steps = get_reverse_denoise_results(pipe, holdout_loader, args.device, args.normalized)
        torch.save(nonmember_scores_all_steps, args.output + f'{pia_or_pian}_nonmember_scores_all_steps.pth')

        member_corr_scores, nonmember_corr_scores = compute_corr_score(member_scores_all_steps, nonmember_scores_all_steps)
        
        benchmark(member_scores_sum_step, nonmember_scores_sum_step, f'{pia_or_pian}_sum_score', args.output)
        benchmark(member_corr_scores, nonmember_corr_scores, f'{pia_or_pian}_corr_score', args.output)

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
    args = parser.parse_args()

    fix_seed(args.seed)

    main(args)