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

from stable_copyright import load_dataset, benchmark, DRCStableDiffusionInpaintPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPImageProcessor

# git clone https://huggingface.co/openai/clip-vit-large-patch14

def load_pipeline(ckpt_path, device='cuda:0'):
    pipe = DRCStableDiffusionInpaintPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

def get_reverse_denoise_results(pipe, dataloader, device,):

    model_id = "models/diffusers/clip-vit-base-patch14"
    processor = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPTextModel.from_pretrained(model_id).to(device)

    weight_dtype = torch.float32
    mean_l2 = 0
    scores, path_log = [], []
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        path_log.extend(batch['path'])
        input_images = processor(images=batch['pixel_values'], return_tensors="pt").to(device)
        latents, encoder_hidden_states, masks = pipe.prepare_inputs(batch, weight_dtype, device)
        restored_images = pipe(mask_image=masks, prompt=None, latents=latents, prompt_embeds=encoder_hidden_states, \
                       guidance_scale=1.0, num_inference_steps=50).images

        with torch.no_grad():
            x_0 = model.get_image_features(**input_images)
            x_1 = model.get_image_features(**restored_images)
            x_0, x_1 = x_0.reshape((x_0.shape[0], -1)), x_1.reshape((x_1.shape[0], -1))
            cosine_sim = torch.sum(x_0 * x_1, dim=-1) / (torch.norm(x_0, 2, dim=-1) * torch.norm(x_1, 2, dim=-1)) 
            for score in cosine_sim:
                scores.append(score.detach().clone().cpu())
        
        mean_l2 += scores[-1]
        print(f'[{batch_idx}/{len(dataloader)}] mean l2-sum: {mean_l2 / (batch_idx + 1):.8f}')

        # if batch_idx > 0:
        #     break

    return torch.stack(scores, dim=0), path_log

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

    _, holdout_loader = load_dataset(args.dataset_root, args.ckpt_path, args.holdout_dataset, args.batch_size)
    _, member_loader = load_dataset(args.dataset_root, args.ckpt_path, args.member_dataset, args.batch_size)

    pipe = load_pipeline(args.ckpt_path, args.device)

    if not args.use_ddp:

        if not os.path.exists(args.output):
            os.mkdir(args.output)

        member_scores, member_path_log = get_reverse_denoise_results(pipe, member_loader, args.device)
        torch.save(member_scores, args.output + 'member_scores.pth')

        nonmember_scores, nonmember_path_log = get_reverse_denoise_results(pipe, holdout_loader, args.device)
        torch.save(nonmember_scores, args.output + 'nonmember_scores.pth')
        
        benchmark(member_scores, nonmember_scores, 'drc_score', args.output)

        with open(args.output + 'drc_image_log.json', 'w') as file:
            json.dump(dict(member=member_path_log, nonmember=nonmember_path_log), file, indent=4)

    else:
        raise NotImplementedError('DDP not implemented')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    running_time = dict(running_time=elapsed_time)
    
    with open(args.output + 'drc_running_time.json', 'w') as file:
        json.dump(running_time, file, indent=4)
    


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
    args = parser.parse_args()

    fix_seed(args.seed)

    main(args)
