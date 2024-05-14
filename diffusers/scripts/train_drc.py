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
from torchvision import transforms
from PIL import Image


from stable_copyright import benchmark, collate_fn, Dataset, DRCStableDiffusionInpaintPipeline, DRCLatentDiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer

# git clone https://huggingface.co/openai/clip-vit-large-patch14

preprocess = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

def load_dataset_drc(dataset_root, ckpt_path, dataset: str='laion-aesthetic-2-5k', batch_size: int=6, model_type: str='sd'):
    if model_type != 'ldm':
        resolution = 512
        transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
            ]
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            ckpt_path, subfolder="tokenizer", revision=None
        )
    else:
        resolution = 256
        transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(resolution),
                transforms.ToTensor(),
            ]
        )
        tokenizer = None
    train_dataset = Dataset(
        dataset=dataset,
        img_root=dataset_root,
        transforms=transform, tokenizer=tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    return train_dataset, train_dataloader

def load_pipeline(ckpt_path, device='cuda:0', model_type='sd'):
    if model_type == 'sd':
        pipe = DRCStableDiffusionInpaintPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
    elif model_type == 'ldm':
        pipe = DRCLatentDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        # pipe.scheduler = SecMIDDIMScheduler.from_config(pipe.scheduler.config)
    elif model_type == 'sdxl':
        raise NotImplementedError('SDXL not implemented yet')
    else:
        raise NotImplementedError(f'Unrecognized model type {model_type}')
    return pipe

def get_reverse_denoise_results(pipe, dataloader, device, output_path, mem_or_nonmem, demo):
    model_id = "../models/diffusers/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_id).to(device)

    log_root = output_path + f'{mem_or_nonmem}_restored'
    if not os.path.exists(log_root):
        os.mkdir(log_root)

    weight_dtype = torch.float32
    mean_l2 = 0
    scores, path_log = [], []
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        path_log.extend(batch['path'])
        original_images = batch['pixel_values'].detach().clone().to(device)

        images, encoder_hidden_states, masks = pipe.prepare_inputs(batch, weight_dtype, device)
        restored_images = pipe(mask_image=masks, prompt=None, image=images, prompt_embeds=encoder_hidden_states, \
                       guidance_scale=1.0, num_inference_steps=50, output_type='pt').images   # tensor
        restored_images = restored_images.detach().clone().clamp(0, 1)

        # store the image
        for idx, file_name in enumerate(batch['path']):
            log_image = torch.cat([original_images[idx, ...], restored_images[idx, ...]], dim=-1)
            log_image = (log_image.detach().clone().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
            log_image = Image.fromarray(log_image, mode='RGB')

            log_path = os.path.join(log_root, file_name)
            log_image.save(log_path, format='PNG')

        original_images = preprocess(original_images)
        restored_images = preprocess(restored_images)

        # print(original_images.shape, original_images.max(), original_images.min(),\
        #       restored_images.shape, restored_images.max(), restored_images.min(), )

        with torch.no_grad():
            x_0 = model.get_image_features(original_images)
            x_1 = model.get_image_features(restored_images)
            # print(x_0.shape, x_1.shape)
            x_0, x_1 = x_0.reshape((x_0.shape[0], -1)), x_1.reshape((x_1.shape[0], -1))
            cosine_sim = torch.sum(x_0 * x_1, dim=-1) / (torch.norm(x_0, 2, dim=-1) * torch.norm(x_1, 2, dim=-1)) 
            # print(cosine_sim.shape)
            for score in cosine_sim:
                score = -1.0 * score
                scores.append(score.detach().clone().cpu())
        
        mean_l2 += scores[-1]
        print(f'[{batch_idx}/{len(dataloader)}] mean l2-sum: {mean_l2 / (batch_idx + 1):.8f}')

        if demo and batch_idx > 0:
            break

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

    _, holdout_loader = load_dataset_drc(args.dataset_root, args.ckpt_path, args.holdout_dataset, args.batch_size, args.model_type)
    _, member_loader = load_dataset_drc(args.dataset_root, args.ckpt_path, args.member_dataset, args.batch_size, args.model_type)

    pipe = load_pipeline(args.ckpt_path, args.device, args.model_type)

    if not args.use_ddp:

        if not os.path.exists(args.output):
            os.mkdir(args.output)

        member_scores, member_path_log = get_reverse_denoise_results(pipe, member_loader, args.device, args.output, 'member', args.demo)
        torch.save(member_scores, args.output + f'drc_{args.model_type}_member_scores.pth')

        nonmember_scores, nonmember_path_log = get_reverse_denoise_results(pipe, holdout_loader, args.device, args.output, 'nonmember', args.demo)
        torch.save(nonmember_scores, args.output + f'drc_{args.model_type}_nonmember_scores.pth')
        
        benchmark(member_scores, nonmember_scores, f'drc_{args.model_type}_score', args.output)

        with open(args.output + f'drc_{args.model_type}_image_log.json', 'w') as file:
            json.dump(dict(member=member_path_log, nonmember=nonmember_path_log), file, indent=4)

    else:
        raise NotImplementedError('DDP not implemented')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    running_time = dict(running_time=elapsed_time)
    
    with open(args.output + f'drc_{args.model_type}_running_time.json', 'w') as file:
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
    parser.add_argument('--member-dataset', default='laion-aesthetic-2-5k')
    parser.add_argument('--holdout-dataset', default='coco2017-val-2-5k')
    parser.add_argument('--dataset-root', default='datasets/', type=str)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--ckpt-path', type=str, default='../models/diffusers/stable-diffusion-v1-5/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='outputs/')
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--use-ddp', type=bool, default=False)
    parser.add_argument('--model-type', type=str, choices=['sd', 'sdxl', 'ldm'], default='sd')
    parser.add_argument('--demo', type=bool, default=False)
    args = parser.parse_args()

    fix_seed(args.seed)

    main(args)
