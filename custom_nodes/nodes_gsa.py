import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import randn_tensor

from .nodes_diffusers import *

def progress_bar(iterable=None, total=None):
    if iterable is not None:
        return tqdm(iterable)
    elif total is not None:
        return tqdm(total=total)
    else:
        raise ValueError("Either `total` or `iterable` has to be defined.")
    
def get_timesteps(num_inference_steps, strength):
    timesteps = (
                np.linspace(0, 1000 - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
    timesteps = torch.from_numpy(timesteps)

    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = timesteps[t_start :]  # order=1
    # [601, 581, ..., 21, 1]
    return timesteps, num_inference_steps - t_start

def run_gsa(model, seed, prompts, added_cond_kwargs, latents, scheduler, gsa_mode, num_inference_steps=50, strength=1.0):
    is_member = False

    timesteps = get_timesteps(num_inference_steps, strength)
    
    accelerator = Accelerator()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0
    )
    for param in model.parameters():
        param.requires_grad = True
    model, optimizer = accelerator.prepare(
        model, optimizer
    )
    device = accelerator.device

    # get [x_201, x_181, ..., x_1]
    # print(timesteps)
    posterior_results = []
    original_latents = latents.detach().clone()
    for i, t in enumerate(timesteps): # from t_max to t_min
        noise = randn_tensor(original_latents.shape, device=device, dtype=original_latents.dtype)
        posterior_results.append(noise)
        
    # 7. Denoising loop
    denoising_results = []
    gsa_features = []
    with accelerator.accumulate(model):
        with progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise = posterior_results[i]
                # expand the latents if we are doing classifier free guidance
                # latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = original_latents.detach().clone()
                latent_model_input = scheduler.add_noise(latent_model_input, noise, t)

                # predict the noise residual
                noise_pred = model(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompts,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # no classifier free guidance
            
                # compute the previous noisy sample x_t -> x_t-1
                denoising_results.append(noise_pred)
                latents = scheduler.step(noise_pred, t, latent_model_input, return_dict=False)[0]

                progress_bar.update()

        if gsa_mode == 1:
            for i in range(original_latents.shape[0]):
                # compute the sum of the loss
                losses_i = None
                for j in range(len(denoising_results)):
                    loss_ij = (((denoising_results[j][i, ...] - posterior_results[j][i, ...]) ** 2).sum())
                    if not losses_i:
                        losses_i = loss_ij
                    else:
                        losses_i += loss_ij
                accelerator.backward(losses_i, retain_graph=True)

                # compute the gradient of the loss sum
                grads_i = []
                for p in model.parameters():
                    grads_i.append(torch.norm(p.grad).detach().clone())
                grads_i = torch.stack(grads_i, dim=0)   # [num_p]
                gsa_features.append(grads_i)
                optimizer.zero_grad()

            # gsa_features = torch.stack(gsa_features, dim=0) # [bsz, num_p]
        elif gsa_mode == 2:
            for i in range(original_latents.shape[0]):
                grads_i = []
                for j in range(len(denoising_results)):
                    # compute the loss
                    loss_ij = (((denoising_results[j][i, ...] - posterior_results[j][i, ...]) ** 2).sum())
                    accelerator.backward(loss_ij, retain_graph=True)

                    # compute the gradient
                    grads_ij = []
                    for p in model.parameters():
                        grads_ij.append(torch.norm(p.grad).detach().clone())
                    grads_ij = torch.stack(grads_ij, dim=0) # [num_p]

                    grads_i.append(grads_ij)   
                    optimizer.zero_grad()

                # compute the sum of gradients
                grads_i = torch.stack(grads_i, dim=0).sum(dim=0)   # [timestep, num_p] -> [num_p]
                gsa_features.append(grads_i)
            # gsa_features = torch.stack(gsa_features, dim=0) # [bsz, num_p]
        else:
            raise NotImplementedError(f"Mode {gsa_mode} out of 1 and 2")


    # Offload all models

    return is_member, gsa_features


GSA_MODE = ["gsa_1", "gsa_2"]

class GSA:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "prompts": ("CONDITIONING", ),
                    "added_cond_kwargs": ("CONDITIONING", ),
                    "latents": ("LATENT", ),
                    "scheduler": ("SCHEDULER", ),
                    "gsa_mode": (GSA_MODE, ),
                    "gsa_metadata": ("GSA_METADATA",),
                     }
                }

    RETURN_TYPES = ("INT", "IMAGE")
    FUNCTION = "gsa"

    CATEGORY = "sampling"

    def gsa(model, seed, prompts, added_cond_kwargs, latents, scheduler, gsa_mode, gsa_metadata, num_inference_steps=50, strength=1.0):
        return run_gsa(model, seed, prompts, added_cond_kwargs, latents, scheduler, gsa_mode, gsa_metadata, num_inference_steps, strength)
    
NODE_CLASS_MAPPINGS = {
    "GSA": GSA,
}