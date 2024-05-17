import os
import torch
import numpy as np
import PIL.Image

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from diffusers import UNet2DModel, DDIMScheduler, VQModel
from accelerate import Accelerator

from diffusers.pipelines.stable_diffusion.pipeline_output import BaseOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import \
    (
        retrieve_timesteps,
        randn_tensor,
        DiffusionPipeline,
    )


from .secmi_scheduling_ddim import SecMIDDIMScheduler

@dataclass
class GSAStableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        gsa_features (`List[Tensor]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    gsa_features: Optional[List]

class GSALatentDiffusionPipeline(
    DiffusionPipeline
):
    def __init__(self, unet, vae, scheduler, device, generator):
        super().__init__()

        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.generator = generator
        self.execution_device = device

        
    @classmethod
    def from_pretrained(self, 
                        pretrained_model_name_or_path: Union[str, os.PathLike]="CompVis/ldm-celebahq-256",
                        torch_dtype: torch.dtype=torch.float32,
                        ):
        unet = UNet2DModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", torch_dtype=torch_dtype)
        vae = VQModel.from_pretrained(pretrained_model_name_or_path, subfolder="vqvae", torch_dtype=torch_dtype)
        scheduler = SecMIDDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

        # cuda and seed
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seed = 2024
        unet.to(device)
        vae.to(device)

        # generate gaussian noise to be decoded
        generator = torch.manual_seed(seed)
        scheduler.set_timesteps(num_inference_steps=50)

        return GSALatentDiffusionPipeline(unet=unet, vae=vae, scheduler=scheduler, device=device, generator=generator)

    def prepare_inputs(self, batch, weight_dtype, device):
        pixel_values = batch["pixel_values"].to(weight_dtype)
        if device == 'cuda':
            pixel_values  = pixel_values.cuda()
        latents = self.vae.encode(pixel_values)[0]
        encoder_hidden_states = None

        for param in self.unet.parameters():
            param.requires_grad = True

        return latents, encoder_hidden_states


    # borrow from Image2Image
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]  # order=1
        # [601, 581, ..., 21, 1]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def __call__(
        self,
        accelerator: Accelerator,
        optimizer: torch.optim.AdamW,
        latents: torch.FloatTensor,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        strength: float=1.0,
        gsa_mode: int = 1,
        prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 1.0,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        
        device = accelerator.device
        latents = latents.to(device=device)

        # check shape
        if len(latents.shape) == 3:
            latents = latents.unsqueeze(dim=0)
        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        posterior_results = []
        original_latents = latents.detach().clone()
        for i, t in enumerate(timesteps): # from t_max to t_min
            noise = randn_tensor(original_latents.shape, generator=generator, device=device, dtype=original_latents.dtype)
            posterior_results.append(noise)
            
        # 7. Denoising loop
        gsa_features = []
        denoising_results = []
        with accelerator.accumulate(self.unet):
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    noise = posterior_results[i]
                    
                    # expand the latents if we are doing classifier free guidance
                    # latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = original_latents.detach().clone()
                    latent_model_input = self.scheduler.add_noise(latent_model_input, noise, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        return_dict=False,
                    )[0]

                    # compute the previous noisy sample x_t -> x_t-1
                    denoising_results.append(noise_pred)
                    # print(f"{timesteps[i]} timestep denoising: {torch.sum(latents)}")

                    if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % self.scheduler.order == 0):
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
                    for p in self.unet.parameters():
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
                        for p in self.unet.parameters():
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

        if not output_type == "latent":
            with torch.no_grad():
                image = self.vae.decode(latents)[0]
        else:
            image = latents

        if not return_dict:
            return (image, gsa_features)
        

        return GSAStableDiffusionPipelineOutput(images=image, gsa_features=gsa_features)