import os
import torch
import numpy as np
import PIL.Image

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from diffusers import UNet2DModel, DDIMScheduler, VQModel

from diffusers.pipelines.stable_diffusion.pipeline_output import BaseOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import \
    (
        replace_example_docstring,
        EXAMPLE_DOC_STRING,
        PipelineImageInput,
        deprecate,
        retrieve_timesteps,
        randn_tensor,
        DiffusionPipeline
    )

from .secmi_pipeline_stable_diffusion import SecMIStableDiffusionPipelineOutput


class PIALatentDiffusionPipeline(
    DiffusionPipeline
):
    def __init__(self, unet, vae, scheduler, device, generator):
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.device = device
        self.generator = generator
        
    @classmethod
    def from_pretrained(self, 
                        pretrained_model_name_or_path: Union[str, os.PathLike]="CompVis/ldm-celebahq-256",
                        torch_dtype: torch.dtype=torch.float32,
                        ):
        unet = UNet2DModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", torch_dtype=torch_dtype)
        vae = VQModel.from_pretrained(pretrained_model_name_or_path, subfolder="vqvae", torch_dtype=torch_dtype)
        scheduler = DDIMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

        # cuda and seed
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seed = 2024
        unet.to(device)
        vae.to(device)

        # generate gaussian noise to be decoded
        generator = torch.manual_seed(seed)
        scheduler.set_timesteps(num_inference_steps=50)

        return PIALatentDiffusionPipeline(unet=unet, vae=vae, scheduler=scheduler, device=device, generator=generator)

    @torch.no_grad()
    def prepare_inputs(self, batch, weight_dtype, device):
        pixel_values, input_ids = batch["pixel_values"].to(weight_dtype), batch["input_ids"]
        if device == 'cuda':
            pixel_values, input_ids = pixel_values.cuda(), input_ids.cuda()

        latents = self.vae.encode(pixel_values)
        encoder_hidden_states = None

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

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        strength: float=0.2,
        normalized: bool=False,
        prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 7.5,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        
        device = self.device
        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # 6.1 Add image embeds for IP-Adapter
        # 6.2 Optionally get Guidance Scale Embedding

        # get [x_201, x_181, ..., x_1]
        # print(timesteps)
        original_latents = latents.detach().clone()
        posterior_results = []
        noise_pred_gt = randn_tensor(original_latents.shape, generator=generator, device=device, dtype=original_latents.dtype)
        # print(f"timestep: {0}, sum: {noise_pred_gt.sum()} {noise_pred_gt[0, 0, :10, 0]}")
        if normalized:
            noise_pred_gt = noise_pred_gt / noise_pred_gt.abs().mean(list(range(1, noise_pred_gt.ndim)), keepdim=True) * (2 / torch.pi) ** 0.5
        for i, t in enumerate(timesteps): # from t_max to t_min
            posterior_results.append(noise_pred_gt.detach().clone())
            # print(f"{t} timestep posterior: {torch.sum(posterior_latents)}")
            
        # 7. Denoising loop
        self._num_timesteps = len(timesteps)
        denoising_results = []
        unit_t = timesteps[0] - timesteps[1]
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                noise = posterior_results[i]
                t = t + unit_t
                # expand the latents if we are doing classifier free guidance
                # latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = original_latents.detach().clone()
                latent_model_input = self.scheduler.add_noise(latent_model_input, noise, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        return_dict=False,
                    )[0]

                # compute the previous noisy sample x_t -> x_t-1
                denoising_results.append(noise_pred.detach().clone())
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                # print(f"{timesteps[i]} timestep denoising: {torch.sum(latents)}")

        if not output_type == "latent":
            with torch.no_grad():
                image = self.vae.decode(image)
        else:
            image = latents

        if not return_dict:
            return (image,)

        return SecMIStableDiffusionPipelineOutput(images=image, posterior_results=posterior_results, denoising_results=denoising_results)