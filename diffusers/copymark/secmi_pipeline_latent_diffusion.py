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
from .secmi_scheduling_ddim import SecMIDDIMScheduler


class SecMILatentDiffusionPipeline(
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

        return SecMILatentDiffusionPipeline(unet=unet, vae=vae, scheduler=scheduler, device=device, generator=generator)

    @torch.no_grad()
    def prepare_inputs(self, batch, weight_dtype, device):
        pixel_values = batch["pixel_values"].to(weight_dtype)
        if device == 'cuda':
            pixel_values  = pixel_values.cuda()
        latents = self.vae.encode(pixel_values)[0]
        encoder_hidden_states = None

        return latents, encoder_hidden_states, None


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
        strength=0.2,
        prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 7.5,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        
        device = self.execution_device
        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # 6.1 Add image embeds for IP-Adapter
        # 6.2 Optionally get Guidance Scale Embedding

        # get [x_201, x_181, ..., x_1]
        # print(timesteps)
        posterior_results = []
        original_latents = latents.detach().clone()
        for i, t in enumerate(timesteps): # from t_max to t_min
            noise = randn_tensor(original_latents.shape, generator=generator, device=device, dtype=original_latents.dtype)
            posterior_latents = self.scheduler.add_noise(original_latents, noise, t)
            posterior_results.append(posterior_latents.detach().clone())
            # print(f"{t} timestep posterior: {torch.sum(posterior_latents)}")

        # get [x_(201+20), x_(181+20), ..., x_(1+20)]
        reverse_results = []
        for i, t in enumerate(timesteps):  # from t_max to t_min
            # predict the noise residual
            latent_model_input = posterior_results[i]
            noise_pred = self.unet(
                latent_model_input,
                t,
                return_dict=False,
            )[0]
            # compute the previous noisy sample x_t -> x_t-1
            reverse_latents = self.scheduler.reverse_step(noise_pred, t, latent_model_input, return_dict=False)[0]
            reverse_results.append(reverse_latents.detach().clone())
            
        # 7. Denoising loop
        denoising_results = []
        unit_t = timesteps[0] - timesteps[1]
        print(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latents = reverse_results[i]
                t = t + unit_t
                # expand the latents if we are doing classifier free guidance
                # latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        return_dict=False,
                    )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                denoising_results.append(latents.detach().clone())
                # print(f"{timesteps[i]} timestep denoising: {torch.sum(latents)}")

                if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            with torch.no_grad():
                image = self.vae.decode(latents)[0]
        else:
            image = latents

        if not return_dict:
            return (image,)

        return SecMIStableDiffusionPipelineOutput(images=image, posterior_results=posterior_results, denoising_results=denoising_results)
