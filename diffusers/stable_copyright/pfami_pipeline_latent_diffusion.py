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

class PFAMILatentDiffusionPipeline(
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

        return PFAMILatentDiffusionPipeline(unet=unet, vae=vae, scheduler=scheduler, device=device, generator=generator)

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
        attack_timesteps: List[int] = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450],
        normalized: bool=False,
        prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 7.5,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        
        device = self.execution_device
        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        attack_timesteps = [torch.tensor(attack_timestep).to(device=device) for attack_timestep in attack_timesteps]
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # 6.1 Add image embeds for IP-Adapter
        # 6.2 Optionally get Guidance Scale Embedding

        # get the intermediate at t in attack_timesteps [x_201, x_181, ..., x_1]
        # print(timesteps)
        original_latents = latents.detach().clone()
        posterior_results = []
        for i, t in enumerate(attack_timesteps): # from t_max to t_min
            noise = randn_tensor(original_latents.shape, generator=generator, device=device, dtype=original_latents.dtype)
            posterior_results.append(noise)
            # print(f"{t} timestep posterior: {torch.sum(posterior_latents)}")
            

        # 7. Denoising loop
        # num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        denoising_results = []
        unit_t = attack_timesteps[1] - attack_timesteps[0]
        with self.progress_bar(total=len(attack_timesteps)) as progress_bar:
            for i, t in enumerate(attack_timesteps):
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

                # no classifier free guidance
                # # perform guidance
                # if self.do_classifier_free_guidance:
                #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                #     noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                #     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                denoising_results.append(noise_pred.detach().clone())
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
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