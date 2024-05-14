import os
import torch
import numpy as np
import PIL.Image

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from diffusers import UNet2DModel, DDIMScheduler, VQModel

from diffusers.pipelines.stable_diffusion.pipeline_output import BaseOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import \
    (
        StableDiffusionInpaintPipeline,
        PipelineImageInput,
        deprecate,
        retrieve_timesteps,
        randn_tensor,
        VaeImageProcessor,
        DiffusionPipeline
    )


from .secmi_pipeline_stable_diffusion import SecMIStableDiffusionPipelineOutput
from .secmi_scheduling_ddim import SecMIDDIMScheduler

class DRCLatentDiffusionPipeline(
    DiffusionPipeline
):
    def __init__(self, unet, vae, scheduler, device, generator):

        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.generator = generator
        self.execution_device = device

        self.vae_scale_factor = 2 ** (len(self.vae.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        
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

        return DRCLatentDiffusionPipeline(unet=unet, vae=vae, scheduler=scheduler, device=device, generator=generator)

    @torch.no_grad()
    def prepare_inputs(self, batch, weight_dtype, device):
        pixel_values = batch["pixel_values"].to(weight_dtype)
        if device == 'cuda':
            pixel_values  = pixel_values.cuda()
        latents = pixel_values
        encoder_hidden_states = None

        masks = []
        for mask in batch["mask"]:
            masks.append(torch.tensor(mask))
        masks = torch.stack(masks, dim=0).cuda()

        return latents, encoder_hidden_states, masks

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
    
    def prepare_latents(
        self,
        image,
        dtype,
        device,
        generator,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        if return_image_latents:
            image = image.to(device=device, dtype=dtype)
            image_latents = self.vae.encode(image)[0]

        noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=dtype)
        latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
        latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents

        outputs = (latents,)
        if return_noise:
            outputs += (noise,)
        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self.vae.encode(masked_image)[0]

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents


    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
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

        crops_coords = None
        resize_mode = "default"
        original_image = image.detach().clone()
        init_image = self.image_processor.preprocess(
            image, height=256, width=256, crops_coords=crops_coords, resize_mode=resize_mode
        )
        init_image = init_image.to(dtype=torch.float32)

        # 6. Prepare latent variables
        latents_outputs = self.prepare_latents(
            image,
            torch.float32,
            device,
            generator,
            is_strength_max=True,
            return_noise=True,
            return_image_latents=True,
        )
        latents, noise, image_latents = latents_outputs

        # 7. Prepare mask latent variables
        mask_condition = self.mask_processor.preprocess(
            mask_image, height=256, width=256, resize_mode=resize_mode, crops_coords=crops_coords
        )
        masked_image = init_image * (mask_condition < 0.5)


        mask_condition = mask_condition.to(dtype=masked_image.dtype)
        mask, _ = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            256,
            256,
            torch.float32,
            device,
            generator,
        )
            
        # 7. Denoising loop
        denoising_results = []
        unit_t = timesteps[0] - timesteps[1]
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        return_dict=False,
                    )[0]

                # compute the previous noisy sample x_t -> x_t-1
                denoising_results.append(noise_pred.detach().clone())
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                init_latents_proper = image_latents
                init_mask = mask
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents_proper, noise, torch.tensor([noise_timestep])
                    )
                latents = (1 - init_mask) * init_latents_proper + init_mask * latents
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

        return SecMIStableDiffusionPipelineOutput(images=image, posterior_results=None, denoising_results=None)