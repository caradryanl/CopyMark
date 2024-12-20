import os
import sys
sys.path.append('custom_nodes/')
sys.path.append('.')
sys.path.append('..')
import hashlib
import folder_paths

from diffusers_ui.copymark import run_gsa, run_secmi, run_pia, run_pfami, pfami_vae_encode_sd
from diffusers_ui.load_diffusers import load_metadata

import torch
torch.inference_mode(False)

class GSA:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"unet": ("unet",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "prompts": ("CONDITIONING", ),
                    "added_cond_kwargs": ("ADDED_COND_KWARGS", ),
                    "latents": ("LATENT", ),
                    "scheduler": ("scheduler", ),
                    "gsa_mode": ("INT", {"default": 1, "min": 1, "max": 2}),
                    "metadata": ("METADATA",),
                     }
                }

    RETURN_TYPES = ("BOOL", "IMAGE")
    FUNCTION = "gsa"

    CATEGORY = "diffusers/copymark"
    
    def gsa(self, unet, seed, prompts, added_cond_kwargs, latents, scheduler, gsa_mode, metadata, num_inference_steps=5, strength=1.0):
        return run_gsa(unet, seed, prompts, added_cond_kwargs, latents, scheduler, gsa_mode, metadata, num_inference_steps, strength)

class SecMI:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"unet": ("unet",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "prompts": ("CONDITIONING", ),
                    "added_cond_kwargs": ("ADDED_COND_KWARGS", ),
                    "latents": ("LATENT", ),
                    "scheduler": ("scheduler", ),
                    "metadata": ("METADATA",),
                     }
                }

    RETURN_TYPES = ("BOOL", "IMAGE")
    FUNCTION = "secmi"

    CATEGORY = "diffusers/copymark"
    
    def secmi(self, unet, seed, prompts, added_cond_kwargs, latents, scheduler, metadata, num_inference_steps=100, strength=0.2):
        return run_secmi(unet, seed, prompts, added_cond_kwargs, latents, scheduler, metadata, num_inference_steps, strength)
    

class PIA:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"unet": ("unet",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "prompts": ("CONDITIONING", ),
                    "added_cond_kwargs": ("ADDED_COND_KWARGS", ),
                    "latents": ("LATENT", ),
                    "scheduler": ("scheduler", ),
                    "metadata": ("METADATA",),
                     }
                }

    RETURN_TYPES = ("BOOL", "IMAGE")
    FUNCTION = "pia"

    CATEGORY = "diffusers/copymark"
    
    def pia(self, unet, seed, prompts, added_cond_kwargs, latents, scheduler, metadata, num_inference_steps=100, strength=0.5):
        return run_pia(unet, seed, prompts, added_cond_kwargs, latents, scheduler, metadata, num_inference_steps, strength)
    
class PFAMIDiffusersVAEEncodeSD:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ), "vae": ("vae", )}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "diffusers/copymark"

    def encode(self, pixels, vae):
        return pfami_vae_encode_sd(vae, pixels)

class PFAMI:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"unet": ("unet",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "prompts": ("CONDITIONING", ),
                    "added_cond_kwargs": ("ADDED_COND_KWARGS", ),
                    "latents": ("LATENT", ),
                    "scheduler": ("scheduler", ),
                    "metadata": ("METADATA",),
                     }
                }

    RETURN_TYPES = ("BOOL", "IMAGE")
    FUNCTION = "pfami"

    CATEGORY = "diffusers/copymark"
    
    def pfami(self, unet, seed, prompts, added_cond_kwargs, latents, scheduler, metadata, num_inference_steps=100, attack_timesteps = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]):
        return run_pfami(unet, seed, prompts, added_cond_kwargs, latents, scheduler, metadata, num_inference_steps, attack_timesteps)

class LoadMetaData:
    @classmethod
    def INPUT_TYPES(s):
        base_path = os.path.dirname(os.path.realpath(__file__))
        input_dir = os.path.join(base_path, "assets/")
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"metadata_path": (sorted(files), {"metadata_path_upload": True})},
                }

    CATEGORY = "diffusers/copymark"

    RETURN_TYPES = ("METADATA",)
    FUNCTION = "load_metadata"
    def load_metadata(self, metadata_path):
        base_path = os.path.dirname(os.path.realpath(__file__))
        input_dir = os.path.join(base_path, "assets/")
        return load_metadata(os.path.join(input_dir, metadata_path))

    @classmethod
    def IS_CHANGED(s, metadata_path):
        image_path = folder_paths.get_annotated_filepath(metadata_path)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, metadata_path):
        return True
    
NODE_CLASS_MAPPINGS = {
    "GSA": GSA,
    "SecMI": SecMI,
    "PIA": PIA,
    "PFAMIDiffusersVAEEncodeSD": PFAMIDiffusersVAEEncodeSD,
    "PFAMI": PFAMI,
    "LoadMetaData": LoadMetaData,
}