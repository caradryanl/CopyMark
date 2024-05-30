import os
import sys
sys.path.append('custom_nodes/')
sys.path.append('.')
sys.path.append('..')
import hashlib
import folder_paths


from diffusers_ui.gsa import run_gsa
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

    CATEGORY = "diffusers/gsa"
    
    def gsa(self, unet, seed, prompts, added_cond_kwargs, latents, scheduler, gsa_mode, metadata, num_inference_steps=5, strength=1.0):
        return run_gsa(unet, seed, prompts, added_cond_kwargs, latents, scheduler, gsa_mode, metadata, num_inference_steps, strength)


class GSALoadMetaData:
    @classmethod
    def INPUT_TYPES(s):
        base_path = os.path.dirname(os.path.realpath(__file__))
        input_dir = os.path.join(base_path, "assets/")
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"metadata_path": (sorted(files), {"metadata_path_upload": True})},
                }

    CATEGORY = "diffusers/gsa"

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
    "GSALoadMetaData": GSALoadMetaData,
}