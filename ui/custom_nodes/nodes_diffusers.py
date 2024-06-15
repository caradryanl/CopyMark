import os
import sys
sys.path.append('custom_nodes/')
sys.path.append('.')
sys.path.append('..')
import hashlib
import folder_paths

from diffusers_ui.encode_diffusers import vae_encode_sd, vae_encode_sdxl, text_encode_sd, text_encode_sdxl
from diffusers_ui.load_diffusers import load_diffusers, load_image

import torch
torch.inference_mode(False)

class DiffusersLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        paths = []
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                for root, subdir, files in os.walk(search_path, followlinks=True):
                    if "model_index.json" in files:
                        paths.append(os.path.relpath(root, start=search_path))

        return {"required": {"model_path": (paths,), }}
    
    RETURN_TYPES = ("unet", "text_encoder", "text_encoder_2", "vae", "tokenizer", "tokenizer_2", "scheduler")
    # RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "diffusers/load"

    def load_checkpoint(self, model_path):
        for search_path in folder_paths.get_folder_paths("diffusers"):
            if os.path.exists(search_path):
                path = os.path.join(search_path, model_path)
                if os.path.exists(path):
                    model_path = path
                    break
        return_types = ("unet", "text_encoder", "text_encoder_2", "vae", "tokenizer", "tokenizer_2", "scheduler")
        # return_types = ("unet",)
        return load_diffusers(model_path, return_types=return_types)
    

class DiffusersVAEEncodeSD:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ), "vae": ("vae", )}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "diffusers/encode"

    def encode(self, vae, pixels):
        return vae_encode_sd(vae, pixels)

class DiffusersTextEncodeSD:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}), 
                             "text_encoder": ("text_encoder", ),
                             "tokenizer": ("tokenizer", ),
                             }}
    RETURN_TYPES = ("CONDITIONING", "ADDED_COND_KWARGS")
    FUNCTION = "encode_prompt"
    CATEGORY = "diffusers/encode"

    def encode_prompt(
            self,
            prompt: str,
            text_encoder, 
            tokenizer, 
        ):
        return text_encode_sd(prompt, text_encoder, tokenizer)

class DiffusersVAEEncodeSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ), "vae": ("vae", )}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "diffusers/encode"

    def encode(self, vae, pixels):
        return vae_encode_sdxl(vae, pixels)



class DiffusersTextEncodeSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}), 
                             "text_encoder": ("text_encoder", ),
                             "text_encoder_2": ("text_encoder_2", ),
                             "tokenizer": ("tokenizer", ),
                             "tokenizer_2": ("tokenizer_2", ),
                             "unet": ("unet",),
                             "vae": ("vae",)
                             }}
    RETURN_TYPES = ("CONDITIONING", "ADDED_COND_KWARGS")
    FUNCTION = "encode_prompt"
    CATEGORY = "diffusers/encode"

    def encode_prompt(
            self,
            prompt: str,
            text_encoder, 
            text_encoder_2, 
            tokenizer, 
            tokenizer_2,
            unet,
            vae,
        ):
        return text_encode_sdxl(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, vae)
    
    
class DiffusersLoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "diffusers/load"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image):
        return load_image(image)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True
    
NODE_CLASS_MAPPINGS = {
    "DiffusersLoadModel": DiffusersLoadModel,
    "DiffusersVAEEncodeSD": DiffusersVAEEncodeSD,
    "DiffusersTextEncodeSD": DiffusersTextEncodeSD,
    "DiffusersLoadImage": DiffusersLoadImage,
}