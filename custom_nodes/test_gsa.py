from diffusers_ui.encode_diffusers import vae_encode_sd, text_encode_sd
from diffusers_ui.gsa import gsa
from diffusers_ui.load_diffusers import load_image, load_diffusers, load_metadata

def test():
    # 0. initialize
    image = '../diffusers/datasets/laion-aesthetic-2-5k/images/0.jpg'
    metadata_path = 'assets/metadata_gsa_1_sd.pkl'
    model_path = '../models/diffusers/stable-diffusion-v1-5/'
    prompt = ""

    # 1. load image
    img, _ = load_image(image)
    metadata = load_metadata(metadata_path)[0]
    # print(type(img), type(mask), img.max(), img.min())

    # 2. load model
    return_types = ("unet", "text_encoder", "text_encoder_2", "vae", "tokenizer", "tokenizer_2", "scheduler")
    unet, text_encoder, _, vae, tokenizer, _, scheduler = load_diffusers(model_path, return_types)


    # 3. encode
    latents = vae_encode_sd(vae, img)[0]
    prompt_embeds, added_cond_kwargs = text_encode_sd(prompt, text_encoder, tokenizer)

    flag, feat_map = gsa(unet, 1, prompt_embeds, added_cond_kwargs, latents, scheduler, 2, metadata)
    print(flag)
    feat_map.show()


    

if __name__ == '__main__':
    test()