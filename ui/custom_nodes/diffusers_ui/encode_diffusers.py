import torch
from torchvision import transforms
torch.inference_mode(False)

def _get_add_time_ids(
        original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids

def vae_encode_sd(vae, pixels):
    weight_dtype = torch.float16
    device = "cuda"

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    pixels = transform(pixels[0].permute(2, 0, 1)).unsqueeze(dim=0)
    # print(pixels.shape)

    pixel_values = pixels.to(weight_dtype).cuda()
    vae = vae.to(device)
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = latents * 0.18215
    vae = vae.to('cpu')

    return (latents.detach().clone(), )

def vae_encode_sdxl(vae, pixels):
    weight_dtype = torch.float16
    device = "cuda"

    pixel_values = pixels.to(weight_dtype)
    pixel_values = pixel_values.cuda()

    # print(pixel_values[0])
    pixel_values = pixel_values.float()
    vae = vae.to(device)
    vae.to(dtype=torch.float32)
    latents = vae.encode(pixel_values).latent_dist.sample()
    vae.to(weight_dtype)
    
    latents = latents.to(weight_dtype)
    latents = vae.config.scaling_factor * latents

    return (latents.detach().clone(), )

def text_encode_sd(prompt, text_encoder, tokenizer):
    device = "cuda"
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    text_encoder = text_encoder.to(device)
    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0]

    if text_encoder is not None:
        prompt_embeds_dtype = text_encoder.dtype
    else:
        prompt_embeds_dtype = prompt_embeds.dtype

    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)
    text_encoder = text_encoder.to('cpu')

    added_cond_kwargs = {}
    return prompt_embeds.detach().clone(), added_cond_kwargs

def text_encode_sdxl(prompt, text_encoder, text_encoder_2, 
                     tokenizer, tokenizer_2, unet, vae):
    device = "cuda"
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)

    prompt = [prompt] if isinstance(prompt, str) else prompt

    # Define tokenizers and text encoders
    tokenizers = [tokenizer, tokenizer_2] if tokenizer is not None else [tokenizer_2]
    text_encoders = (
        [text_encoder, text_encoder_2] if text_encoder is not None else [text_encoder_2]
    )
    prompt_2 = prompt_2 or prompt
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

    prompt_embeds_list = []
    prompts = [prompt, prompt_2]
    for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    if text_encoder_2 is not None:
        prompt_embeds = prompt_embeds.to(dtype=text_encoder_2.dtype, device=device)
    else:
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, 1, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(
        bs_embed * 1, -1
    )

    height = height or unet.config.sample_size * (2 ** (len(vae.config.block_out_channels) - 1))
    width = width or unet.config.sample_size * (2 ** (len(vae.config.block_out_channels) - 1))
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)
    crops_coords_top_left = (0, 0),

    add_time_ids = _get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
    )

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = pooled_prompt_embeds.to(device).detach().clone()
    add_time_ids = add_time_ids.to(device).repeat(batch_size * 1, 1).detach().clone()
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    return prompt_embeds.detach().clone(), added_cond_kwargs