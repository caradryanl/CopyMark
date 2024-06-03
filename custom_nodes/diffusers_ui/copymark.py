from typing import Optional, Tuple, Union
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms

from accelerate import Accelerator
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import randn_tensor
from diffusers.schedulers.scheduling_ddim import \
    (
        DDIMScheduler,
        DDIMSchedulerOutput,
        randn_tensor
    )

torch.inference_mode(False)



# add one function for DDIM
class SecMIDDIMScheduler(DDIMScheduler):

    # listed here just for display
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def reverse_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        '''
            The only difference is that the prev_timestep is exactly the next timestep, because
            we are to use x_t to get x_t+1 with ddim
        '''

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. get previous step value (=t-1)
        # prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        # ==================================================
        #       where we have modifications: - -> +
        # ==================================================
        prev_timestep = timestep + self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5) if eta != 0 else 0

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

def progress_bar(iterable=None, total=None):
    if iterable is not None:
        return tqdm(iterable)
    elif total is not None:
        return tqdm(total=total)
    else:
        raise ValueError("Either `total` or `iterable` has to be defined.")
    
def get_timesteps(num_inference_steps, strength, device):
    timesteps = (
                np.linspace(0, 1000 - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
    timesteps = torch.from_numpy(timesteps).to(device)

    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = timesteps[t_start :]  # order=1
    # [601, 581, ..., 21, 1]
    return timesteps, num_inference_steps - t_start

def normalize(array, min=None, max=None):
    eps = 1e-5
    if min is not None and max is not None:
        min_val = min
        max_val = max
    else:
        min_val = np.min(array)
        max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val + eps)
    normalized_array = (normalized_array - 0.5) / 0.5
    return normalized_array, min_val, max_val

def infer_gsa(x_input, metadata):
    x_input = x_input.detach().clone().cpu().numpy()

    x, y, x_min, x_max, model = \
        metadata['x'], metadata['y'], metadata['x_min'], \
            metadata['x_max'], metadata['model']
    
    # preprocess
    x_max, x_min, x_avg = x[np.isfinite(x)].max(), x[np.isfinite(x)].min(), x[np.isfinite(x)].mean()
    x = np.nan_to_num(x, nan=x_avg, posinf=x_max, neginf=x_min) # deal with exploding gradients

    x_input, _, _ = normalize(x_input, x_min, x_max)
    y_input = model.predict(x_input)

    is_member = bool(1 - y_input.item())

    # print(y.shape, x.shape, y_input.shape, x_input.shape)
    y_all = np.concatenate((y, y_input+2), axis=0)
    x_all = np.concatenate((x, x_input), axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(x_all)
    # pca = PCA(n_components=2, random_state=1)
    # x_pca = pca.fit_transform(x_all)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_all, cmap='viridis')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Member', 'Non-Member', 'Input'])
    plt.title("Visualization of Member and Non-member Data Manifold")
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    # plt.show()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    feat_map = Image.open(img_buf)
    feat_map = np.array(feat_map).astype(np.float32) / 255.0
    feat_map = torch.from_numpy(feat_map)[None,]
    # img_buf.close()

    return is_member, feat_map

def run_gsa(unet, 
            seed, 
            prompts, 
            added_cond_kwargs, 
            latents, 
            scheduler, 
            gsa_mode, 
            metadata, 
            num_inference_steps=5, 
            strength=1.0):
    accelerator = Accelerator()
    for param in unet.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=0
    )

    unet, optimizer = accelerator.prepare(
        unet, optimizer
    )
    device = accelerator.device
    timesteps, num_inference_steps = get_timesteps(num_inference_steps, strength, device)

    # get [x_201, x_181, ..., x_1]
    # print(timesteps)
    posterior_results = []
    original_latents = latents.detach().clone()
    for i, t in enumerate(timesteps): # from t_max to t_min
        noise = randn_tensor(original_latents.shape, device=device, dtype=original_latents.dtype)
        posterior_results.append(noise)
        
    # 7. Denoising loop
    denoising_results = []
    gsa_features = []
    with accelerator.accumulate(unet):
        with progress_bar(total=num_inference_steps) as bar:
            for i, t in enumerate(timesteps):
                noise = posterior_results[i]
                # expand the latents if we are doing classifier free guidance
                # latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = original_latents.detach().clone()
                latent_model_input = scheduler.add_noise(latent_model_input, noise, t)
                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompts,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # no classifier free guidance
            
                # compute the previous noisy sample x_t -> x_t-1
                denoising_results.append(noise_pred)
                # latents = scheduler.step(noise_pred, t, latent_model_input, return_dict=False)[0]

                bar.update()

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
                accelerator.backward(losses_i)

                # compute the gradient of the loss sum
                grads_i = []
                for p in unet.parameters():
                    grads_i.append(torch.norm(p.grad).detach().clone())
                grads_i = torch.stack(grads_i, dim=0)   # [num_p]
                gsa_features.append(grads_i)
                optimizer.zero_grad()

            gsa_features = torch.stack(gsa_features, dim=0) # [bsz, num_p]
        elif gsa_mode == 2:
            for i in range(original_latents.shape[0]):
                grads_i = []
                for j in range(len(denoising_results)):
                    # compute the loss
                    loss_ij = (((denoising_results[j][i, ...] - posterior_results[j][i, ...]) ** 2).sum())
                    loss_ij = Variable(loss_ij, requires_grad=True)
                    accelerator.backward(loss_ij, retain_graph=True)

                    # compute the gradient
                    grads_ij = []
                    for p in unet.parameters():
                        grads_ij.append(torch.norm(p.grad).detach().clone())
                    grads_ij = torch.stack(grads_ij, dim=0) # [num_p]

                    grads_i.append(grads_ij)   
                    optimizer.zero_grad()

                # compute the sum of gradients
                grads_i = torch.stack(grads_i, dim=0).sum(dim=0)   # [timestep, num_p] -> [num_p]
                gsa_features.append(grads_i)
            gsa_features = torch.stack(gsa_features, dim=0) # [bsz, num_p]
        else:
            raise NotImplementedError(f"Mode {gsa_mode} out of 1 and 2")


    # Offload all models
    is_member, feat_map = infer_gsa(x_input=gsa_features, metadata=metadata)

    return is_member, feat_map

def infer(scores, metadata):
    

    x_mem, x_nonmem, threshold = \
        metadata['member_scores'], metadata['nonmember_scores'], metadata['threshold']    
    
    x = np.array(scores).repeat(x_nonmem.shape[0])
    print(x_mem.shape, x_nonmem.shape, x.shape)
    data = pd.DataFrame({
        'member': x_mem,
        'non-member': x_nonmem,
        'input': x
    })

    # Melt the DataFrame to long format
    data_melted = data.melt(var_name='Membership', value_name='Scores')

    # Create the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Membership', y='Scores', data=data_melted)

    # Add a horizontal threshold line
    plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')

    # Add titles and labels
    plt.title('Three Columns Box Plot with Threshold')
    plt.xlabel('Membership')
    plt.ylabel('Scores')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    feat_map = Image.open(img_buf)
    feat_map = np.array(feat_map).astype(np.float32) / 255.0
    feat_map = torch.from_numpy(feat_map)[None,]
    # img_buf.close()

    is_member = x[0] <= threshold

    return is_member, feat_map

def run_secmi(
        unet, 
        seed, 
        prompts, 
        added_cond_kwargs, 
        latents, 
        scheduler, 
        metadata, 
        num_inference_steps=100, 
        strength=0.2):
    device = "cuda"
    generator = None
    scheduler = SecMIDDIMScheduler.from_config(scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device)
    timesteps, num_inference_steps = get_timesteps(num_inference_steps, strength, device)
    # print(timesteps)

    unet = unet.to(device)

    posterior_results = []
    original_latents = latents.detach().clone()
    for i, t in enumerate(timesteps): # from t_max to t_min
        noise = randn_tensor(original_latents.shape, generator=generator, device=device, dtype=original_latents.dtype)
        posterior_latents = scheduler.scale_model_input(original_latents, t)
        posterior_latents = scheduler.add_noise(posterior_latents, noise, t)
        posterior_results.append(posterior_latents.detach().clone())

    # get [x_(201+20), x_(181+20), ..., x_(1+20)]
    reverse_results = []
    for i, t in enumerate(timesteps):  # from t_max to t_min
        latent_model_input = posterior_results[i]
        # predict the noise residual
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompts,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        # compute the previous noisy sample x_t -> x_t-1
        reverse_latents = scheduler.reverse_step(noise_pred, t, latent_model_input, return_dict=False)[0]
        reverse_results.append(reverse_latents.detach().clone())
        
    # 7. Denoising loop
    denoising_results = []
    unit_t = timesteps[0] - timesteps[1]
    with progress_bar(total=num_inference_steps) as bar:
        for i, t in enumerate(timesteps):
            latents = reverse_results[i]
            t = t + unit_t
            latent_model_input = latents

            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompts,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            denoising_results.append(latents.detach().clone())

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) % scheduler.order == 0):
                bar.update()

    # compute score
    scores = [((denoising_results[14] - posterior_results[14]) ** 2).sum().item()]
    # Offload all models
    is_member, feat_map = infer(scores=scores, metadata=metadata)

    return is_member, feat_map

def run_pia(
        unet, 
        seed, 
        prompts, 
        added_cond_kwargs, 
        latents, 
        scheduler, 
        metadata, 
        num_inference_steps=100, 
        strength=0.2):
    device = "cuda"
    generator = None
    scheduler = SecMIDDIMScheduler.from_config(scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device)
    timesteps, num_inference_steps = get_timesteps(num_inference_steps, strength, device)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = get_timesteps(num_inference_steps, strength, device)

    # get [x_201, x_181, ..., x_1]
    # print(timesteps)
    original_latents = latents.detach().clone()
    posterior_results = []
    noise_pred_gt = randn_tensor(original_latents.shape, generator=generator, device=device, dtype=original_latents.dtype)
    # print(f"timestep: {0}, sum: {noise_pred_gt.sum()} {noise_pred_gt[0, 0, :10, 0]}")
    for i, t in enumerate(timesteps): # from t_max to t_min
        posterior_results.append(noise_pred_gt.detach().clone())
        # print(f"{t} timestep posterior: {torch.sum(posterior_latents)}")
        
    # 7. Denoising loop
    denoising_results = []
    unit_t = timesteps[0] - timesteps[1]
    # print(unit_t, timesteps)
    with progress_bar(total=num_inference_steps) as bar:
        for i, t in enumerate(timesteps):
            noise = posterior_results[i]
            t = t + unit_t
            # expand the latents if we are doing classifier free guidance
            # latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = original_latents.detach().clone()
            latent_model_input = scheduler.add_noise(latent_model_input, noise, t)

            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompts,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        # compute the previous noisy sample x_t -> x_t-1
        denoising_results.append(noise_pred.detach().clone())

        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) % scheduler.order == 0):
            bar.update()

    # compute score
    for i in range(len(denoising_results)):
        scores.append(((denoising_results[i] - posterior_results[i]) ** 2).sum())
    scores = torch.stack(scores, dim=0).sum().item() # torch.Size([50])
    scores = [scores]
    # Offload all models
    is_member, feat_map = infer(scores=scores, metadata=metadata)

    return is_member, feat_map

def image_perturbation(image, strength, image_size=512):
    perturbation = transforms.Compose([
        transforms.CenterCrop(size=int(image_size * strength)),
        transforms.Resize(size=image_size, antialias=True),
    ])
    return perturbation(image)

def pfami_vae_encode_sd(vae, pixels, strengths = np.linspace(0.95, 0.7, 10)):
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
    original_pixel_values = pixel_values.detach().clone()
    pixel_values_list = [original_pixel_values]
    
    for strength in strengths:
        pixel_values = original_pixel_values.detach().clone()
        pixel_values = image_perturbation(pixel_values, strength)
        pixel_values_list.append(pixel_values)
    pixel_values = torch.stack(pixel_values_list, dim=0)

    vae = vae.to(device)
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = latents * 0.18215 # [11, C, H, W]
    vae = vae.to('cpu')

    return (latents.detach().clone(), )

def run_pfami(
        unet, 
        seed, 
        prompts, 
        added_cond_kwargs, 
        latents, 
        scheduler, 
        metadata, 
        num_inference_steps=100, 
        attack_timesteps=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450]):
    device = "cuda"
    generator = None
    scheduler = SecMIDDIMScheduler.from_config(scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device)
    attack_timesteps = [torch.tensor(attack_timestep).to(device=device) for attack_timestep in attack_timesteps]

    # get [x_201, x_181, ..., x_1]
    # print(timesteps)
    perturb_losses = []
    for idx, latents_i in enumerate(latents):
        original_latents = latents_i.detach().clone()
        posterior_results = []
        for i, t in enumerate(attack_timesteps): # from t_max to t_min
            noise = randn_tensor(original_latents.shape, generator=generator, device=device, dtype=original_latents.dtype)
            posterior_results.append(noise)
            # print(f"{t} timestep posterior: {torch.sum(posterior_latents)}")
            
        # 7. Denoising loop
        denoising_results = []
        unit_t = attack_timesteps[0] - attack_timesteps[1]
        # print(unit_t, timesteps)
        with progress_bar(total=num_inference_steps) as bar:
            for i, t in enumerate(attack_timesteps):
                noise = posterior_results[i]
                t = t + unit_t
                # expand the latents if we are doing classifier free guidance
                # latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = original_latents.detach().clone()
                latent_model_input = scheduler.add_noise(latent_model_input, noise, t)

                # predict the noise residual
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompts,
                    timestep_cond=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                denoising_results.append(noise_pred.detach().clone())

            # call the callback, if provided
            if i == len(attack_timesteps) - 1 or ((i + 1) % scheduler.order == 0):
                bar.update()

        # [len(attack_timesteps) x [B, 4, 64, 64]]
        perturb_losses_strength = []
        for i in range(len(posterior_results)):
            losses_batch = []
            for idx in range(posterior_results[0].shape[0]):
                loss = ((posterior_results[i][idx, ...] - denoising_results[i][idx, ...]) ** 2).sum()
                losses_batch.append(loss)
            losses_batch = torch.stack(losses_batch, dim=0).reshape(-1)
            perturb_losses_strength.append(losses_batch)
        perturb_losses_strength = torch.stack(perturb_losses_strength, dim=0) # [T, 1]
    perturb_losses.append(perturb_losses_strength)

    ori_losses = perturb_losses[0]
    perturb_losses = torch.stack(perturb_losses[1:], dim=0) # [M, T, 1]

    # compute the probability flunctuation delta_prob
    eps = 1e-6
    ori_losses = ori_losses.unsqueeze(dim=0).repeat(len(perturb_losses),1, 1) # [M, T, 1]
    delta_prob = (ori_losses - perturb_losses) / (ori_losses + eps) # [M, T, 1]
    delta_prob = delta_prob.mean(dim=0) # [T, 1]
    delta_prob_sum = delta_prob.sum().item()   # []

    # compute score
    scores = [delta_prob_sum]
    # Offload all models
    is_member, feat_map = infer(scores=scores, metadata=metadata)

    return is_member, feat_map