import numpy as np
import torch
from tqdm import tqdm
from accelerate import Accelerator
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import randn_tensor
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import io
from PIL import Image
from torch.autograd import Variable

torch.inference_mode(False)

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

def infer(x_input, metadata):
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

def run_gsa(unet, seed, prompts, added_cond_kwargs, latents, scheduler, gsa_mode, metadata, num_inference_steps=5, strength=1.0):
    
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
    is_member, feat_map = infer(x_input=gsa_features, metadata=metadata)

    return is_member, feat_map