### Impls of the SD3 core diffusion model and VAE

import math
import re

import einops
from safetensors import safe_open
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn as nn

from dit_embedder import ControlNetEmbedder
from sd3_mmditx_illusions import MMDiTX
from typing import Tuple

import matplotlib.pyplot as plt

from sd3_helpers import flip_latent, merge_denoised_outputs, flip_latent_with_blend, rotate_latent_tiles


#################################################################################################
### Samplers
#################################################################################################


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16)
def sample_euler(model, x, sigmas, extra_args=None):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in tqdm(range(len(sigmas) - 1)):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x

@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16)
def sample_dpmpp_2m(
    model,
    x_a,
    x_b,
    conditioning_a,
    conditioning_b,
    sigmas,
    method,
    method_param,
    illusion_type,
    extra_args=None,
    SD3Inferencer=None,
    SD3LatentFormat=None):
    """
    DPM-Solver++(2M) that creates true orientation-dependent illusions.
    The key is to make the same latent space naturally decode to different 
    interpretations based on orientation, rather than averaging two separate images.
    
    Args:
        model: The diffusion model
        noise: Initial noise latent
        conditioning_a: Conditioning for primary orientation
        conditioning_b: Conditioning for flipped orientation
        sigmas: Noise schedule
        flip_fn: Function to flip latents
        guidance_scale: Classifier-free guidance scale
        illusion_strength: How strongly to enforce orientation-dependent features
        extra_args: Additional arguments for the model
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x_a.new_ones([x_a.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    
    x = x_a.clone()
    old_denoised = None
    latent_history = []

    for i in tqdm(range(len(sigmas) - 1)):
        
        # Get features for both orientations
        extra_args["cond"] = conditioning_a
        features_a = model(x, sigmas[i] * s_in, **extra_args)
        latent_history.append(features_a.detach().clone())
        
        extra_args["cond"] = conditioning_b
        #x_flipped = flip_latent(x, illusion_type) # Flip latent
        x_flipped = rotate_latent_tiles(x,inverse=False,num_divisions=4)
        features_b = model(x_flipped, sigmas[i] * s_in, **extra_args)

        latent_history.append(features_b.detach().clone())
        features_b = rotate_latent_tiles(features_b,inverse=True,num_divisions=4)
        #features_b = flip_latent(features_b, -illusion_type)  # Flip latent back


        if method == "alternate":
            method_param == i

        denoised = merge_denoised_outputs(
            denoised_a=features_a, 
            denoised_b=features_b,
            method=method, 
            method_param=method_param
            )
        
        # Calculate timesteps
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t

        # Update latents
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        
        old_denoised = denoised
    
    return x, latent_history


