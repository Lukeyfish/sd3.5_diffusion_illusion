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

from sd3_helpers import flip_latent_upside_down, merge_denoised_outputs


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
    x_a, # Init a
    x_b, # Init b
    conditioning_a, 
    conditioning_b, 
    sigmas,
    method,
    method_param, 
    extra_args=None,
    SD3Inferencer=None,
    SD3LatentFormat=None):
    """DPM-Solver++(2M)."""

    ### UPDATE FOR IF DOING INIT_IMAGES ###
    # Combines init latents for first iteration
    if method == "mean":
        x = ((1 - method_param) * x_a) + (method_param * flip_latent_upside_down(x_b))
    else:
        x = torch.stack([x_a, flip_latent_upside_down(x_b)]).mean(0)

    latents_history = []

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in tqdm(range(len(sigmas) - 1)):
        latents_history.append(x.detach().clone())

        # First orientation with conditioning_a
        extra_args["cond"] = conditioning_a
        denoised_a = model(x, sigmas[i] * s_in, **extra_args)
    
        # Second orientation with conditioning_b
        extra_args["cond"] = conditioning_b
        x_temp = flip_latent_upside_down(x)
        denoised_b = model(x_temp, sigmas[i] * s_in, **extra_args)
        denoised_b = flip_latent_upside_down(denoised_b)

        ''' 
        # Flip latent in pixel space to merge with denoised_b
        latent = SD3LatentFormat().process_out(x)
        image = SD3Inferencer.vae_decode(latent)
        image = image.rotate(180)
        latent = SD3Inferencer.vae_encode(image)
        latent = SD3LatentFormat().process_in(latent).to("cuda")
        #x_temp = torch.stack([x_temp, latent]).mean(0)
        #print(f"pixel_flipped_sigma{1 - sigmas[i]}, latent_flipped_sigma{sigmas[i]}")
        #x_temp = ((sigmas[i]) * x_temp) + ((1- sigmas[i]) * latent)

        '''


        # sets method _param to iteration
        if method == "alternate":
            method_param = i

        denoised = merge_denoised_outputs(
            denoised_a=denoised_a,
            denoised_b=denoised_b,
            method=method,
            method_param=method_param
            )

        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d

        old_denoised = denoised
    return x, latents_history