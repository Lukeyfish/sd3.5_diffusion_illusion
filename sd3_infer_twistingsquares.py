# NOTE: Must have folder `models` with the following files:
# - `clip_g.safetensors` (openclip bigG, same as SDXL)
# - `clip_l.safetensors` (OpenAI CLIP-L, same as SDXL)
# - `t5xxl.safetensors` (google T5-v1.1-XXL)
# - `sd3_medium.safetensors` (or whichever main MMDiT model file)
# Also can have
# - `sd3_vae.safetensors` (holds the VAE separately if needed)

import datetime
import math
import os
import pickle
import re
import clip

import fire
import numpy as np
import sd3_impls_twistingsquares
import torch
from other_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
from PIL import Image, ImageOps
from safetensors import safe_open
from sd3_impls_twistingsquares import (
    SDVAE,
    BaseModel,
    CFGDenoiser,
    SD3LatentFormat,
    SkipLayerCFGDenoiser,
)
from tqdm import tqdm

import torchvision.transforms as transforms

from helpers import calculate_clip_score, save_parameters_to_file, generate_filename, generate_outdir_name, sanitize_filename

#################################################################################################
### Wrappers for model parts
#################################################################################################


def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
                if obj.shape != tensor.shape:
                    print(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e


CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}


class ClipG:
    def __init__(self, model_folder: str, device: str = "cpu"):
        with safe_open(
            f"{model_folder}/clip_g.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device=device, dtype=torch.float32)
            load_into(f, self.model.transformer, "", device, torch.float32)


CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}


class ClipL:
    def __init__(self, model_folder: str):
        with safe_open(
            f"{model_folder}/clip_l.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device="cpu",
                dtype=torch.float32,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}


class T5XXL:
    def __init__(self, model_folder: str, device: str = "cpu", dtype=torch.float32):
        with safe_open(
            f"{model_folder}/t5xxl.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
            load_into(f, self.model.transformer, "", device, dtype)


CONTROLNET_MAP = {
    "time_text_embed.timestep_embedder.linear_1.bias": "t_embedder.mlp.0.bias",
    "time_text_embed.timestep_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
    "time_text_embed.timestep_embedder.linear_2.bias": "t_embedder.mlp.2.bias",
    "time_text_embed.timestep_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
    "pos_embed.proj.bias": "x_embedder.proj.bias",
    "pos_embed.proj.weight": "x_embedder.proj.weight",
    "time_text_embed.text_embedder.linear_1.bias": "y_embedder.mlp.0.bias",
    "time_text_embed.text_embedder.linear_1.weight": "y_embedder.mlp.0.weight",
    "time_text_embed.text_embedder.linear_2.bias": "y_embedder.mlp.2.bias",
    "time_text_embed.text_embedder.linear_2.weight": "y_embedder.mlp.2.weight",
}


class SD3:
    def __init__(
        self, model, shift, control_model_file=None, verbose=False, device="cpu"
    ):

        # NOTE 8B ControlNets were trained with a slightly different forward pass and conditioning,
        # so this is a flag to enable that logic.
        self.using_8b_controlnet = False

        with safe_open(model, framework="pt", device="cpu") as f:
            control_model_ckpt = None
            if control_model_file is not None:
                control_model_ckpt = safe_open(
                    control_model_file, framework="pt", device=device
                )
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device="cuda",
                dtype=torch.float16,
                control_model_ckpt=control_model_ckpt,
                verbose=verbose,
            ).eval()
            load_into(f, self.model, "model.", "cuda", torch.float16)
        if control_model_file is not None:
            control_model_ckpt = safe_open(
                control_model_file, framework="pt", device=device
            )
            self.model.control_model = self.model.control_model.to(device)
            load_into(
                control_model_ckpt,
                self.model.control_model,
                "",
                device,
                dtype=torch.float16,
                remap=CONTROLNET_MAP,
            )

            self.using_8b_controlnet = (
                self.model.control_model.y_embedder.mlp[0].in_features == 2048
            )
            self.model.control_model.using_8b_controlnet = self.using_8b_controlnet
        control_model_ckpt = None


class VAE:
    def __init__(self, model, dtype: torch.dtype = torch.float16):
        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=dtype).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", dtype)


#################################################################################################
### Main inference logic
#################################################################################################


# Note: Sigma shift value, publicly released models use 3.0
SHIFT = 3.0
# Naturally, adjust to the width/height of the model you have
WIDTH = 1024
HEIGHT = 1024
# Pick your prompts
PROMPT_A = "a photo of a cat"
PROMPT_B = "a photo of a dog"
# Most models prefer the range of 4-5, but still work well around 7
CFG_SCALE = 4.5
# Different models want different step counts but most will be good at 50, albeit that's slow to run
# sd3_medium is quite decent at 28 steps
STEPS = 40
# Seed
SEED = 23

SEEDTYPE = "fixed"
# SEEDTYPE = "rand"

# SEEDTYPE = "roll"
# Actual model file path
MODEL = "models/sd3.5_medium.safetensors"
# MODEL = "models/sd3.5_large_turbo.safetensors"
# MODEL = "models/sd3.5_large.safetensors"
# VAE model file path, or set None to use the same model file
VAEFILE = None #"models/diffusion_pytorch_model.safetensors" # None  # "models/sd3_vae.safetensors"
# Optional init image file path
INIT_IMAGE_A = None
INIT_IMAGE_B = None
# ControlNet
CONTROLNET_COND_IMAGE = None
# If init_image is given, this is the percentage of denoising steps to run (1.0 = full denoise, 0.0 = no denoise at all)
DENOISE = 0.8

# Process for combining images 
REDUCTION = 'mean'

# Weighted mean amount 
# (Closer to PROMPT_A) 0.0 <<<<<<< 0.5 (mean) >>>>>>> 1.0 (closer to PROMPT_B)
WEIGHTED_MEAN = 0.5

# Output file path
OUTDIR = "outputs"
# SAMPLER
SAMPLER = "dpmpp_2m"

# SCHEDULER
SCHEDULER = "linear"

# MODEL FOLDER
MODEL_FOLDER = "models"


class SD3Inferencer:

    def __init__(self):
        self.verbose = False

    def print(self, txt):
        if self.verbose:
            print(txt)

    def load(
        self,
        model=MODEL,
        vae=VAEFILE,
        shift=SHIFT,
        controlnet_ckpt=None,
        model_folder: str = MODEL_FOLDER,
        text_encoder_device: str = "cpu",
        verbose=False,
        load_tokenizers: bool = True,
    ):
        self.verbose = verbose
        print("Loading tokenizers...")
        # NOTE: if you need a reference impl for a high performance CLIP tokenizer instead of just using the HF transformers one,
        # check https://github.com/Stability-AI/StableSwarmUI/blob/master/src/Utils/CliplikeTokenizer.cs
        # (T5 tokenizer is different though)
        self.tokenizer = SD3Tokenizer()
        if load_tokenizers:
            print("Loading Google T5-v1-XXL...")
            self.t5xxl = T5XXL(model_folder, text_encoder_device, torch.float32)
            print("Loading OpenAI CLIP L...")
            self.clip_l = ClipL(model_folder)
            print("Loading OpenCLIP bigG...")
            self.clip_g = ClipG(model_folder, text_encoder_device)
        print(f"Loading SD3 model {os.path.basename(model)}...")
        self.sd3 = SD3(model, shift, controlnet_ckpt, verbose, "cuda")
        print("Loading VAE model...")
        self.vae = VAE(vae or model)

        print("Loading CLIP ViT-B/32 Model")
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32')

        print("All Models loaded.")

    def get_empty_latent(self, batch_size, width, height, seed, device="cuda"):
        self.print("Prep an empty latent...")
        shape = (batch_size, 16, height // 8, width // 8)
        latents = torch.zeros(shape, device=device)
        for i in range(shape[0]):
            prng = torch.Generator(device=device).manual_seed(int(seed + i))
            latents[i] = torch.randn(shape[1:], generator=prng, device=device)
        return latents

    def get_sigmas_linear(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)
    
    def get_sigmas_quadratic(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        
        # Create base timesteps
        base_timesteps = torch.linspace(0, 1, steps) ** 2  # Squaring for quadratic scale
        timesteps = start + base_timesteps.to("cuda") * (end - start)  # Scale to [start, end]
        
        # Build list of sigmas like in linear version
        sigs = []
        for ts in timesteps:
            sigs.append(sampling.sigma(ts))
        sigs.append(0.0)  # Append final sigma
        
        return torch.FloatTensor(sigs)
    
    def get_sigmas_logarithmic(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        
        # Logarithmic spacing for slower start
        timesteps = torch.logspace(0, 1, steps, base=10.0) - 1
        timesteps = timesteps / timesteps.max()  # Normalize to [0, 1]
        timesteps = start + timesteps * (end - start)  # Scale to [start, end]
        
        sigs = [sampling.sigma(ts) for ts in timesteps]
        sigs.append(0.0)  # Append final sigma
        return torch.FloatTensor(sigs)
    
    def get_sigmas_cosine(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        
        # Cosine schedule
        timesteps = torch.linspace(0, math.pi / 2, steps)  # Half cosine wave
        timesteps = torch.cos(timesteps)  # Cosine for smooth decay
        timesteps = start + timesteps * (end - start)  # Scale to [start, end]
        
        sigs = [sampling.sigma(ts) for ts in timesteps]
        sigs.append(0.0)  # Append final sigma
        return torch.FloatTensor(sigs)
    
    def get_sigmas_custom(self, sampling, steps, alpha=0.5):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        
        # Mix linear and quadratic schedules with weight `alpha`
        linear_timesteps = torch.linspace(start, end, steps)
        quadratic_timesteps = torch.linspace(0, 1, steps) ** 2
        quadratic_timesteps = start + quadratic_timesteps.to("cuda") * (end - start)
        timesteps = alpha * linear_timesteps + (1 - alpha) * quadratic_timesteps
        
        sigs = [sampling.sigma(ts) for ts in timesteps]
        sigs.append(0.0)  # Append final sigma
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        self.print(
            f"dtype = {latent.dtype}, layout = {latent.layout}, device = {latent.device}"
        )
        return torch.randn(
            latent.size(),
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        ).to(latent.dtype)

    def get_cond(self, prompt):
        self.print("Encode prompt...")
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        t5_out, t5_pooled = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1
        )

    def max_denoise(self, sigmas):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
        return {"c_crossattn": cond, "y": pooled}

    def do_sampling(
        self,
        latent_a,
        latent_b,
        seed,
        conditioning_a,
        conditioning_b,
        neg_cond,
        steps,
        cfg_scale,
        sampler="dpmpp_2m",
        scheduler="linear",
        controlnet_cond=None,
        denoise=1.0,
        reduction='mean',
        weighted_mean=0.5,
        skip_layer_config={},
    ) -> torch.Tensor:
        self.print("Sampling...")
        latent_a = latent_a.half().cuda()
        latent_b = latent_b.half().cuda()

        self.sd3.model = self.sd3.model.cuda()
        noise_a = self.get_noise(seed, latent_a).cuda()
        noise_b = self.get_noise(seed, latent_b).cuda()

        sigma_function_name = f"get_sigmas_{scheduler}"  # e.g., get_sigmas_dpmpp_2m
        try:
            # Attempt to get the method dynamically and call it
            get_sigmas_fn = getattr(self, sigma_function_name)
            sigmas = get_sigmas_fn(self.sd3.model.model_sampling, steps).cuda()
        except AttributeError:
            # Handle the case where the specified method does not exist
            raise ValueError(f"Unknown sigma scheduler: {scheduler}")

#        sigmas = self.get_sigmas_linear(self.sd3.model.model_sampling, steps).cuda()
        #sigmas = self.get_sigmas_quadratic(self.sd3.model.model_sampling, steps).cuda()

        sigmas = sigmas[int(steps * (1 - denoise)) :]

        print(sigma_function_name, sigmas)

        conditioning_a = self.fix_cond(conditioning_a)
        conditioning_b = self.fix_cond(conditioning_b)

        neg_cond = self.fix_cond(neg_cond)
        extra_args = {
            "uncond": neg_cond,
            "cond_scale": cfg_scale,
            "controlnet_cond": controlnet_cond,
        }
        # Creates latent noise for init a and b
        noise_scaled_a = self.sd3.model.model_sampling.noise_scaling(
            sigmas[0], noise_a, latent_a, self.max_denoise(sigmas)
        )
        noise_scaled_b = self.sd3.model.model_sampling.noise_scaling(
            sigmas[0], noise_b, latent_b, self.max_denoise(sigmas)
        )
        sample_fn = getattr(sd3_impls_twistingsquares, f"sample_{sampler}")
        denoiser = (
            SkipLayerCFGDenoiser
            if skip_layer_config.get("scale", 0) > 0
            else CFGDenoiser
        )
        latent, sample_history = sample_fn(
            denoiser(self.sd3.model, steps, skip_layer_config),
            noise_scaled_a,
            noise_scaled_b,
            conditioning_a,
            conditioning_b,
            sigmas,
            reduction=reduction,
            weighted_mean=weighted_mean,
            extra_args=extra_args,
        )
        latent = SD3LatentFormat().process_out(latent)
        self.sd3.model = self.sd3.model.cpu()
        self.print("Sampling done")
        return latent, sample_history

    def vae_encode(
        self, image, using_2b_controlnet: bool = False, controlnet_type: int = 0
    ) -> torch.Tensor:
        self.print("Encoding image to latent...")
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images).cuda()
        if using_2b_controlnet:
            image_torch = image_torch * 2.0 - 1.0
        elif controlnet_type == 1:  # canny
            image_torch = image_torch * 255 * 0.5 + 0.5
        else:
            image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.cuda()
        self.vae.model = self.vae.model.cuda()
        latent = self.vae.model.encode(image_torch).cpu()
        self.vae.model = self.vae.model.cpu()
        self.print("Encoded")
        return latent

    def vae_encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.unsqueeze(0)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def vae_decode(self, latent) -> Image.Image:
       # self.print("Decoding latent to image...")
        latent = latent.cuda()
        self.vae.model = self.vae.model.cuda()
        image = self.vae.model.decode(latent)
        image = image.float()
        self.vae.model = self.vae.model.cpu()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = Image.fromarray(decoded_np)
        # self.print("Decoded")
        return out_image

    def _image_to_latent(
        self,
        image,
        width,
        height,
        using_2b_controlnet: bool = False,
        controlnet_type: int = 0,
    ) -> torch.Tensor:
        image_data = Image.open(image)
        image_data = image_data.resize((width, height), Image.LANCZOS)
        latent = self.vae_encode(image_data, using_2b_controlnet, controlnet_type)
        latent = SD3LatentFormat().process_in(latent)
        return latent
    
    def gen_image(
        self,
        prompts=[PROMPT_A, PROMPT_B],
        width=WIDTH,
        height=HEIGHT,
        steps=STEPS,
        cfg_scale=CFG_SCALE,
        sampler=SAMPLER,
        scheduler=SCHEDULER,
        seed=SEED,
        seed_type=SEEDTYPE,
        out_dir=OUTDIR,
        controlnet_cond_image=CONTROLNET_COND_IMAGE,
        init_image_a=INIT_IMAGE_A,
        init_image_b=INIT_IMAGE_B,
        denoise=DENOISE,
        reduction=REDUCTION,
        weighted_mean=WEIGHTED_MEAN,
        skip_layer_config={},
    ):
        controlnet_cond = None

        if init_image_a:
            latent_a = self._image_to_latent(init_image_a, width, height)
        else:
            latent_a = self.get_empty_latent(1, width, height, seed, "cpu")
            latent_a = latent_a.cuda()

        if init_image_b:
            latent_b = self._image_to_latent(init_image_b, width, height)
        else:
            latent_b = self.get_empty_latent(1, width, height, seed + 10, "cpu")
            latent_b = latent_b.cuda()

        if controlnet_cond_image:
            using_2b, control_type = False, 0
            if self.sd3.model.control_model is not None:
                using_2b = not self.sd3.using_8b_controlnet
                control_type = int(self.sd3.model.control_model.control_type.item())
            controlnet_cond = self._image_to_latent(
                controlnet_cond_image, width, height, using_2b, control_type
            )
        neg_cond = self.get_cond("Blurry, ugly")
        seed_num = None
        pbar = tqdm(enumerate(prompts), total=len(prompts), position=0, leave=True)
        for i, prompt in pbar:
            if seed_type == "roll":
                seed_num = seed if seed_num is None else seed_num + 1
            elif seed_type == "rand":
                seed_num = torch.randint(0, 100000, (1,)).item()
            else:  # fixed
                seed_num = seed

        conditioning_a = self.get_cond(prompts[0])
        conditioning_b = self.get_cond(prompts[1])

        sampled_latent, sample_history = self.do_sampling(
            latent_a,
            latent_b,
            seed_num,
            conditioning_a,
            conditioning_b,
            neg_cond,
            steps,
            cfg_scale,
            sampler,
            scheduler,
            controlnet_cond,
            denoise if init_image_a else 1.0,
            reduction,
            weighted_mean,
            skip_layer_config,
        )
        
        for k, sample in enumerate(sample_history, start=1):
            if (k % 10) == 0:
                latent = SD3LatentFormat().process_out(sample)
                image = self.vae_decode(latent)
                
                score = calculate_clip_score(image, prompts[0], self.clip_model, self.clip_preprocess)
                print("CLIP_SCORE:", score, " PROMPT: ", prompts[0])

                save_path = generate_filename(out_dir, prompts, step=k)
                print(save_path)
                image.save(save_path)

                image = ImageOps.flip(image)
                score = calculate_clip_score(image, prompts[1], self.clip_model, self.clip_preprocess)
                print("CLIP_SCORE:", score, " PROMPT: ", prompts[1])

                save_path = generate_filename(out_dir, prompts, step=k)
                print(save_path)
                image.save(save_path)
            
        image = self.vae_decode(sampled_latent)
        save_path = generate_filename(out_dir, prompts, step=i)
        self.print(f"Saving to to {save_path}")
        image.save(save_path)
        self.print("Done")


CONFIGS = {
    "sd3_medium": {
        "shift": 1.0,
        "steps": 50,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
    },
    "sd3.5_medium": {
        "shift": 3.0,
        "steps": 50,
        "cfg": 5.0,
        "sampler": "dpmpp_2m",
        "skip_layer_config": {
            "scale": 2.5,
            "start": 0.01,
            "end": 0.20,
            "layers": [7, 8, 9],
            "cfg": 4.0,
        },
    },
    "sd3.5_large": {
        "shift": 3.0,
        "steps": 40,
        "cfg": 4.5,
        "sampler": "dpmpp_2m",
    },
    "sd3.5_large_turbo": {"shift": 3.0, "cfg": 1.0, "steps": 4, "sampler": "euler"},
    "sd3.5_large_controlnet_blur": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
    "sd3.5_large_controlnet_canny": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
    "sd3.5_large_controlnet_depth": {
        "shift": 3.0,
        "steps": 60,
        "cfg": 3.5,
        "sampler": "euler",
    },
}



def open_image(image_path, width, height) -> torch.Tensor:
    # Open the image
    image_data = Image.open(image_path)
    # Resize the image
    image_data = image_data.resize((width, height), Image.LANCZOS)
    
    # Transform image to PyTorch tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image_data)
    
    return image_tensor

def save_image(image_tensor, save_path):
    # Convert tensor back to PIL Image
    image_pil = transforms.ToPILImage()(image_tensor)
    # Save the image
    image_pil.save(save_path)



@torch.no_grad()
def main(
    prompt_a=PROMPT_A,
    prompt_b=PROMPT_B,
    model=MODEL,
    out_dir=OUTDIR,
    postfix=None,
    seed=SEED,
    seed_type=SEEDTYPE,
    sampler=None,
    scheduler=None,
    steps=None,
    cfg=None,
    shift=None,
    width=WIDTH,
    height=HEIGHT,
    controlnet_ckpt=None,
    controlnet_cond_image=None,
    vae=VAEFILE,
    init_image_a=INIT_IMAGE_A,
    init_image_b=INIT_IMAGE_B,
    denoise=DENOISE,
    skip_layer_cfg=False,
    verbose=False,
    reduction=REDUCTION,
    weighted_mean=WEIGHTED_MEAN,
    model_folder=MODEL_FOLDER,
    text_encoder_device="cpu",
    **kwargs,
):

    assert not kwargs, f"Unknown arguments: {kwargs}"

    config = CONFIGS.get(os.path.splitext(os.path.basename(model))[0], {})
    _shift = shift or config.get("shift", 3)
    _steps = steps or config.get("steps", 50)
    _cfg = cfg or config.get("cfg", 5)
    _sampler = sampler or config.get("sampler", "dpmpp_2m")
    _scheduler = scheduler or config.get("scheduler", "linear")

    if skip_layer_cfg:
        skip_layer_config = CONFIGS.get(
            os.path.splitext(os.path.basename(model))[0], {}
        ).get("skip_layer_config", {})
        cfg = skip_layer_config.get("cfg", cfg)
    else:
        skip_layer_config = {}

    if controlnet_ckpt is not None:
        controlnet_config = CONFIGS.get(
            os.path.splitext(os.path.basename(controlnet_ckpt))[0], {}
        )
        _shift = shift or controlnet_config.get("shift", shift)
        _steps = steps or controlnet_config.get("steps", steps)
        _cfg = cfg or controlnet_config.get("cfg", cfg)
        _sampler = sampler or controlnet_config.get("sampler", sampler)
        _scheduler = scheduler or config.get("scheduler", scheduler)

    inferencer = SD3Inferencer()

    inferencer.load(
        model,
        vae,
        _shift,
        controlnet_ckpt,
        model_folder,
        text_encoder_device,
        verbose,
    )

    if isinstance(prompt_a, str):
        if os.path.splitext(prompt_a)[-1] == ".txt":
            with open(prompt_a, "r") as f:
                prompts = [l.strip() for l in f.readlines()]
        else:
            prompts = [prompt_a]

    if isinstance(prompt_b, str):
        if os.path.splitext(prompt_b)[-1] == ".txt":
            with open(prompt_b, "r") as f:
                # Use extend to add lines to the list instead of appending a list
                prompts.extend([l.strip() for l in f.readlines()])
        else:
            prompts.append([prompt_b])

    print("prompts: ", prompts)
    print("len prompts: ", len(prompts))
    
    out_dir = os.path.join(
        out_dir,
        (
            os.path.splitext(os.path.basename(model))[0]
            + (
                "_" + os.path.splitext(os.path.basename(controlnet_ckpt))[0]
                if controlnet_ckpt is not None
                else ""
            )
        ),
        generate_outdir_name(prompts, steps, cfg),
    )
    
    os.makedirs(out_dir, exist_ok=False)
    save_parameters_to_file(out_dir, locals()) # Save parameters for reproducability

    inferencer.gen_image(
        prompts,
        width,
        height,
        _steps,
        _cfg,
        _sampler,
        _scheduler,
        seed,
        seed_type,
        out_dir,
        controlnet_cond_image,
        init_image_a,
        init_image_b,
        denoise,
        reduction,
        weighted_mean,
        skip_layer_config,
    )


if __name__ == "__main__":
    fire.Fire(main)
