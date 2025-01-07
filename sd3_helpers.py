import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import clip
import os
import datetime
import re
import math
import torch.nn.functional as F
import numpy as np

def calculate_clip_score(sample, prompt, clip_model, preprocess):
    """
    Calculate the CLIP score between an image and a text prompt.
    
    Args:
        image_path (str): Path to the image file
        prompt (str): Text prompt to compare against
        clip_model: Pre-loaded CLIP model (clip_g or clip_l)
        preprocess: CLIP model's preprocessing transform
    
    Returns:
        float: CLIP similarity score between the image and text
    """
    
        # Preprocess the image and tokenize the text
    image_input = preprocess(sample).unsqueeze(0)
    text_input = clip.tokenize([prompt])
    
    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    clip_model = clip_model.to(device)
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()
    
    return clip_score

def flip_latent_with_blend(latent: torch.Tensor, blend_zone: int = 4) -> torch.Tensor:
    """Flip latent with smooth blending at the boundary"""
    b, c, h, w = latent.shape
    flipped = torch.flip(latent, dims=[-2])
    
    # Create a blending mask
    mask = torch.ones_like(latent)
    for i in range(blend_zone):
        # Blend middle rows
        mid = h // 2
        alpha = i / blend_zone
        mask[:, :, mid-i:mid+i, :] = alpha
        
    return latent * (1 - mask) + flipped * mask



def save_parameters_to_file(out_dir, params_dict):
    """
    Save all parameters to a text file in the output directory.
    
    Args:
        out_dir (str): Output directory path
        params_dict (dict): Dictionary of parameters from locals()
    """
    
    
    # Create parameters log file path
    params_file = os.path.join(out_dir, "parameters.txt")
    
    # Filter out internal variables and None values
    filtered_params = {
        k: str(v) if v is not None else "None"
        for k, v in params_dict.items()
        if not k.startswith('_') and k != 'kwargs' and k != 'out_dir'
    }
    
    # Write parameters to file
    with open(params_file, "w") as f:
        # Write timestamp header
        f.write(f"Generation Parameters - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        # Write each parameter
        for key, value in sorted(filtered_params.items()):
            f.write(f"{key}: {value}\n")
            
    return params_file

def sanitize_filename(text):
    return re.sub(r"[^\w\-\.]+", "_", text)

def generate_outdir_name(prompts, steps, cfg):
    prompt_part = sanitize_filename(prompts[0])[:30] + sanitize_filename(prompts[1])[:30]  # Limit prompt part for readability
    timestamp = datetime.datetime.now().strftime("_%Y-%m-%dT%H-%M-%S")
    return os.path.join(f"{prompt_part}_s{steps}_cfg{cfg}{timestamp}")

def generate_filename(out_dir, prompts, step):
    prompt_part = sanitize_filename(prompts[0])[:30] + sanitize_filename(prompts[1])[:30]
    return os.path.join(out_dir, f"{prompt_part}_{step:06d}.png")

def merge_images(image_a, image_b, method="average", method_param=0.5):
    tensor_a = pil_to_tensor(image_a)
    tensor_b = pil_to_tensor(image_b)

    if method == "mean":
        weighted_mean = method_param
        denoised = ((1 - weighted_mean) * tensor_a) + (weighted_mean * tensor_b)
        return to_pil_image(denoised)




def merge_denoised_outputs(denoised_a, denoised_b, method="mean", method_param=0.5, method_param2 = 0.5):
    """
    Merge denoised outputs using various methods
    
    Args:
        denoised_a: First orientation output
        denoised_b: Second orientation output
        method: Merging strategy to use
        **kwargs: Additional arguments for specific methods
    """
    if method == "mean":
        weighted_mean = method_param
        denoised = ((1 - weighted_mean) * denoised_a) + (weighted_mean * denoised_b)
        return denoised
    
    elif method == "features":
        # Find shared and orientation-dependent features
        shared_features = (denoised_a + denoised_b) / 2
        orientation_features_a = denoised_a - shared_features
        orientation_features_b = denoised_b - shared_features
        
        # Create orientation-dependent denoised result
        denoised = shared_features + method_param * (orientation_features_a - orientation_features_b)

        return denoised
    
    elif method == "alternate":
        iteration = method_param
        return denoised_a if iteration % 2 == 0 else denoised_b

    elif method == "channel_attention":
        attention = torch.sigmoid(
            F.adaptive_avg_pool2d(denoised_a, 1) + 
            F.adaptive_avg_pool2d(denoised_b, 1)
        )
        return attention * denoised_a + (1 - attention) * denoised_b
    
    elif method == "spatial_attention":
        # Compute spatial attention maps
        a_pool = torch.mean(denoised_a, dim=1, keepdim=True)
        b_pool = torch.mean(denoised_b, dim=1, keepdim=True)
        
        attention = torch.sigmoid(torch.cat([a_pool, b_pool], dim=1))
        attention = F.softmax(attention, dim=1)
        
        return attention[:, 0:1] * denoised_a + attention[:, 1:2] * denoised_b
    
    elif method == "dual_attention":
        # Channel attention 
        channel_attn = torch.sigmoid(
            F.adaptive_avg_pool2d(denoised_a, 1) +
            F.adaptive_avg_pool2d(denoised_b, 1)
        )
        
        # Spatial attention
        max_pool = torch.max(denoised_a + denoised_b, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(denoised_a + denoised_b, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(max_pool + avg_pool)
        
        # Combine both attentions
        result = channel_attn * spatial_attn * denoised_a + (1 - channel_attn * spatial_attn) * denoised_b
        return result
    
    elif method == "pyramid_attention":
        """
        Multi-scale attention using different pooling sizes
        """
        # Multiple scales of pooling
        attn_1 = F.adaptive_avg_pool2d(denoised_a + denoised_b, 1)
        attn_2 = F.adaptive_avg_pool2d(denoised_a + denoised_b, 2)
        attn_4 = F.adaptive_avg_pool2d(denoised_a + denoised_b, 4)
        
        # Upsample all to original size
        attn_2 = F.interpolate(attn_2, size=denoised_a.shape[2:])
        attn_4 = F.interpolate(attn_4, size=denoised_a.shape[2:])
        
        # Combine multi-scale attention
        attention = torch.sigmoid(attn_1 + attn_2 + attn_4)
        
        return attention * denoised_a + (1 - attention) * denoised_b

        
    elif method == "interpolation":
        alpha = method_param
        # Spherical linear interpolation using vector_norm
        dot_product = (denoised_a * denoised_b).sum(dim=(1, 2, 3), keepdim=True)
        denoised_a_norm = torch.linalg.vector_norm(denoised_a, dim=(1, 2, 3), keepdim=True)
        denoised_b_norm = torch.linalg.vector_norm(denoised_b, dim=(1, 2, 3), keepdim=True)
        
        # Normalize vectors before computing angle
        denoised_a_normalized = denoised_a / (denoised_a_norm + 1e-8)
        denoised_b_normalized = denoised_b / (denoised_b_norm + 1e-8)
        cos_omega = (denoised_a_normalized * denoised_b_normalized).sum(dim=(1, 2, 3), keepdim=True).clamp(-1, 1)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega)
        
        # Handle case where vectors are nearly parallel
        if sin_omega.abs().item() < 1e-6:
            return alpha * denoised_a + (1 - alpha) * denoised_b
            
        return (torch.sin((1 - alpha) * omega) / sin_omega) * denoised_a + \
               (torch.sin(alpha * omega) / sin_omega) * denoised_b
    
    elif method == "feature_mixing":
        alpha = method_param
        # Mix features using frequency decomposition
        freqs1 = torch.fft.rfft2(denoised_a)
        freqs2 = torch.fft.rfft2(denoised_b)
        
        # Mix high and low frequencies differently
        freq_weights = torch.linspace(alpha, 1-alpha, freqs1.shape[-1], device=freqs1.device)
        freq_weights = freq_weights.view(1, 1, 1, -1)
        
        mixed_freqs = freq_weights * freqs1 + (1 - freq_weights) * freqs2
        return torch.fft.irfft2(mixed_freqs, s=(denoised_a.shape[-2], denoised_a.shape[-1]))
    
    elif method == "adaptive":
        # Compute channel-wise statistics
        mean1 = denoised_a.mean(dim=(2, 3), keepdim=True)
        mean2 = denoised_b.mean(dim=(2, 3), keepdim=True)
        std1 = denoised_a.std(dim=(2, 3), keepdim=True)
        std2 = denoised_b.std(dim=(2, 3), keepdim=True)
        
        # Create adaptive thresholds
        threshold = (std1 + std2) / 2
        
        # Create masks based on deviation from mean
        mask1 = (torch.abs(denoised_a - mean1) > threshold).float()
        mask2 = (torch.abs(denoised_b - mean2) > threshold).float()
        
        # Smooth masks
        kernel_size = 5
        smoothing = torch.ones(1, 1, kernel_size, kernel_size, device=denoised_a.device) / (kernel_size ** 2)
        mask1 = F.conv2d(mask1, smoothing.repeat(16, 1, 1, 1), padding=kernel_size//2, groups=16)
        mask2 = F.conv2d(mask2, smoothing.repeat(16, 1, 1, 1), padding=kernel_size//2, groups=16)
        
        # Normalize masks
        total_mask = mask1 + mask2
        mask1 = mask1 / (total_mask + 1e-6)
        mask2 = mask2 / (total_mask + 1e-6)
        
        return mask1 * denoised_a + mask2 * denoised_b
    else:
        raise ValueError(f"Unknown combination method: {method}")
    
def rotate_tiles(image, num_divisions=4):
    # image=as_torch_image(image)
    # Assuming image is a tensor of shape (num_channels, height, width)
    num_channels, height, width = image.shape

    tile_size=width//num_divisions
    
    # Calculate the number of tiles in each dimension
    tiles_x = width // tile_size
    tiles_y = height // tile_size

    # Initialize an output tensor
    output = torch.zeros_like(image)

    for x in range(tiles_x):
        for y in range(tiles_y):
            # Extract the tile
            tile = image[:, y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]

            # Check if the tile should be rotated 90 or -90 degrees (checker pattern)
            if (x + y) % 2 == 0:
                # Rotate 90 degrees
                tile = tile.rot90(1, [1, 2])
            else:
                # Rotate -90 degrees
                tile = tile.rot90(-1, [1, 2])

            # Place the rotated tile back in the output tensor
            output[:, y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size] = tile

    return output

    
def rotate_latent_tiles(latent, inverse=False, num_divisions=4):
        
        batch_size, channels, height, width = latent.shape
        tile_size = width // num_divisions
        tiles_x = width // tile_size
        tiles_y = height // tile_size
        
        output = torch.zeros_like(latent)
        
        for b in range(batch_size):
            for x in range(tiles_x):
                for y in range(tiles_y):
                    tile = latent[b:b+1, :, y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]
                    

                    # Flips tiles depending on needed orientation
                    if (x + y) % 2 == 0:
                        tile = tile.rot90(-1 if inverse else 1, [2, 3])
                    else:
                        tile = tile.rot90(1 if inverse else -1, [2, 3])
                    
                    output[b:b+1, :, y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size] = tile
        
        return output

    
def flip_latent(latent: torch.Tensor, degree: int = 0) -> torch.Tensor:
    """
    Flips or rotates a latent tensor based on the specified degree.
    
    Args:
        latent (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
        degree (int): Degree of rotation/flip. 
            - Any degree will be normalized to 0, 90, 180, or 270
            - Negative degrees work as counter-clockwise rotations
            - -1 is reserved for vertical flip
            - 1 is reserved for reverting vertical flip
    
    Returns:
        torch.Tensor: Flipped/rotated tensor with same shape as input
    
    Example:
        For a tensor of shape [1, 16, 128, 128]:
        - Vertical/horizontal flips will operate on the 128x128 spatial dimensions
        - Rotations preserve the batch and channel dimensions
    """
    # Verify input has 4 dimensions
    if len(latent.shape) != 4:
        raise ValueError(f"Expected 4D tensor [batch, channels, height, width], got shape {latent.shape}")
    
    # Special cases for vertical flip
    if degree == -1:
        return torch.flip(latent, dims=[2])  # Flip height dimension
    elif degree == 1:  # Revert vertical flip
        return torch.flip(latent, dims=[2])  # Flip height dimension
    
    # Normalize the degree to be between 0 and 360
    normalized_degree = degree % 360
    if normalized_degree < 0:
        normalized_degree += 360
        
    # Now convert to one of the four standard rotations
    if normalized_degree == 0:
        return latent
    elif normalized_degree == 180:
        return torch.flip(latent, dims=[3])  # Flip width dimension
    elif normalized_degree == 90:
        return torch.rot90(latent, k=1, dims=(2, 3))  # Rotate spatial dimensions
    elif normalized_degree == 270:
        return torch.rot90(latent, k=3, dims=(2, 3))  # Rotate spatial dimensions
    else:
        # Round to nearest 90 degrees
        rounded_degree = round(normalized_degree / 90) * 90
        if rounded_degree == 360:
            rounded_degree = 0
        print(f"Warning: Degree {degree} rounded to {rounded_degree}")
        return flip_latent(latent, rounded_degree)