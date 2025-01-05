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


"""def rotate_latent(latent: torch.Tensor, degree: int) -> torch.Tensor:
    # Validate the degree input
    if degree not in [90, 180, 270]:
        raise ValueError("Degree must be 90, 180, or 270")

    # Rotate the latent tensor based on the degree
    if degree == 90:
        rotated_latent = torch.rot90(latent, k=1, dims=[-2, -1])
    elif degree == 180:
        rotated_latent = torch.rot90(latent, k=2, dims=[-2, -1])
    elif degree == 270:
        rotated_latent = torch.rot90(latent, k=3, dims=[-2, -1])
    
    return rotated_latent"""
'''
def rotate_latent(latent: torch.Tensor, degree: int) -> torch.Tensor:
        # Flipping the latent tensor upside down (vertically)
        flipped_latent = torch.flip(latent, dims=[-2])
        return flipped_latent'''

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




def merge_denoised_outputs(denoised_a, denoised_b, method="average", method_param=0.5, method_param2 = 0.5):
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
    
    elif method == "alternate":
        iteration = method_param
        return denoised_a if iteration % 2 == 0 else denoised_b

    elif method == "attention":
        
            
        return denoised_a

    elif method == "frequency_smooth":
        # Merge in frequency domain
        fa = torch.fft.rfft2(denoised_a)
        fb = torch.fft.rfft2(denoised_b)
        
        # Create proper sized mask
        h, w = fa.shape[-2:]  # Get the actual FFT dimensions
        y = torch.linspace(0, 1, h).view(-1, 1).to(fa.device)
        x = torch.linspace(0, 1, w).view(1, -1).to(fa.device)
        
        # Compute radius for each point
        radius = torch.sqrt(y**2 + x**2)
        radius = radius / radius.max()  # Normalize to [0,1]
        
        # Create smooth transition
        transition_width = 0.1
        mask = torch.sigmoid((radius - method_param) / transition_width)
        mask = mask.unsqueeze(0).unsqueeze(0).expand_as(fa)
        
        # Apply mask
        merged = fa * mask + fb * (1 - mask)
        return torch.fft.irfft2(merged, s=denoised_a.shape[-2:])
    
    elif method == "frequency":
        
        # Merge in frequency domain
        fa = torch.fft.rfft2(denoised_a)
        fb = torch.fft.rfft2(denoised_b)
        
        # Take high frequencies from one, low from other
        threshold = method_param
        mask = torch.ones_like(fa, dtype=torch.bool)
        mid_x = mask.shape[-2] // 2
        mid_y = mask.shape[-1] // 2
        mask[..., :int(mid_x*threshold), :int(mid_y*threshold)] = False
        
        merged = torch.where(mask, fa, fb)
        return torch.fft.irfft2(merged, s=denoised_a.shape[-2:])

    elif method == "feature_mapping":
        blend_factor = 0.8
        B, C, H, W = denoised_a.shape
    
        # Reshape to (B, C, H*W)
        a_feat = denoised_a.view(B, C, -1)
        b_feat = denoised_b.view(B, C, -1)
        
        # Normalize features
        a_norm = F.normalize(a_feat, dim=1)
        b_norm = F.normalize(b_feat, dim=1)
        
        # Compute bidirectional correspondence
        sim_ab = torch.bmm(a_norm.transpose(1, 2), b_norm)
        sim_ba = torch.bmm(b_norm.transpose(1, 2), a_norm)
        
        # Get matches and confidence both ways
        conf_ab, matches_ab = sim_ab.max(dim=2)
        conf_ba, matches_ba = sim_ba.max(dim=2)
        
        # Combine confidences
        confidence = (conf_ab + conf_ba) / 2
        
        # Apply confidence threshold
        confidence = torch.where(
            confidence > method_param,
            confidence,
            torch.ones_like(confidence) * blend_factor
        )
        
        # Gather matched features
        b_matched = torch.gather(b_feat, 2, matches_ab.unsqueeze(1).expand(-1, C, -1))
        
        # Reshape back and blend
        b_matched = b_matched.view(B, C, H, W)
        confidence = confidence.view(B, 1, H, W)
        
        return confidence * denoised_a + (1 - confidence) * b_matched
    
    elif method == "cross_corr":
        B, C, H, W = denoised_a.shape
        pad = method_param // 2
        
        # Normalize inputs
        a_norm = (denoised_a - denoised_a.mean(dim=(2,3), keepdim=True)) / (denoised_a.std(dim=(2,3), keepdim=True) + 1e-8)
        b_norm = (denoised_b - denoised_b.mean(dim=(2,3), keepdim=True)) / (denoised_b.std(dim=(2,3), keepdim=True) + 1e-8)
        
        # Pad inputs
        a_pad = F.pad(a_norm, (pad, pad, pad, pad), mode='reflect')
        b_pad = F.pad(b_norm, (pad, pad, pad, pad), mode='reflect')
        
        # Compute local correlation using convolution
        correlation = torch.zeros_like(denoised_a)
        for i in range(method_param):
            for j in range(method_param):
                shifted_b = b_pad[:, :, i:i+H, j:j+W]
                correlation += a_norm * shifted_b
        
        correlation = correlation / (method_param * method_param)
        
        # Convert correlation to weights
        weights = torch.sigmoid(correlation)
        return weights * denoised_a + (1 - weights) * denoised_b
    
    elif method == "frequency_2":
        # Decompose both predictions into frequency bands
        freq_a = decompose_frequencies(denoised_a)
        freq_b = decompose_frequencies(denoised_b)
        
        # Mix frequencies with different weights
        mixed_freqs = []
        for j, (fa, fb) in enumerate(zip(freq_a, freq_b)):
            # Higher weights for low frequencies in both predictions
            weight = 0.5 if j == len(freq_a) - 1 else 0.5
            mixed_freqs.append(weight * fa + (1 - weight) * fb)
        
        denoised = combine_frequencies(mixed_freqs)
        return denoised
    
    elif method == "balanced":
        # Use the balanced influence approach
        alpha = adaptive_balance_weight(denoised_a, denoised_b, method_param, num_timesteps=1000)
        balanced_denoised = alpha * denoised_a + (1 - alpha) * denoised_b
        
        # Optional: add stability term to prevent either interpretation from dominating
        stability_term = 0.1 * torch.mean(torch.abs(denoised_a - denoised_b))
        denoised = balanced_denoised - stability_term
        return denoised
    
    elif method == "progressive":
        progress = method_param / 1000
        weight = 0.5 * (1 + math.cos(math.pi * progress))

        weight = weight * (2 * method_param2)

        return (1 - weight) * denoised_a + weight * denoised_b

    else:
        raise ValueError(f"Unknown combination method: {method}")
    
def calculate_influence(prediction: torch.Tensor, method="energy") -> torch.Tensor:
    """
    Calculate the influence score of a prediction in latent space.
    
    Args:
        prediction: Tensor of model predictions [B, C, H, W]
        method: Calculation method ("energy", "magnitude", or "spatial")
    
    Returns:
        Tensor of influence scores [B, 1]
    """
    if method == "energy":
        # Calculate total energy in the prediction
        energy = torch.sum(prediction ** 2, dim=(1, 2, 3))
        return energy.unsqueeze(1)
    
    elif method == "magnitude":
        # Use average magnitude of activations
        magnitude = torch.mean(torch.abs(prediction), dim=(1, 2, 3))
        return magnitude.unsqueeze(1)
    
    elif method == "spatial":
        # Calculate spatial attention map
        spatial_weights = torch.mean(torch.abs(prediction), dim=1)
        # Get overall spatial influence
        spatial_score = torch.mean(spatial_weights.view(prediction.shape[0], -1), dim=1)
        return spatial_score.unsqueeze(1)
    
    raise ValueError(f"Unknown method: {method}")

def balance_scores(score_a: torch.Tensor, score_b: torch.Tensor, 
                  target_ratio: float = 1.0, smoothing: float = 0.1) -> torch.Tensor:
    """
    Balance influence scores between two predictions.
    
    Args:
        score_a: Influence scores for first prediction [B, 1]
        score_b: Influence scores for second prediction [B, 1]
        target_ratio: Desired ratio between scores (default 1.0 for equal influence)
        smoothing: Smoothing factor for weight calculation
    
    Returns:
        Tensor of weights for prediction A [B, 1]
    """
    # Add smoothing to prevent division by zero
    scores_sum = score_a + score_b + smoothing
    
    # Calculate current ratio
    current_ratio = score_a / score_b
    
    # Calculate adjustment factor to reach target ratio
    adjustment = torch.sqrt(target_ratio / (current_ratio + smoothing))
    
    # Calculate balanced weight for prediction A
    alpha = (score_a * adjustment) / scores_sum
    
    # Clamp weights to prevent extreme values
    alpha = torch.clamp(alpha, 0.3, 0.7)
    
    return alpha

def adaptive_balance_weight(pred_a: torch.Tensor, pred_b: torch.Tensor, 
                          timestep: int, num_timesteps: int) -> torch.Tensor:
    """
    Calculate adaptive balance weight based on timestep and predictions.
    
    Args:
        pred_a: First prediction
        pred_b: Second prediction
        timestep: Current timestep
        num_timesteps: Total number of timesteps
    
    Returns:
        Balance weight for prediction A
    """
    # Calculate progress through diffusion (0 to 1)
    progress = timestep / num_timesteps
    
    # Calculate influence scores
    # method: Calculation method ("energy", "magnitude", or "spatial")
    score_a = calculate_influence(pred_a, method="energy")
    score_b = calculate_influence(pred_b, method="energy")
    
    # Adjust target ratio based on diffusion progress
    # Early steps: Allow more variation
    # Later steps: Force more balance
    target_ratio = 1.0
    smoothing = 0.1 + 0.4 * (1 - progress)  # More smoothing early on
    
    # Get balanced weight
    alpha = balance_scores(score_a, score_b, target_ratio, smoothing)
    
    return alpha

def balanced_denoising_step(pred_a: torch.Tensor, pred_b: torch.Tensor, 
                          timestep: int, num_timesteps: int) -> torch.Tensor:
    """
    Perform a single denoising step with balanced influence.
    
    Args:
        model: Diffusion model
        x: Input latents
        sigmas: Noise levels
        cond_a: First conditioning
        cond_b: Second conditioning
        timestep: Current timestep
        num_timesteps: Total number of timesteps
    
    Returns:
        Updated latents
    """
    
    # Calculate adaptive balance weight
    alpha = adaptive_balance_weight(pred_a, pred_b, timestep, num_timesteps)
    
    # Combine predictions
    balanced_pred = alpha * pred_a + (1 - alpha) * pred_b
    
    # Optional: Add stability term to prevent either interpretation from dominating
    stability_term = 0.1 * torch.mean(torch.abs(pred_a - pred_b))
    balanced_pred = balanced_pred - stability_term
    
    return balanced_pred

def decompose_frequencies(latents: torch.Tensor, num_bands: int = 4):
    """
    Decompose latents into different frequency bands using Laplacian pyramid
    
    Args:
        latents: Input tensor [B, C, H, W]
        num_bands: Number of frequency bands to decompose into
    
    Returns:
        List of tensors, each containing different frequency information
    """
    pyramid = []
    current = latents
    
    for i in range(num_bands - 1):
        # Blur and downsample
        blurred = F.avg_pool2d(current, kernel_size=2, stride=2)
        
        # Upsample blurred version
        upsampled = F.interpolate(
            blurred, 
            size=current.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Get high frequency details
        high_freq = current - upsampled
        pyramid.append(high_freq)
        
        current = blurred
    
    # Add remaining low frequencies
    pyramid.append(current)
    
    return pyramid

def combine_frequencies(pyramid: list):
    """
    Reconstruct latents from frequency bands
    """
    result = pyramid[-1]  # Start with lowest frequencies
    
    for high_freq in reversed(pyramid[:-1]):
        result = result + high_freq
        
    return result
    
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

def create_attention_mask(latents: torch.Tensor, method: str = "gradient") -> torch.Tensor:
    """
    Create attention masks for latent space combination.
    
    Args:
        latents: Input tensor [B, C, H, W]
        method: Mask generation method ("gradient", "energy", or "spatial")
    
    Returns:
        Attention mask [B, 1, H, W]
    """
    if method == "gradient":
        return create_gradient_mask(latents)
    elif method == "energy":
        return create_energy_mask(latents)
    elif method == "spatial":
        return create_spatial_mask(latents)
    else:
        raise ValueError(f"Unknown method: {method}")

def create_gradient_mask(latents: torch.Tensor) -> torch.Tensor:
    """
    Create masks based on gradient magnitude in latent space.
    """
    # Calculate gradients in x and y directions
    grad_x = torch.abs(latents[..., 1:, :] - latents[..., :-1, :])
    grad_y = torch.abs(latents[..., :, 1:] - latents[..., :, :-1])
    
    # Pad to match original size
    grad_x = F.pad(grad_x, (0, 0, 0, 1))
    grad_y = F.pad(grad_y, (0, 1, 0, 0))
    
    # Combine gradients
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    
    # Average across channels
    mask = torch.mean(grad_magnitude, dim=1, keepdim=True)
    
    # Normalize to [0, 1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    
    return mask

def create_energy_mask(latents: torch.Tensor) -> torch.Tensor:
    """
    Create masks based on local energy distribution.
    """
    # Calculate local energy
    energy = torch.sum(latents**2, dim=1, keepdim=True)
    
    # Apply local averaging
    kernel_size = 3
    padding = kernel_size // 2
    energy = F.avg_pool2d(energy, kernel_size=kernel_size, 
                         stride=1, padding=padding)
    
    # Normalize
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    
    # Apply soft thresholding
    mask = torch.sigmoid((energy - 0.5) * 10)
    
    return mask

def create_spatial_mask(latents: torch.Tensor) -> torch.Tensor:
    """
    Create masks based on spatial attention patterns.
    """
    B, C, H, W = latents.shape
    
    # Calculate cross-channel attention
    latents_flat = latents.view(B, C, -1)
    attention = torch.bmm(latents_flat.transpose(1, 2), latents_flat)
    attention = F.softmax(attention, dim=-1)
    
    # Project back to spatial dimensions
    mask = attention.mean(1).view(B, 1, H, W)
    
    # Normalize
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    
    return mask

def combine_masks(mask_a: torch.Tensor, mask_b: torch.Tensor, 
                 smoothing: float = 0.1) -> torch.Tensor:
    """
    Combine two masks ensuring they sum to 1.
    """
    # Add smoothing to prevent hard transitions
    mask_sum = mask_a + mask_b + smoothing
    
    # Normalize
    mask_a = mask_a / mask_sum
    mask_b = mask_b / mask_sum
    
    return mask_a, mask_b

def create_complementary_masks(latents_a: torch.Tensor, 
                             latents_b: torch.Tensor) -> tuple:
    """
    Create complementary masks for two sets of latents.
    """
    # Create initial masks
    mask_a = create_attention_mask(latents_a, method="energy")
    mask_b = create_attention_mask(latents_b, method="energy")
    
    # Ensure masks are complementary
    mask_a, mask_b = combine_masks(mask_a, mask_b)
    
    # Optional: Apply spatial smoothing
    kernel_size = 3
    padding = kernel_size // 2
    mask_a = F.avg_pool2d(mask_a, kernel_size=kernel_size, 
                         stride=1, padding=padding)
    mask_b = F.avg_pool2d(mask_b, kernel_size=kernel_size, 
                         stride=1, padding=padding)
    
    return mask_a, mask_b