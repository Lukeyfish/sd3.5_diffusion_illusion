import torch
from PIL import Image
#from sd3_infer_twistingsquares import SD3LatentFormat, SD3Inferencer
import clip
import os
import datetime
import re
import torch.nn.functional as F

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


def flip_latent_upside_down(latent: torch.Tensor) -> torch.Tensor:
        # Flipping the latent tensor upside down (vertically)
        flipped_latent = torch.flip(latent, dims=[-2])
        return flipped_latent

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



def merge_denoised_outputs(denoised_a, denoised_b, method="average", method_param=0.5):
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

        B, C, H, W = denoised_a.shape
    
        # Reshape to handle channels separately
        a_view = denoised_a.view(B, C, H*W)
        b_view = denoised_b.view(B, C, H*W)
        
        # Normalize per channel
        a_norm = F.normalize(a_view, dim=-1)
        b_norm = F.normalize(b_view, dim=-1)
        
        # Compute attention weights per channel
        attention = torch.bmm(a_norm, b_norm.transpose(-2, -1))
        attention = F.softmax(attention / method_param, dim=-1)
        
        # Apply attention and reshape
        merged = torch.bmm(attention, b_view)
        merged = merged.view(B, C, H, W)
        
        # Blend with original using a residual connection
        return 0.5 * (denoised_a + merged)
        '''
        # Use attention mechanism to weight features
        query = denoised_a
        key = denoised_b
        attention_weights = torch.matmul(
            F.normalize(query.flatten(2), dim=-1),
            F.normalize(key.flatten(2), dim=-1).transpose(-2, -1)
        )
        attention_weights = F.softmax(attention_weights / method_param, dim=-1)
        merged = torch.matmul(attention_weights, denoised_b.flatten(2)).view_as(denoised_a)
        return 0.5 * (denoised_a + merged)'''

    elif method == "frequency":
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
    
        '''
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
        return torch.fft.irfft2(merged, s=denoised_a.shape[-2:])'''

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

def as_torch_image(image):
    """
    Convert an image to a PyTorch tensor.
    
    Parameters:
        image (PIL Image or numpy array): The input image.
        
    Returns:
        torch.Tensor: The image converted to a PyTorch tensor.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    # If image is a numpy array, ensure it's in (H, W, C) format
    image = image.transpose((2, 0, 1)) if image.ndim == 3 else image
    return torch.tensor(image).float()