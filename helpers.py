import torch
from PIL import Image
#from sd3_infer_twistingsquares import SD3LatentFormat, SD3Inferencer
import clip
import os
from datetime import datetime
import re

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
        f.write(f"Generation Parameters - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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