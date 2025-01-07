#!/bin/bash
# Script to run sd3_infer_illusions.py with predefined arguments

# Define variables
PROMPT_A="prompts/prompt_a.txt"
INIT_IMAGE_A="input_images/JAMES_CROPPED.jpg"
PROMPT_B="prompts/prompt_b.txt"
INIT_IMAGE_B="input_images/EMMY_CROPPED.jpg"
MODEL="models/sd3.5_medium.safetensors"
STEPS=30
CFG=7.5 
VERBOSE="True"
DENOISE=1.0 
SCHEDULER="linear" # Sigma scheduler, (linear, quadratic, cosine, logarithmic, custom)

ILLUSION_TYPE="1" # 90, 180, 1 for flip
METHOD="dual_attention" # Method for latent combination (mean, alternate, attention, frequency, gradient, feature_mapping)
METHOD_PARAM="0.5" # Specific kwargs required depending on method, value excluded if not needed:
#     mean:                       (Closer to PROMPT_B) 0.0 <<<<<<< 0.5 (default) >>>>>>> 1.0 (closer to PROMPT_A)
#     attention:                (very sharp attention) 0.1 <<<<<<< 1.0 (default) >>>>>>> 10 (soft attention, uniform blending)
#     frequency:  (takes all frequencies from image B) 0.0 <<<<<<< 0.5 (default) >>>>>>> 1.0 (takes all frequencies from image A)

CONTINUOUS="False" # If you want to generate more than one image
SEEDTYPE="fixed" # rand, roll, or fixed for replication
SKIPLAYERCFG="True" # if True, potentially better struture and anatomy coherency from SD3.5-Medium

#    --init_image_a "$INIT_IMAGE_A" \
#    --init_image_b "$INIT_IMAGE_B" \

# Run the Python script with the arguments
python3 sd3_infer_illusions.py \
    --prompt_a "$PROMPT_A" \
    --prompt_b "$PROMPT_B" \
    --model "$MODEL" \
    --seed_type "$SEEDTYPE" \
    --steps "$STEPS" \
    --cfg "$CFG" \
    --denoise "$DENOISE" \
    --skip_layer_cfg "$SKIPLAYERCFG" \
    --verbose "$VERBOSE" \
    --method "$METHOD" \
    --method_param "$METHOD_PARAM" \
    --scheduler "$SCHEDULER" \
    --illusion_type "$ILLUSION_TYPE" \
    --continuous "$CONTINUOUS" \