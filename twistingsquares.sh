#!/bin/bash
# Script to run sd3_infer_twistingsquares.py with predefined arguments

# Define variables
PROMPT_A="prompts/prompt_a.txt"
INIT_IMAGE_A="input_images/JAMES_CROPPED.jpg"
PROMPT_B="prompts/prompt_b.txt"
INIT_IMAGE_B="input_images/EMMY_CROPPED.jpg"
MODEL="models/sd3.5_medium.safetensors"
STEPS=60  # Define steps
CFG=6 # Define cfg
VERBOSE="True"
DENOISE=1.0 # Define denoise
REDUCTION="mean" # "mean" # mean or alternate for latent combination

WEIGHTED_MEAN="0.5" # (ONLY APPLIES FOR MEAN REDUCTION) Applies weighted average across latent combination
# (Closer to PROMPT_A) 0.0 <<<<<<< 0.5 (mean) >>>>>>> 1.0 (closer to PROMPT_B)

SEEDTYPE="fixed" # rand, roll, or fixed for replication
SKIPLAYERCFG="True" # if True, potentially better struture and anatomy coherency from SD3.5-Medium

#    --init_image_a "$INIT_IMAGE_A" \
#    --init_image_b "$INIT_IMAGE_B" \

# Run the Python script with the arguments
python3 sd3_infer_twistingsquares.py \
    --prompt_a "$PROMPT_A" \
    --prompt_b "$PROMPT_B" \
    --model "$MODEL" \
    --seed_type "$SEEDTYPE" \
    --steps "$STEPS" \
    --cfg "$CFG" \
    --denoise "$DENOISE" \
    --skip_layer_cfg "$SKIPLAYERCFG" \
    --verbose "$VERBOSE" \
    --reduction "$REDUCTION" \
    --weighted_mean "$WEIGHTED_MEAN"
