#!/bin/bash

# Default values from your original script
DEFAULT_PARAMS=(
    "--prompt_a prompts/prompt_a.txt"
    "--prompt_b prompts/prompt_b.txt"
    "--init_image_a input_images/JAMES_CROPPED.jpg"
    "--init_image_b input_images/EMMY_CROPPED.jpg"
    "--model models/sd3.5_medium.safetensors"
    "--steps 50"
    "--cfg 7"
    "--verbose True"
    "--denoise 1.0"
    "--scheduler linear"
    "--method attention"
    "--method_param 0.1"
    "--seed_type fixed"
    "--skip_layer_cfg True"
)

# Function to print usage
print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --config FILE    Path to job configuration file"
    echo "  --output DIR     Directory to store outputs (default: experiments)"
    echo "  --delay N        Delay between jobs in seconds (default: 0)"
    echo "  --dry-run       Print commands without executing"
}

# Default values
CONFIG_FILE="job-config.txt"
OUT_DIR="experiments"
DELAY=0
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output)
            OUT_DIR="$2"
            shift 2
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if config file is provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Config file is required"
    print_usage
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Function to merge parameters with defaults
merge_params() {
    local custom_params=$1
    local final_params=""
    
    # Start with default parameters
    for default_param in "${DEFAULT_PARAMS[@]}"; do
        param_name=$(echo "$default_param" | cut -d' ' -f1)
        # Only add default if not overridden in custom params
        if [[ ! "$custom_params" =~ $param_name ]]; then
            final_params="$final_params $default_param"
        fi
    done
    
    # Add custom parameters
    final_params="$final_params $custom_params"
    echo "$final_params"
}

# Read and process the config file
job_count=0
timestamp=$(date +%Y%m%d_%H%M%S)

echo "Starting job queue at $timestamp"
echo "Output directory: $OUT_DIR"
echo "Configuration file: $CONFIG_FILE"
echo "-------------------"

while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    if [[ -z "$line" ]] || [[ "$line" =~ ^#.* ]]; then
        continue
    fi

    # Extract job name if provided (format: [name] parameters)
    if [[ "$line" =~ ^\[(.*)\](.*) ]]; then
        job_name="${BASH_REMATCH[1]}"
        parameters="${BASH_REMATCH[2]}"
    else
        job_count=$((job_count + 1))
        job_name="job_${job_count}"
        parameters="$line"
    fi

    # Create job-specific output directory
    job_dir="${OUT_DIR}/${timestamp}_${job_name}"
    mkdir -p "$job_dir"

    # Merge default and custom parameters
    merged_params=$(merge_params "$parameters")

    # Construct the command
    cmd="python3 sd3_infer_illusions.py $merged_params --out_dir \"$job_dir\""

    # Log the command
    echo "Job $job_count: $job_name"
    echo "Command: $cmd"
    echo "Output directory: $job_dir"
    
    # Execute or simulate the command
    if [ "$DRY_RUN" = false ]; then
        eval "$cmd"
        echo "Job completed"
        
        # Add delay if specified
        if [ "$DELAY" -gt 0 ]; then
            echo "Waiting $DELAY seconds before next job..."
            sleep "$DELAY"
        fi
    else
        echo "[DRY RUN] Command would be executed"
    fi
    
    echo "-------------------"
done < "$CONFIG_FILE"

echo "All jobs completed" 
