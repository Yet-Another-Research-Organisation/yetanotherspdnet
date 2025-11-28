#!/bin/bash
# Grid search script 2: Batch size variation for SPDNet hyperparameter optimization
# Varies batch_size from 4 to 1024 (powers of 2)
# Runs experiments in parallel across 2 GPUs and collects F1 scores

set -e  # Exit on error

# Get script directory and navigate to experiments folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_DIR="$(dirname "$SCRIPT_DIR")/experiments"

# Configuration
RESULTS_DIR="$(dirname "$SCRIPT_DIR")/results/grid_search_batchsize_$(date +%Y%m%d_%H%M%S)"
# GPU configuration: adjust these to control concurrency per GPU
# GPU ids as visible to the system (usually 0 and 1)
GPU1_ID=0
GPU2_ID=1
# Maximum concurrent jobs per GPU
MAX_PARALLEL_GPU1=12
MAX_PARALLEL_GPU2=12
# Total max (computed)
MAX_PARALLEL=$((MAX_PARALLEL_GPU1 + MAX_PARALLEL_GPU2))

# Create results directory
mkdir -p "$RESULTS_DIR"

# Generate batch sizes (powers of 2 from 4 to 1024)
BATCH_SIZES=(64) 
# Seeds to test
SEEDS=(42) # 123 456 789)

# Batchnorm configurations: enabled with different methods, or disabled
# Format: "true:method" or "false"
BATCHNORM_CONFIGS=("true:geometric_arithmetic_harmonic" "false")

# Fixed parameters - TO BE FILLED
HIDDEN_SIZE_1=186  # TODO: Fill with best first layer size from grid search 1 (e.g., 94)
HIDDEN_SIZE_2=120  # Fixed second layer size
IMAGE_DIR="cov"  # TODO: Choose 'images' for raw data or 'cov' for precomputed covariances


# Function to run a single experiment
run_experiment() {
    local batch_size=$1
    local seed=$2
    local batchnorm_config=$3
    local exp_id=$4
    
    # Parse batchnorm configuration
    if [ "$batchnorm_config" = "false" ]; then
        local batchnorm="false"
        local batchnorm_method="geometric_arithmetic_harmonic"
        local bn_suffix="bnfalse"
    else
        local batchnorm="true"
        local batchnorm_method="${batchnorm_config#true:}"
        # Create short suffix for method
        case "$batchnorm_method" in
            "geometric_arithmetic_harmonic")
                local bn_suffix="bngah"
                ;;
            "arithmetic")
                local bn_suffix="bnarith"
                ;;
            "harmonic")
                local bn_suffix="bnharm"
                ;;
            "log_euclidean")
                local bn_suffix="bnlogeuc"
                ;;
            "affine_invariant")
                local bn_suffix="bnaffinv"
                ;;
            *)
                local bn_suffix="bn${batchnorm_method}"
                ;;
        esac
    fi
    
    local exp_name="bs${batch_size}_s${seed}_${bn_suffix}"
    local exp_dir="$RESULTS_DIR/$exp_name"
    local config_file="$exp_dir/config.yaml"
    
    echo "[$exp_id] Starting experiment: $exp_name"
    
    # Create experiment directory
    mkdir -p "$exp_dir"
    
    # Create modified config file
    cat > "$config_file" << EOF
# Auto-generated config for grid search - Batch size variation
# Batch size: $batch_size, Seed: $seed, Batchnorm: $batchnorm

epochs: 2
lr: 0.0005
optimizer: adam
scheduler: plateau
scheduler_patience: 4
scheduler_factor: 0.5
scheduler_min_lr: 1.0e-06
early_stopping_patience: 15
storage_path: $exp_dir
group_id: grid_search_batchsize
save_best_model: true
device: cuda
dtype: float64
seed: $seed
verbose: false
preload: false
skip_label_check: true

dataset_config:
  name: hyperleaf
  path: ~/Documents/DATASET/HyperLeaf2024
  val_ratio: 0.1
  test_ratio: 0.2
  batch_size: $batch_size
  shuffle: true
  max_samples: 50
  num_workers: 4
  train_transform: None
  test_transform: None
  task_type: classification
  load_metadata: false
  images_dir: $IMAGE_DIR
  target_size:
  - 204
  - 204

model_name: spdnet
model_config:
  input_dim: 204
  input_channels: 1
  hidden_layers_size:
  - $HIDDEN_SIZE_1
  - $HIDDEN_SIZE_2
  eps: 0.6
  batchnorm: $batchnorm
  batchnorm_method: $batchnorm_method
  softmax: false
  use_autograd: false
  dropout_rate: 0.0
  use_vech: false
EOF
    
    # Run training
    local log_file="$exp_dir/training.log"
    local timestamp=$(date +%Y-%m-%d_%H:%M:%S)
    
    # Activate virtual environment and run training
    if source ~/Documents/CODE/batchnorm_aaai/.venv/bin/activate && \
       cd "$EXPERIMENTS_DIR" && \
       python train_model.py --config "$config_file" > "$log_file" 2>&1; then
        echo "[$exp_id] SUCCESS: $exp_name"
    else
        echo "[$exp_id] FAILED: $exp_name (check $log_file)"
    fi
}

# Export function for parallel execution
export -f run_experiment
export RESULTS_DIR
export HIDDEN_SIZE_1
export HIDDEN_SIZE_2
export IMAGE_DIR
export EXPERIMENTS_DIR

# Generate all experiment combinations
exp_counter=0
experiments=()

for batch_size in "${BATCH_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for batchnorm in "${BATCHNORM_CONFIGS[@]}"; do
            exp_counter=$((exp_counter + 1))
            experiments+=("$batch_size $seed $batchnorm $exp_counter")
        done
    done
done

total_experiments=${#experiments[@]}
echo "========================================"
echo "Grid Search 2: Batch Size Variation"
echo "========================================"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Batchnorm configs: ${BATCHNORM_CONFIGS[*]}"
echo "Fixed hidden layers: [$HIDDEN_SIZE_1, $HIDDEN_SIZE_2]"
echo "Images directory: $IMAGE_DIR"
echo "Total experiments: $total_experiments"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "Results directory: $RESULTS_DIR"
echo "========================================"
echo ""

echo "Running experiments in parallel across GPUs (GPU $GPU1_ID: $MAX_PARALLEL_GPU1 jobs, GPU $GPU2_ID: $MAX_PARALLEL_GPU2 jobs)..."

# Per-GPU tracking
running_gpu1=0
running_gpu2=0
pids_gpu1=()
pids_gpu2=()

cleanup_finished() {
    # Remove finished PIDs from pids_gpu1 and pids_gpu2 and update counters
    local new_pids=()
    for pid in "${pids_gpu1[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
        else
            running_gpu1=$((running_gpu1 - 1))
        fi
    done
    pids_gpu1=("${new_pids[@]}")

    new_pids=()
    for pid in "${pids_gpu2[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
        else
            running_gpu2=$((running_gpu2 - 1))
        fi
    done
    pids_gpu2=("${new_pids[@]}")
}

for exp in "${experiments[@]}"; do
    read -r batch_size seed batchnorm exp_id <<< "$exp"

    # Wait until one GPU has capacity
    while true; do
        cleanup_finished

        if [ $running_gpu1 -lt $MAX_PARALLEL_GPU1 ]; then
            target_gpu=1
            break
        elif [ $running_gpu2 -lt $MAX_PARALLEL_GPU2 ]; then
            target_gpu=2
            break
        else
            # No capacity: wait a bit and try again (reap finished jobs)
            sleep 1
        fi
    done

    # Launch on chosen GPU by setting CUDA_VISIBLE_DEVICES for the process
    if [ "$target_gpu" -eq 1 ]; then
        CUDA_VISIBLE_DEVICES="$GPU1_ID" run_experiment "$batch_size" "$seed" "$batchnorm" "$exp_id" &
        pid=$!
        pids_gpu1+=("$pid")
        running_gpu1=$((running_gpu1 + 1))
        echo "[$exp_id] Launched on GPU $GPU1_ID (pid=$pid) - running_gpu1=$running_gpu1"
    else
        CUDA_VISIBLE_DEVICES="$GPU2_ID" run_experiment "$batch_size" "$seed" "$batchnorm" "$exp_id" &
        pid=$!
        pids_gpu2+=("$pid")
        running_gpu2=$((running_gpu2 + 1))
        echo "[$exp_id] Launched on GPU $GPU2_ID (pid=$pid) - running_gpu2=$running_gpu2"
    fi
done

# Wait for all remaining jobs to finish (both GPUs)
echo "Waiting for remaining jobs to finish..."
while true; do
    cleanup_finished
    if [ ${#pids_gpu1[@]} -eq 0 ] && [ ${#pids_gpu2[@]} -eq 0 ]; then
        break
    fi
    sleep 1
done

echo ""
echo "========================================"
echo "Grid Search Complete!"
echo "========================================"
echo "Results saved to: $RESULTS_DIR"
echo ""

# Extract and analyze results
echo "Extracting results from experiment directories..."
if [ -f "$EXPERIMENTS_DIR/extract_grid_results.py" ]; then
    source ~/Documents/CODE/batchnorm_aaai/.venv/bin/activate
    python "$EXPERIMENTS_DIR/extract_grid_results.py" "$RESULTS_DIR"
else
    echo "Warning: extract_grid_results.py not found at $EXPERIMENTS_DIR/extract_grid_results.py"
fi

echo ""
echo "Done!"
