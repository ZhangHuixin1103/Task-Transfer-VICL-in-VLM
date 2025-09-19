#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
export CUDA_VISIBLE_DEVICES=1,2,3,4         # Specify GPUs to use
NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')  # Automatically detects available GPUs

# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"    # [ModelArguments] Pretrained model path
OUTPUT_DIR="./output"                       # Directory for saving checkpoints
CACHE_DIR="../../.cache"                    # [TrainingArguments] Cache directory for models
export HF_HOME=$CACHE_DIR                   # Set HF cache directory

# ======================
# Model Configuration
# ======================
DATASETS="VICL%100"                         # [DataArguments] Dataset with sampling rate

# ======================
# Training Hyperparameters
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision True \
         --tune_mm_mlp True \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 4 \
         --gradient_accumulation_steps 4 \
         --learning_rate 5e-7 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 8192 \
         --data_flatten True \
         --data_packing True \
         --max_pixels 1605632 \
         --min_pixels 12544 \
         --num_train_epochs 3 \
         --logging_steps 10 \
         --save_steps 1000 \
         --save_total_limit 3 \
         --deepspeed ./scripts/zero3.json
