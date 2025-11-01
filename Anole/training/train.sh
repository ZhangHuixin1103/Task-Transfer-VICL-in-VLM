CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1234 finetune.py --initial_model ../../.cache/anole-7b  --trained_model ../../.cache/anole-7b-trained
