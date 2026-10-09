#!/usr/bin/env bash
set -euo pipefail
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
export MASTER_ADDR MASTER_PORT
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -lc 'cd /scratch/jcuesta/BME/2026Fall-CSE507-Ark-Container && ./nvidia.sh python -m torch.distributed.run --nnodes=4 --nproc-per-node=1 --node-rank=$SLURM_PROCID --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT main_ark.py --opt sgd --warmup-epochs 20 --batch_size 200 --model swin_base --init imagenet --pretrain_epochs 200 --pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth --momentum_teacher 0.9 --projector_features 1376 --ema_mode epoch --device cuda --exp_name mimic_only_centercrop_1gpu --seed 100 --resume true --data_set MIMIC --test_augment false'
