#!/usr/bin/env python3
"""MIMIC-only ablation for BME 507 Practice 1 Step VII(c).

Run from /workspace/base using ./gaudi.sh. Initialize the two MIMIC
networks from the shared weights of the best VinDr-17 teacher. Train
only on MIMIC, validate student and teacher after every epoch.
"""
import argparse
import csv
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import MIMIC, build_transform_classification
from federated_vii_c import (latest_teacher, load_shared, local_epoch,
                             make_model, mean_auc, shared_only, cpu_state)
from utils import get_config


def save_plot(csv_path, plot_path):
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    xs = [int(row['epoch']) for row in rows]
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, [float(row['student_mAUC']) for row in rows], marker='o', label='Student')
    plt.plot(xs, [float(row['teacher_mAUC']) for row in rows], marker='o', label='Teacher')
    plt.xlabel('MIMIC-only training epoch')
    plt.ylabel('MIMIC validation mAUC')
    plt.title('MIMIC-only Ark+ ablation')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=1, help='Total epochs to reach')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 0.9 <= args.momentum < 1.0:
        parser.error('--momentum must be in [0.9, 1.0)')

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    device = torch.device('hpu')
    cfg = get_config('datasets_config.yaml')

    train_trans = build_transform_classification(normalize='imagenet', crop_size=224,
                                                  resize=256, mode='valid')
    val_trans = build_transform_classification(normalize='imagenet', crop_size=224,
                                                resize=256, mode='test', test_augment=False)
    mimic_cfg = cfg['MIMIC']
    train_data = MIMIC(images_path=mimic_cfg['data_dir'],
                       file_path=mimic_cfg['train_list'], crop_size=224,
                       resize=256, augment=train_trans)
    val_data = MIMIC(images_path=mimic_cfg['data_dir'],
                     file_path=mimic_cfg['val_list'], crop_size=224,
                     resize=256, augment=val_trans)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)
    print(f'MIMIC examples: train={len(train_data)} validation={len(val_data)}', flush=True)

    teacher_file, vindr_cycle = latest_teacher(args.init_dir)
    base = torch.load(teacher_file, map_location='cpu', weights_only=False)
    if isinstance(base, dict) and 'teacher' in base:
        base = base['teacher']
    elif isinstance(base, dict) and 'state_dict' in base:
        base = base['state_dict']
    student = make_model(14, device)
    teacher = make_model(14, device)
    load_shared(student, shared_only(base))
    teacher.load_state_dict(student.state_dict(), strict=True)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    checkpoint_path = os.path.join(args.output_dir, 'latest.pt')
    csv_path = os.path.join(args.output_dir, 'validation_mauc.csv')
    plot_path = os.path.join(args.output_dir, 'mimic_mauc.png')

    first_epoch = 0
    if args.resume and os.path.isfile(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        student.load_state_dict(ck['student'], strict=True)
        teacher.load_state_dict(ck['teacher'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        first_epoch = int(ck['epoch'])
        print(f'Resumed from completed MIMIC epoch {first_epoch}', flush=True)
    else:
        print(f'Initializing from VinDr teacher cycle {vindr_cycle}: {teacher_file}', flush=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'student_mAUC', 'teacher_mAUC'])
        # A randomly initialized 14-label head is evaluated only on MIMIC labels.
        baseline_student = mean_auc(student, val_loader, device)
        baseline_teacher = mean_auc(teacher, val_loader, device)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([0, baseline_student, baseline_teacher])
        print(f'Epoch 0 MIMIC val student_mAUC={baseline_student:.6f} teacher_mAUC={baseline_teacher:.6f}', flush=True)
        save_plot(csv_path, plot_path)

    if first_epoch >= args.epochs:
        print('Already reached target epochs.', flush=True)
        return

    for epoch in range(first_epoch + 1, args.epochs + 1):
        print(f'===== MIMIC-ONLY EPOCH {epoch}/{args.epochs} =====', flush=True)
        local_epoch(student, teacher, train_loader, optimizer, args.momentum, device, 'MIMIC')
        student_auc = mean_auc(student, val_loader, device)
        teacher_auc = mean_auc(teacher, val_loader, device)
        print(f'Epoch {epoch} MIMIC val student_mAUC={student_auc:.6f} teacher_mAUC={teacher_auc:.6f}', flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, student_auc, teacher_auc])
        torch.save({'epoch': epoch, 'init_vindr_cycle': vindr_cycle,
                    'student': cpu_state(student), 'teacher': cpu_state(teacher),
                    'optimizer': optimizer.state_dict()}, checkpoint_path)
        save_plot(csv_path, plot_path)
        print(f'Saved checkpoint {checkpoint_path} and plot {plot_path}', flush=True)
    print('MIMIC-only ablation run complete.', flush=True)


if __name__ == '__main__':
    main()
