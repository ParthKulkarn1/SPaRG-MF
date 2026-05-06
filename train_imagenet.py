"""
train_imagenet.py — Distributed ImageNet-1K Training Pipeline for SPaRG-MF.

This script leverages PyTorch DDP and the `timm` library to train the
Spiking Max-Former on ImageNet-1K. It heavily utilizes TIMM's data loaders,
MixUp/CutMix implementations, and metrics.

Usage (Single Node, multi-GPU):
    torchrun --nproc_per_node=8 train_imagenet.py \
        --data_dir /path/to/imagenet/ \
        --model sparg_mf_10_384 \
        --batch_size 64 \
        --epochs 300

Usage (Single GPU / Colab):
    python train_imagenet.py --data_dir /path/to/data --epochs 10
"""

import argparse
import time
import os
import torch
import torch.nn as nn
from pathlib import Path

# TIMM imports
from timm.models import create_model
from timm.data import create_dataset, create_loader
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer_v2
from timm.utils import NativeScaler, get_state_dict, ModelEmaV2

# SpikingJelly functional for network state reset
from spikingjelly.activation_based import functional

# SPaRG-MF Model Registration
import models.maxformer_snn  # Automatically executes the @register_model decorators


def get_args_parser():
    parser = argparse.ArgumentParser('SPaRG-MF ImageNet Training', add_help=False)
    # Model parameters
    parser.add_argument('--model', default='sparg_mf_10_384', type=str, metavar='MODEL')
    parser.add_argument('--time_steps', default=4, type=int)
    # Standard training params
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, help='Manual epoch override (useful for restarts)')
    parser.add_argument('--data_dir', default='', type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='./output', type=str)
    # Optimizer/Scheduler (TIMM params)
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--sched', default='cosine', type=str)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_lr', type=float, default=1e-6)
    # Augmentation (Mixup/Cutmix)
    parser.add_argument('--mixup', type=float, default=0.8)
    parser.add_argument('--cutmix', type=float, default=1.0)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--workers', default=8, type=int)
    # EMA
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable Exponential Moving Average of model weights')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='EMA decay rate (default: 0.9998)')
    # Checkpoint resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to checkpoint to resume from (e.g. output/checkpoint.pth)')
    # Distributed
    parser.add_argument('--local_rank', default=0, type=int)
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # 1. Distributed Training Setup
    distributed = False
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda', args.gpu)
    else:
        device = torch.device('cpu')

    if distributed:
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://',
            world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        print(f"========== Initiating SPaRG-MF Training ==========")
        print(f"Model: {args.model} | Time steps: {args.time_steps} | Batch: {args.batch_size}")
        print(f"Distributed: {distributed} | Device: {device} | EMA: {args.model_ema}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 2. Build Model using TIMM
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        time_steps=args.time_steps,
    )
    model.to(device)

    if args.rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f} M")

    # ── EMA setup (before DDP wrapping) ──
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay)
        if args.rank == 0:
            print(f"  [EMA] Enabled with decay={args.model_ema_decay}")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module if distributed else model

    # 3. Create Dataset and DataLoaders
    dataset_train = create_dataset('imagenet', root=args.data_dir, split='train', is_training=True)
    dataset_eval = create_dataset('imagenet', root=args.data_dir, split='validation', is_training=False)

    sampler_train = None
    if distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)

    train_loader = create_loader(
        dataset_train,
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=True,
        no_aug=False,
        num_workers=args.workers,
        distributed=distributed,
        pin_memory=True,
    )

    eval_loader = create_loader(
        dataset_eval,
        input_size=(3, 224, 224),
        batch_size=int(args.batch_size * 1.5),
        is_training=False,
        use_prefetcher=True,
        num_workers=args.workers,
        distributed=distributed,
        crop_pct=0.875,
        pin_memory=True,
    )

    # 4. MixUp & Loss Setup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0
    if mixup_active:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, label_smoothing=args.smoothing, num_classes=1000)
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # 5. Optimizer, Scheduler, Amp
    optimizer = create_optimizer_v2(model_without_ddp, opt=args.opt, lr=args.lr, weight_decay=args.weight_decay)
    scheduler, _ = create_scheduler(args, optimizer)
    loss_scaler = NativeScaler()

    # ══════════════════════════════════════════════════════════════════
    #  6. Resume from checkpoint
    # ══════════════════════════════════════════════════════════════════
    best_acc1 = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            if args.rank == 0:
                print(f"  [RESUME] Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')

            model_without_ddp.load_state_dict(checkpoint['state_dict'])

            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc1' in checkpoint:
                best_acc1 = checkpoint['best_acc1']
            if model_ema is not None and 'model_ema' in checkpoint:
                model_ema.module.load_state_dict(checkpoint['model_ema'])

            if args.rank == 0:
                print(f"  [RESUME] Loaded checkpoint at epoch {checkpoint.get('epoch', '?')}, "
                      f"best_acc1={best_acc1:.2f}%")
        else:
            if args.rank == 0:
                print(f"  [RESUME] WARNING: No file found at '{args.resume}' — training from scratch.")

    # ══════════════════════════════════════════════════════════════════
    #  7. Primary Training Loop
    # ══════════════════════════════════════════════════════════════════
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        n_steps = 0

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                inputs, targets = mixup_fn(inputs, targets)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            loss_scaler(loss, optimizer, parameters=model.parameters())
            optimizer.zero_grad()
            
            # CLEAR SPIKE BUFFERS
            functional.reset_net(model)

            # ── EMA update ──
            if model_ema is not None:
                model_ema.update(model_without_ddp)

            train_loss += loss.item()
            n_steps += 1

            if args.rank == 0 and step % 100 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f}")

        scheduler.step(epoch)
        avg_train_loss = train_loss / max(n_steps, 1)

        # ══════════════════════════════════════════════════════════════
        #  8. Evaluation Phase
        # ══════════════════════════════════════════════════════════════
        # Evaluate both the base model and the EMA model (if enabled)
        eval_model = model_ema.module if model_ema is not None else model
        eval_model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = eval_model(inputs)

                functional.reset_net(eval_model)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Gather eval results across processes
        if distributed:
            acc_tensor = torch.tensor([correct, total], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(acc_tensor, op=torch.distributed.ReduceOp.SUM)
            acc1 = (acc_tensor[0] / acc_tensor[1]) * 100.0
            acc1 = acc1.item()
        else:
            acc1 = (correct / max(total, 1)) * 100.0

        if args.rank == 0:
            elapsed = time.time() - start_time
            print(f">>> Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Top-1 Acc: {acc1:.2f}% | "
                  f"Best: {max(best_acc1, acc1):.2f}% | "
                  f"Time: {elapsed/3600:.1f}h")
            
            # Print SNN Stats
            ref_model = model_ema.module if model_ema is not None else model_without_ddp
            stats = ref_model.count_spikes()
            if stats:
                print("  [Homeostatic Spiking Rates]")
                rts = []
                for k, v in list(stats.items())[:3] + list(stats.items())[-3:]:
                    rts.append(f"{k.split('.')[-1]}: {v['avg_spike_rate']:.3f}")
                print("    " + ", ".join(rts))

            # ── Save latest checkpoint ──
            state = {
                'epoch': epoch,
                'state_dict': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': loss_scaler.state_dict(),
                'acc1': acc1,
                'best_acc1': max(best_acc1, acc1),
            }
            if model_ema is not None:
                state['model_ema'] = model_ema.module.state_dict()

            torch.save(state, os.path.join(args.output_dir, 'checkpoint.pth'))

            # ── Save best model ──
            if acc1 > best_acc1:
                best_acc1 = acc1
                torch.save(state, os.path.join(args.output_dir, 'model_best.pth'))
                print(f"  ★ New best model saved! Acc: {best_acc1:.2f}%")
            else:
                best_acc1 = max(best_acc1, acc1)

    if args.rank == 0:
        total_time = time.time() - start_time
        print(f"\n========== Training Complete ==========")
        print(f"Total time: {total_time/3600:.2f} hours | Best Top-1: {best_acc1:.2f}%")
            
if __name__ == '__main__':
    main()
