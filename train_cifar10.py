"""
train_cifar10.py — Basic training loop for SPaRG-MF on CIFAR-10.

Demonstrates:
  - Dataloader setup (CIFAR-10, torchvision)
  - SpikingMaxFormer initialization
  - Integrating functional.reset_net(model) in the training loop
  - Standard classification training loop with AdamW + CosineAnnealing
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import functional

# SPaRG-MF Imports
from models.maxformer_snn import SpikingMaxFormer


def get_cifar10_loaders(batch_size=128, data_dir='./data'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    return train_loader, test_loader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Hyperparameters
    batch_size = 64
    epochs = 2 
    learning_rate = 1e-3

    # Load data
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    # Initialize a lightweight configuration of our SPaRG-MF model for CIFAR-10
    # Image size is 32x32, so patch embedding downsamples to 8x8.
    model = SpikingMaxFormer(
        img_size=32,            
        in_channels=3,
        num_classes=10,
        embed_dims=128,         
        depths=[1, 1, 3],       # 1 block stage1, 1 block stage2, 3 blocks stage 3 (SSA)
        num_heads=4,
        time_steps=2,           # Keep time_steps small for fast test
        enable_head_gate=True,
        enable_token_gate=False, # Disable token gating for tiny 32x32 images
        enable_mixed_prec=False, 
    ).to(device)

    # Setup Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Simple Train Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass 
            loss.backward()
            optimizer.step()
            
            # MANDATORY: Reset SNN states between batches!
            functional.reset_net(model)

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        scheduler.step()
        
        train_loss = train_loss / total
        train_acc = 100. * correct / total
        
        # Test Loop
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                functional.reset_net(model)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        test_loss = test_loss / total
        test_acc = 100. * correct / total
        
        print(f"Epoch {epoch} | Time: {time.time()-start_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")
              
        # Print an example of Homeostatic Spike Statistics
        stats = model.count_spikes()
        if stats:
             print("  Homeostatic Rates -> " + ", ".join([f"{k.split('.')[-1]}: {v['avg_spike_rate']:.3f}" for k, v in list(stats.items())[:3]]))

if __name__ == '__main__':
    main()
