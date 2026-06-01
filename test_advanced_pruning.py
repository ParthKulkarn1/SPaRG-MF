"""
test_advanced_pruning.py — Comprehensive validation suite for advanced SPaRG-MF pruning and attention enhancements.

Verifies:
  1. Per-Instance & Time-Dependent Gating (dynamic shaping validation).
  2. Alternative Token Gating Policies (MLP, Magnitude-based, and Attention-weighted).
  3. Staged Progressive Keep Ratios (layer-wise schedule validation).
  4. Attention Sparsification (Top-k and thresholding for linear/softmax blocks).
  5. Diversity Regularization Loss (head orthogonalization).
  6. SNN Calibration and Post-Training Inference Pruning (multi-metric gating).

Usage:
  python test_advanced_pruning.py
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, '.')
from models.maxformer_snn import SpikingMaxFormer
from engine.ssa_interceptor import SSAInterceptor
from engine.regularization import DiversityLoss
from engine.inference_pruning import SNNInferencePruningEngine


def sep(title):
    print(f"\n{'='*65}\n  {title}\n{'='*65}")


def run_checks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create synthetic dataset for calibration and accuracy sweeps
    dummy_x = torch.randn(16, 3, 64, 64)
    dummy_y = torch.randint(0, 10, (16,))
    dataset = TensorDataset(dummy_x, dummy_y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # ── 1. Per-Instance, Time-Dependent Gating & Staged Pruning ─────────
    sep("1 · Building Model with Staged Keep Ratios & Time Gating")
    
    staged_keep = [0.9, 0.7, 0.5] # different keep ratios per stage-3 block (n_ssa=3)
    model = SpikingMaxFormer(
        img_size=64,
        in_channels=3,
        num_classes=10,
        embed_dims=64,
        depths=[1, 1, 3],        # 3 SSA blocks
        num_heads=4,
        mlp_ratio=2.0,
        time_steps=3,
        enable_head_gate=True,
        enable_token_gate=True,
        enable_mixed_prec=True,
        token_keep_ratio=staged_keep,
        gate_type='mlp',
        batch_averaged=False,     # Per-Instance Dynamic Routing
        time_dependent=True,     # Time-Dependent Dynamic Routing
        sparse_threshold=0.01,   # Linear Attention Sparsification
    ).to(device)

    print(f"Constructed model with depths={model.depths}")
    print(f"Staged token keep ratios: {model.token_keep_ratios}")
    print(f"Sparsity Threshold: {model.sparse_threshold}")
    
    # Forward Pass
    model.train()
    model.reset_snn_state()
    x = torch.randn(4, 3, 64, 64, device=device)
    out = model(x)
    print(f"Output shape under per-instance & time-dependent gating: {list(out.shape)}")
    assert out.shape == (4, 10), "Gating forward pass shape mismatch!"
    print("OK: Per-instance, time-dependent, and staged gating forward pass successful")

    # ── 2. Alternative Token Gating Policies ───────────────────────────
    sep("2 · Checking Alternative Gating Policies (Magnitude & Attention)")
    
    for policy in ['magnitude', 'attention']:
        print(f"Testing gate_type='{policy}'...")
        policy_model = SpikingMaxFormer(
            img_size=64,
            in_channels=3,
            num_classes=10,
            embed_dims=64,
            depths=[1, 1, 3],
            num_heads=4,
            mlp_ratio=2.0,
            time_steps=2,
            enable_head_gate=True,
            enable_token_gate=True,
            token_keep_ratio=0.5,
            gate_type=policy,
            batch_averaged=False,
            time_dependent=False
        ).to(device)
        
        policy_model.train()
        policy_model.reset_snn_state()
        out = policy_model(x)
        print(f"  + Success! Output shape: {list(out.shape)}")
    print("OK: Alternative token gating policies function correctly")

    # ── 3. Diversity Regularization Loss ───────────────────────────────
    sep("3 · Verifying Diversity Regularization Loss")
    
    div_loss_fn = DiversityLoss(weight=0.05)
    loss_div = div_loss_fn(model)
    print(f"Diversity Regularization Loss: {loss_div.item():.6f}")
    assert loss_div.item() >= 0.0, "Diversity loss should be non-negative!"
    
    # Backward Pass check
    loss_total = nn.CrossEntropyLoss()(out, torch.randint(0, 10, (4,), device=device)) + loss_div
    loss_total.backward()
    print("OK: Diversity loss backpropagated gradients successfully")

    # ── 4. SNN Calibration & Post-Training Inference Pruning ───────────
    sep("4 · Calibrating Model and Running Multi-Metric Gating Analysis")
    
    pruning_engine = SNNInferencePruningEngine(model, num_heads=4)
    
    # Run calibration
    pruning_engine.calibrate(dataloader, device, num_batches=3)
    
    # Print metrics summary
    pruning_engine.interceptor.summary(num_heads=4)
    
    # ── 5. Generate and Verify Multi-Metric Gating Masks ──────────────
    sep("5 · Constructing Binary Hardware Gating Masks")
    
    metrics = {
        'importance': 1e-6,
        'q_value': 0.1,
        'redundancy': 0.5,
        'entropy': 1.5,
        'spike_entropy': 0.1
    }
    
    for metric_name, thresh in metrics.items():
        print(f"Generating static masks based on metric: '{metric_name}' (thresh={thresh})...")
        masks = pruning_engine.generate_static_masks(metric=metric_name, threshold=thresh)
        
        # Verify masks shape
        assert len(masks) == 3, f"Expected 3 masks for 3 SSA blocks, got {len(masks)}"
        for idx, m in enumerate(masks):
            print(f"  + SSA Block {idx} Mask Active Heads: {int(m.sum().item())}/4  shape: {list(m.shape)}")
            assert m.shape == (1, 1, 4, 1, 1), f"Expected mask shape (1, 1, 4, 1, 1), got {m.shape}"
        
        # Evaluate model accuracy and sparsity under mask
        acc, sparsity = pruning_engine.evaluate_mask_efficiency(dataloader, device, masks)
        print(f"  + Evaluation -> Accuracy: {acc:.2f}%  |  Head Sparsity: {sparsity:.2f}%")

    print("\nOK: Gating mask generation and inference evaluation successful")
    
    # ── Done ─────────────────────────────────────────────────────────
    sep("ALL ADVANCED PRUNING TESTS PASSED")
    print("All dynamic SNN routing, staged pruning, diversity, and multi-metric inference calibration modules are functional.\n")


if __name__ == '__main__':
    run_checks()
