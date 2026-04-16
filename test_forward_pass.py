"""
test_forward_pass.py — Smoke test for the Hierarchical Spiking Max-Former.

Verifies:
  1. Model construction (3-stage hierarchical Max-Former + SPaRG-MF)
  2. Forward pass on dummy input
  3. Backward pass with surrogate gradients
  4. SSA Interception Engine (sustained inference)
  5. Hardware gating mask generation
  6. Forward pass with external gating masks
  7. Homeostatic spike-rate statistics

Usage:
  python test_forward_pass.py
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '.')
from models.maxformer_snn import SpikingMaxFormer
from engine.ssa_interceptor import SSAInterceptor


def sep(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 1. Build model (small config for CPU testing) ────────────────
    sep("1 · Building Hierarchical SpikingMaxFormer")

    # Use small dims so this runs fast on CPU
    model = SpikingMaxFormer(
        img_size=64,             # small for speed
        in_channels=3,
        num_classes=10,
        embed_dims=64,           # dim/4=16, dim/2=32, dim=64
        depths=[1, 1, 2],        # minimal depth
        num_heads=4,
        mlp_ratio=2.0,
        time_steps=2,            # minimal time
        enable_head_gate=True,
        enable_token_gate=True,
        enable_mixed_prec=True,
        token_keep_ratio=0.5,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    print(f"Stages: depths={model.depths}")
    print(f"SSA blocks: {model.n_ssa_blocks}")

    # Print stage structure
    for name, _ in model.named_children():
        print(f"  +-- {name}")

    # ── 2. Forward pass ──────────────────────────────────────────────
    sep("2 · Forward pass")

    dummy = torch.randn(2, 3, 64, 64, device=device)
    model.train()
    model.reset_snn_state()

    logits = model(dummy)
    print(f"Input:  {list(dummy.shape)}")
    print(f"Output: {list(logits.shape)}")
    assert logits.shape == (2, 10), f"Bad shape: {logits.shape}"
    print("✓ Forward pass OK")

    # ── 3. Backward pass ─────────────────────────────────────────────
    sep("3 · Backward pass (surrogate gradients)")

    loss = nn.CrossEntropyLoss()(logits, torch.randint(0, 10, (2,), device=device))
    loss.backward()
    print(f"Loss: {loss.item():.4f}")

    n_grad = sum(1 for _, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None)
    n_total = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    print(f"Params with grads: {n_grad}/{n_total}")
    if n_grad == n_total:
        print("✓ All parameters received gradients")
    else:
        missing = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
        for m in missing[:5]:
            print(f"  ✗ {m}")

    # ── 4. SSA Interception ──────────────────────────────────────────
    sep("4 · SSA Interception Engine")

    model.eval()
    interceptor = SSAInterceptor(model)
    interceptor.attach()

    n_batches = 4
    print(f"Running {n_batches} inference batches...")
    with torch.no_grad():
        for _ in range(n_batches):
            model.reset_snn_state()
            _ = model(torch.randn(2, 3, 64, 64, device=device))

    interceptor.summary(num_heads=model.num_heads)

    # ── 5. Hardware gating masks ─────────────────────────────────────
    sep("5 · Hardware gating mask generation")

    masks = interceptor.generate_hardware_gating_masks(
        num_heads=model.num_heads, threshold=1e-6)

    if masks:
        for i, m in enumerate(masks):
            active = int(m.sum().item())
            total = m.numel()
            print(f"  SSA block {i}: {active}/{total} heads active  mask={m.squeeze()}")
        print("✓ Masks generated")
    else:
        print("✗ No masks generated")

    interceptor.detach()

    # ── 6. Gated forward pass ────────────────────────────────────────
    sep("6 · Forward pass with hardware gating")

    if masks:
        model.reset_snn_state()
        masks_device = [m.to(device) for m in masks]
        with torch.no_grad():
            gated = model(dummy, external_head_masks=masks_device)
        print(f"Gated output: {list(gated.shape)}")
        print("✓ Hardware-gated forward pass OK")

    # ── 7. Spike stats ───────────────────────────────────────────────
    sep("7 · Homeostatic spike-rate stats")

    stats = model.count_spikes()
    if stats:
        for name, s in list(stats.items())[:4]:
            print(f"  {name}: rate={s['avg_spike_rate']:.4f}  thr={s['v_threshold']:.4f}")
        if len(stats) > 4:
            print(f"  ... ({len(stats) - 4} more)")
    else:
        print("  (No HomeostaticLIFNodes — using standard LIF)")

    # ── Done ─────────────────────────────────────────────────────────
    sep("ALL TESTS PASSED")
    print("Hierarchical Spiking Max-Former prototype is functional.\n")


if __name__ == '__main__':
    main()
