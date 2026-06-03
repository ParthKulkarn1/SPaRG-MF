"""Generates colab_training.ipynb with proper nbformat structure for VS Code."""
import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
}

def clean_src(text):
    # VS Code prefers a list of lines
    lines = text.split('\n')
    return [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

def md(src):
    return nbformat.v4.new_markdown_cell(clean_src(src))

def code(src):
    return nbformat.v4.new_code_cell(clean_src(src))

# ── Cell definitions ─────────────────────────────────────────────────

cell_header = md(
    "# SPaRG-MF: Spiking Max-Former - CIFAR-100 Training\n"
    "\n"
    "This notebook trains **SPaRG-MF** on **CIFAR-100**.\n"
    "\n"
    "**Setup:** Go to `Runtime > Change runtime type > GPU` (T4 or better), then run all cells."
)

cell_clone_md = md("## 1. Get Code from GitHub")

cell_clone = code(
    'import os\n'
    '\n'
    '# Clone the repository if it doesn\'t exist and we are in Colab\n'
    'if not os.path.isdir("SPaRG-MF") and os.path.isdir("/content"):\n'
    '    !git clone https://github.com/ParthKulkarn1/SPaRG-MF.git\n'
    '\n'
    'if os.path.isdir("/content/SPaRG-MF"):\n'
    '    PROJECT_DIR = "/content/SPaRG-MF"\n'
    '    os.chdir(PROJECT_DIR)\n'
    'else:\n'
    '    PROJECT_DIR = os.getcwd()\n'
    'print(f"Working directory: {PROJECT_DIR}")'
)

cell_drive_md = md(
    "## 2. (Optional) Mount Google Drive for Checkpoints\n"
    "\n"
    "Colab deletes all files when the runtime disconnects. If you want your checkpoints to survive, mount Google Drive."
)

cell_drive = code(
    'OUTPUT_DIR = "/content/SPaRG-MF/output"\n'
    '\n'
    '# Uncomment the following lines to save checkpoints to your Google Drive\n'
    '# from google.colab import drive\n'
    '# drive.mount("/content/drive")\n'
    '# OUTPUT_DIR = "/content/drive/MyDrive/SPaRG-MF-Checkpoints"\n'
    '\n'
    'os.makedirs(OUTPUT_DIR, exist_ok=True)\n'
    'print(f"Checkpoints will be saved to: {OUTPUT_DIR}")'
)

cell_deps_md = md("## 3. Install Dependencies")

cell_deps = code(
    '!pip install -q "timm>=0.9.0" "spikingjelly>=0.0.0.0.14" einops\n'
    '\n'
    'import torch\n'
    'print(f"PyTorch: {torch.__version__}")\n'
    'print(f"CUDA: {torch.cuda.is_available()}")\n'
    'if torch.cuda.is_available():\n'
    '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n'
    '    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")'
)

cell_verify_md = md("## 4. Verify Model Builds")

cell_verify = code(
    'import sys\n'
    'sys.path.insert(0, PROJECT_DIR)\n'
    '\n'
    'from models.maxformer_snn import SpikingMaxFormer\n'
    'from spikingjelly.activation_based import functional\n'
    '\n'
    'import torch\n'
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
    'test_model = SpikingMaxFormer(\n'
    '    img_size=32, in_channels=3, num_classes=100,\n'
    '    embed_dims=128, depths=[1, 1, 3], num_heads=4,\n'
    '    time_steps=4, enable_head_gate=True,\n'
    '    enable_token_gate=True, enable_mixed_prec=True,\n'
    '    token_keep_ratio=[0.95, 0.90, 0.85], gate_type="magnitude",\n'
    '    batch_averaged=False, time_dependent=True,\n'
    ').to(device)\n'
    '\n'
    'params = sum(p.numel() for p in test_model.parameters())\n'
    'print(f"Model built! Parameters: {params / 1e6:.2f} M")\n'
    '\n'
    'dummy = torch.randn(2, 3, 32, 32).to(device)\n'
    'with torch.no_grad():\n'
    '    out = test_model(dummy)\n'
    '    functional.reset_net(test_model)\n'
    'print(f"Forward pass OK! {dummy.shape} -> {out.shape}")\n'
    '\n'
    'del test_model, dummy, out\n'
    'if torch.cuda.is_available():\n'
    '    torch.cuda.empty_cache()'
)

cell_config_md = md("## 5. Training Configuration")

cell_config = code(
    'BATCH_SIZE = 64\n'
    'EPOCHS = 100\n'
    'LR = 1e-3\n'
    'WEIGHT_DECAY = 1e-4\n'
    'TIME_STEPS = 4\n'
    'EMBED_DIMS = 384\n'
    'DEPTHS = [1, 1, 3]\n'
    'NUM_HEADS = 4\n'
    'NUM_WORKERS = 2'
)

cell_data_md = md("## 6. Load CIFAR-100 Dataset")

cell_data = code(
    'import torch\n'
    'from torch.utils.data import DataLoader\n'
    'from torchvision import datasets, transforms\n'
    '\n'
    'transform_train = transforms.Compose([\n'
    '    transforms.RandomCrop(32, padding=4),\n'
    '    transforms.RandomHorizontalFlip(),\n'
    '    transforms.ToTensor(),\n'
    '    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),\n'
    '])\n'
    'transform_test = transforms.Compose([\n'
    '    transforms.ToTensor(),\n'
    '    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),\n'
    '])\n'
    '\n'
    'train_set = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)\n'
    'test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)\n'
    'train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)\n'
    'test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)\n'
    'print(f"CIFAR-100: {len(train_set)} train / {len(test_set)} test")'
)

cell_model_md = md("## 7. Build Model")

cell_model = code(
    'import sys\n'
    'sys.path.insert(0, PROJECT_DIR)\n'
    '\n'
    'import torch.nn as nn\n'
    'from models.maxformer_snn import SpikingMaxFormer\n'
    'from spikingjelly.activation_based import functional\n'
    'from engine.regularization import DiversityLoss\n'
    '\n'
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
    '\n'
    'model = SpikingMaxFormer(\n'
    '    img_size=32, in_channels=3, num_classes=100,\n'
    '    embed_dims=EMBED_DIMS, depths=DEPTHS, num_heads=NUM_HEADS,\n'
    '    time_steps=TIME_STEPS, mlp_ratio=4.0,\n'
    '    enable_head_gate=True, enable_token_gate=False, enable_mixed_prec=False,\n'
    ').to(device)\n'
    '\n'
    'diversity_loss_fn = DiversityLoss(weight=0.001)\n'
    '\n'
    'from timm.data import Mixup\n'
    'mixup_fn = Mixup(\n'
    '    mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,\n'
    '    mode="batch", label_smoothing=0.1, num_classes=100\n'
    ')\n'
    '\n'
    'params = sum(p.numel() for p in model.parameters())\n'
    'print(f"Model on {device} | Parameters: {params / 1e6:.2f} M")\n'
    '\n'
    'optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)\n'
    'criterion = nn.CrossEntropyLoss()\n'
    '\n'
    '# Sequential Scheduler with 5-epoch Warmup\n'
    'warmup_epochs = 5\n'
    'scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)\n'
    'scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs)\n'
    'scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])'
)

cell_check_md = md(
    "## 8. Checkpoint Manager\n"
    "This cell checks for existing checkpoints. It will automatically load them and prepare the training loop to resume from where it left off."
)

cell_check = code(
    'import os\n'
    '\n'
    'start_epoch = 1\n'
    'best_acc = 0.0\n'
    'ckpt_path = os.path.join(OUTPUT_DIR, "cifar100_checkpoint.pth")\n'
    'best_path = os.path.join(OUTPUT_DIR, "cifar100_best.pth")\n'
    '\n'
    'print("--- Checkpoint Status ---")\n'
    'if os.path.isfile(ckpt_path):\n'
    '    ckpt = torch.load(ckpt_path, map_location="cpu")\n'
    '    print(f"✅ Found checkpoint: {ckpt_path}")\n'
    '    print(f"   Epoch: {ckpt[\'epoch\']}")\n'
    '    print(f"   Validation Acc: {ckpt.get(\'acc\', 0.0):.2f}%")\n'
    '    print(f"   Best Acc so far: {ckpt.get(\'best_acc\', 0.0):.2f}%")\n'
    '    \n'
    '    # Load the states\n'
    '    model.load_state_dict(ckpt["state_dict"])\n'
    '    optimizer.load_state_dict(ckpt["optimizer"])\n'
    '    scheduler.load_state_dict(ckpt["scheduler"])\n'
    '    \n'
    '    start_epoch = ckpt["epoch"] + 1\n'
    '    best_acc = ckpt.get("best_acc", 0.0)\n'
    '    print(f"\\n➡️ Ready to resume training from epoch {start_epoch}.")\n'
    '    del ckpt\n'
    'else:\n'
    '    print(f"❌ No checkpoint found at: {ckpt_path}")\n'
    '    print("➡️ Ready to start training from scratch (Epoch 1).")\n'
    '\n'
    'if os.path.isfile(best_path):\n'
    '    print(f"\\n🏆 Best model file exists at: {best_path}")'
)

cell_train_md = md("## 9. Train")

cell_train = code(
    'import time\n'
    '\n'
    'print(f"Training SPaRG-MF on CIFAR-100 | Epochs {start_epoch}-{EPOCHS}")\n'
    'print("=" * 60)\n'
    '\n'
    'total_start = time.time()\n'
    'for epoch in range(start_epoch, EPOCHS + 1):\n'
    '    model.train()\n'
    '    train_loss, correct, total = 0.0, 0, 0\n'
    '    t0 = time.time()\n'
    '\n'
    '    for inputs, targets in train_loader:\n'
    '        inputs, targets = inputs.to(device), targets.to(device)\n'
    '        targets_orig = targets.clone()\n'
    '        inputs, targets = mixup_fn(inputs, targets)\n'
    '        optimizer.zero_grad()\n'
    '        outputs = model(inputs)\n'
    '        loss_cls = criterion(outputs, targets)\n'
    '        loss_div = diversity_loss_fn(model)\n'
    '        loss = loss_cls + loss_div\n'
    '        loss.backward()\n'
    '        optimizer.step()\n'
    '        functional.reset_net(model)\n'
    '\n'
    '        train_loss += loss.item() * inputs.size(0)\n'
    '        _, predicted = outputs.max(1)\n'
    '        total += targets_orig.size(0)\n'
    '        correct += predicted.eq(targets_orig).sum().item()\n'
    '\n'
    '    scheduler.step()\n'
    '    train_acc = 100.0 * correct / total\n'
    '\n'
    '    # Evaluate\n'
    '    model.eval()\n'
    '    correct, total = 0, 0\n'
    '    with torch.no_grad():\n'
    '        for inputs, targets in test_loader:\n'
    '            inputs, targets = inputs.to(device), targets.to(device)\n'
    '            outputs = model(inputs)\n'
    '            functional.reset_net(model)\n'
    '            _, predicted = outputs.max(1)\n'
    '            total += targets.size(0)\n'
    '            correct += predicted.eq(targets).sum().item()\n'
    '    test_acc = 100.0 * correct / total\n'
    '\n'
    '    is_best = test_acc > best_acc\n'
    '    best_acc = max(best_acc, test_acc)\n'
    '    marker = " ** BEST" if is_best else ""\n'
    '    print(f"Epoch {epoch:3d}/{EPOCHS} | {time.time()-t0:.0f}s | "\n'
    '          f"Train: {train_acc:.1f}% | Test: {test_acc:.1f}% | Best: {best_acc:.1f}%{marker}")\n'
    '\n'
    '    # Save checkpoint\n'
    '    state = {\n'
    '        "epoch": epoch,\n'
    '        "state_dict": model.state_dict(),\n'
    '        "optimizer": optimizer.state_dict(),\n'
    '        "scheduler": scheduler.state_dict(),\n'
    '        "acc": test_acc,\n'
    '        "best_acc": best_acc,\n'
    '    }\n'
    '    torch.save(state, ckpt_path)\n'
    '    if is_best:\n'
    '        torch.save(state, best_path)\n'
    '\n'
    '    if epoch % 10 == 0:\n'
    '        stats = model.count_spikes()\n'
    '        if stats:\n'
    '            for name, val in list(stats.items())[:4]:\n'
    '                short = name.split(".")[-1]\n'
    '                print(f"  Spike {short}: rate={val[\'avg_spike_rate\']:.3f}")\n'
    '\n'
    'elapsed = (time.time() - total_start) / 3600\n'
    'print(f"\\nTraining Complete! Best: {best_acc:.2f}% | Time: {elapsed:.1f}h")'
)

cell_eval_md = md("## 10. Evaluate Best Model")

cell_eval = code(
    'best_path = os.path.join(OUTPUT_DIR, "cifar100_best.pth")\n'
    'if not os.path.isfile(best_path):\n'
    '    print(f"No best model found at {best_path}")\n'
    'else:\n'
    '    ckpt = torch.load(best_path, map_location="cpu")\n'
    '    model.load_state_dict(ckpt["state_dict"])\n'
    '    print(f"Loaded best model from epoch {ckpt[\'epoch\']} (saved acc: {ckpt[\'acc\']:.2f}%)")\n'
    '    model.eval()\n'
    '\n'
    '    correct, total = 0, 0\n'
    '    with torch.no_grad():\n'
    '        for inputs, targets in test_loader:\n'
    '            inputs, targets = inputs.to(device), targets.to(device)\n'
    '            outputs = model(inputs)\n'
    '            functional.reset_net(model)\n'
    '            _, predicted = outputs.max(1)\n'
    '            total += targets.size(0)\n'
    '            correct += predicted.eq(targets).sum().item()\n'
    '\n'
    '    final_acc = 100.0 * correct / total\n'
    '    print(f"Final Validation Accuracy: {final_acc:.2f}%")\n'
    '\n'
    '    stats = model.count_spikes()\n'
    '    if stats:\n'
    '        print("\\nHomeostatic Spike Rates:")\n'
    '        for k, v in stats.items():\n'
    '            print(f"  {k}: rate={v[\'avg_spike_rate\']:.4f}, threshold={v[\'v_threshold\']:.4f}")'
)

cell_prune_md = md(
    "## 11. Post-Training Calibration & Multi-Metric Inference Pruning\n"
    "\n"
    "This section runs post-training calibration on the trained model using `SNNInferencePruningEngine` and computes "
    "metrics (QK-alignment Q-value, Head Redundancy, Attention Entropy, Bernoulli Spike Entropy) to generate "
    "highly optimized static hardware bit-masks. It then sweeps thresholds to check accuracy vs head sparsity."
)

cell_prune = code(
    'from engine.inference_pruning import SNNInferencePruningEngine\n'
    '\n'
    'print("--- Initiating Calibration Engine ---")\n'
    'pruning_engine = SNNInferencePruningEngine(model, num_heads=NUM_HEADS)\n'
    '\n'
    '# Run calibration on a subset of the test loader\n'
    'pruning_engine.calibrate(test_loader, device, num_batches=10)\n'
    '\n'
    '# Print a beautiful summary of all diagnostic metrics across heads\n'
    'pruning_engine.interceptor.summary(num_heads=NUM_HEADS)\n'
    '\n'
    '# We use relative pruning (prune_ratio) because embedding scale-up shifts the metric ranges.\n'
    '# Let\'s sweep different prune ratios (e.g., prune bottom 10%, 25%, 50% of heads) for each metric!\n'
    'print("\\n--- Evaluating Gated Pareto Sparsity vs. Accuracy ---")\n'
    'for metric_name in ["q_value", "redundancy", "entropy", "spike_entropy"]:\n'
    '    print(f"\\n Gating Metric: {metric_name.upper()}")\n'
    '    for ratio in [0.0, 0.1, 0.25, 0.5, 0.75]:\n'
    '        masks = pruning_engine.generate_static_masks(metric=metric_name, prune_ratio=ratio)\n'
    '        acc, sparsity = pruning_engine.evaluate_mask_efficiency(test_loader, device, masks)\n'
    '        print(f"   Prune Ratio: {ratio:4.2f} | Accuracy: {acc:6.2f}% | Head Sparsity: {sparsity:6.2f}%")'
)

nb.cells = [
    cell_header,
    cell_clone_md, cell_clone,
    cell_drive_md, cell_drive,
    cell_deps_md, cell_deps,
    cell_verify_md, cell_verify,
    cell_config_md, cell_config,
    cell_data_md, cell_data,
    cell_model_md, cell_model,
    cell_check_md, cell_check,
    cell_train_md, cell_train,
    cell_eval_md, cell_eval,
    cell_prune_md, cell_prune,
]

# Write using the VS Code compatible format minor 4
nb.nbformat_minor = 4
nbformat.write(nb, "colab_training.ipynb")
print(f"Notebook written: {len(nb.cells)} cells")
