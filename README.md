# SPaRG-MF: Spiking, Precision-Aware, Routed, and Gated Max-Former

## Overview

**SPaRG-MF** is an advanced neuromorphic computer vision architecture designed to dramatically reduce inference energy costs without sacrificing accuracy. It achieves this by taking the state-of-the-art NeurIPS 2025 [Max-Former](https://github.com/bic-L/MaxFormer) backbone and aggressively evolving it into an ultra-sparse, dynamically routed, and mixed-precision **Spiking Neural Network (SNN)**.

Traditional artificial neural networks compute using dense, continuous, energy-hungry floating-point multiply-accumulate (MAC) operations. SNNs communicate asynchronously via binary "spikes" (0 or 1), translating heavy MACs into extremely cheap localized additions (ACs). SPaRG-MF pushes this frontier further by actively pruning unnecessary tokens and attention heads dynamically while utilizing homeostatic self-regulation to keep the network stable.

---

## 1. The Core Backbone: Hierarchical Max-Former

Like its non-spiking predecessor, SPaRG-MF drops the traditional "flat" pipeline of early Vision Transformers (like ViT) in favor of a 3-stage **hierarchical design** to efficiently process high-resolution images.

*   **Stage 1 (High Res, Local Features):** Contains an initial `Embed_Orig_ImageNet` stem that downsamples the image by 4×, followed by Depthwise Convolution (DWC-7) mixers mapped into Spiking MLPs.
*   **Stage 2 (Mid Res, Local Features):** Uses a spiking `MaxPool` shortcut downsampler (reducing spatial resolution again) coupled with 5×5 Depthwise formers (DWC-5).
*   **Stage 3 (Low Res, Global Context):** The final stage downsamples by a final 2× factor and uses full **Spiking Self-Attention (SSA)** to build global semantic understanding necessary for dense classification.

---

## 2. SPaRG-MF Custom Architectural Enhancements

To make this architecture natively hardware-efficient and heavily sparse, we developed five isolated but interconnected modules:

### A. Frequency-Preserving Patch Embedding (`patch_embed.py`)
Traditional transformers lose high-frequency visual data during their initial tokenization. We replaced this with a multi-stage overlapping convolutional stem. Crucially, the downsampling passes inject "membrane shortcuts" (residual connections *before* the LIF spike activation module triggers) directly into the downstream stages to preserve high-frequency signals.

### B. Soft Spiking Self-Attention (`mixer_hub.py`)
Instead of dense Softmax operations (which disrupt spiking topologies), we implement spike-driven linear attention: $\text{Out} = (\mathbf{Q} \otimes (\mathbf{K}^T \otimes \mathbf{V}))$. We enhanced this by introducing an adaptive **Temperature Scaling** parameter. This scales the pre-spike potentials to coerce the distribution to be "soft-sparse"—making active pathways dominant while naturally silencing low-entropy background signals without forcing brittle threshold breaks.

### C. Homeostatic Spike Control (`homeostasis.py`)
When you aggressively prune networks (such as gating complete attention heads or tokens), the downstream features get starved of voltage potentials ("dead pathways"), or alternatively, unchecked neurons "seize" and fire every frame ("dominant pathways").
To solve this, we implemented a `HomeostaticLIFNode`. It silently tracks its exponential moving average (EMA) firing rate. If a neuron fires too often compared to a target rate (e.g., 30%), it dynamically raises its own firing threshold. If it fires too rarely, it lowers it. This stabilizes aggressive gating penalties.

### D. Dynamic Head and Token Gating (`dynamic_gate.py` & `mixed_precision.py`)
Instead of processing the entire image blindly, the network actively decides what to ignore:
*   **Token Gating:** Unimportant background patches (e.g., a solid blue sky) are completely muted.
*   **Mixed Precision:** Tokens that border on "medium importance" aren't killed entirely. Instead, they are quantized via a Straight-Through Estimator (STE) down to 4-bit precision, while complex details (e.g., a dog's face) retain 8-bit precision.
*   **Head Gating:** Entire attention heads that continuously produce low entropy are masked.

### E. Neuromorphic SSA Interceptor (`engine/ssa_interceptor.py`)
Because dynamic gating is difficult for static edge-hardware to optimize at runtime, we built a forward-hook Interception Engine. During a testing phase, this engine secretly records pre-spike and post-spike potentials across all attention heads. It mathematically computes importance scores $\text{Variance}(A) \times \text{Mean}(A)$. We can use this data to generate un-changing, hardcoded bit-masks to permanently configure highly-efficient ASIC/FPGA hardware layouts.

---

## 3. Project File Structure & Training Framework

The repository is built entirely on generic PyTorch and standard SpikingJelly logic, allowing it to interface natively with normal DL infrastructure.

*   `models/mixer_hub.py`: Contains the Spiking MLP, Depthwise Conv blocks, and Soft SSA formulations.
*   `models/homeostasis.py`: Contains our self-regulating Membrane algorithms.
*   `models/maxformer_snn.py`: Assembles the stages and houses the `@register_model` TIMM hooks.
*   `train_cifar10.py`: A lightweight, heavily annotated PyTorch sequence that validates the structural integrity on 32x32 targets.
*   `train_imagenet.py`: A full-scale PyTorch `timm` distributed training harness. Automatically implements `torchrun` Distributed Data Parallel (DDP), MixUp augmentations, AutoCast routines, and resets SpikingJelly membrane buffer states between sequential ImageNet iterations.

### To Start Validating:
```bash
# Run lightweight CIFAR-10 verification on minimal architecture
python train_cifar10.py

# Scale to ImageNet via distributed torchrun
torchrun --nproc_per_node=4 train_imagenet.py --data_dir ./data --batch_size 64
```
