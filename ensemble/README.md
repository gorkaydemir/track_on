# 📦 Ensemble Models

This folder contains wrappers for the **state-of-the-art trackers** used in our ensemble.  
Each model has a corresponding predictor class implemented in this repository.

We use the **official checkpoints** released by the authors (many thanks to them for making their work publicly available).  
These models can also be evaluated individually using the evaluation pipeline.

Note that **not all models are used during real-world fine-tuning**. Some are included for completeness, as they were used during evaluation when official benchmark numbers were unavailable.

---

## Ensemble Model Summary

| Model | Paper | Project Page |
|------|------|------|
| TAPIR | TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement | [Project](https://deepmind-tapir.github.io/) |
| BootsTAPIR | BootsTAP: Bootstrapped Training for Tracking-Any-Point | [Project](https://bootstap.github.io/) |
| TAPNext | TAPNext: Tracking Any Point (TAP) as Next Token Prediction | [Project](https://tap-next.github.io/) |
| BootsTAPNext | TAPNext: Tracking Any Point (TAP) as Next Token Prediction | [Project](https://tap-next.github.io/) |
| CoTracker3 | CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos | [Project](https://cotracker3.github.io/) |
| LocoTrack | Local All-Pair Correspondence for Point Tracking | [Project](https://cvlab-kaist.github.io/locotrack/) |
| Anthro-LocoTrack | AnthroTAP: Learning Point Tracking with Real-World Motion | [Project](https://cvlab-kaist.github.io/AnthroTAP/) |
| AllTracker | AllTracker: Efficient Dense Point Tracking at High Resolution | [Project](https://alltracker.github.io/) |

---

## TAPIR and BootsTAPIR

Official repository:  
👉 [TAPNet](https://github.com/google-deepmind/tapnet/tree/main/tapnet/torch)

The **BootsTAPIR** model and **TAPIR** predictor use the official PyTorch implementation provided in the TAPIR demo notebook:

👉 [TAPIR PyTorch Demo](https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/torch_tapir_demo.ipynb)

For **real-world fine-tuning**, we require **BootsTAPIR**:

```bash
wget -P path/to/ckpt https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
```

Optionally, you may download **TAPIR**, although it is not used in our pipelines:

```bash
wget -P path/to/ckpt https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt
```

---

## TAPNext and BootsTAPNext

Official repository:  
👉 [TAPNet/TAPNext](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapnext)

We use the **TAPNext-B** models.

The predictor implementation follows the official PyTorch demo notebook:

👉 [TAPNext PyTorch Demo](https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/torch_tapnext_demo.ipynb)

For **real-world fine-tuning**, we require **BootsTAPNext**:

```bash
wget -P path/to/ckpt https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz
```

Optionally, you may download **TAPNext**, although it is not used in our pipelines:

```bash
wget -P path/to/ckpt https://storage.googleapis.com/dm-tapnet/tapnext/tapnext_ckpt.npz
```

---

## CoTracker3

Official repository:  
👉 [CoTracker](https://github.com/facebookresearch/co-tracker)

We use the **CoTracker3 Window** variant.

During inference we enable **joint tracking** with a fixed support grid of size `10`.

Note that when evaluation follows the exact setup of the original paper (single-query inference with the corresponding grid sizes), the reported results are fully reproducible. However, this configuration is less practical when tracking hundreds of points simultaneously.

The predictor uses the official API and the checkpoint is **automatically downloaded** to the local cache when first used.

---

## LocoTrack and Anthro-LocoTrack

Official repositories:  
👉 [LocoTrack](https://github.com/cvlab-kaist/locotrack/tree/main/locotrack_pytorch)
👉 [Anthro-TAP](https://github.com/cvlab-kaist/AnthroTAP)

We use the **Base model** at resolution `(256, 256)`.  
Using `(384, 512)` significantly increases memory usage and may lead to **OOM errors** for longer sequences in our evaluation setup.

For **real-world fine-tuning**, we require **Anthro-LocoTrack**:

```bash
gdown 1Rj7sIby_ylZkuy4pccAA28dtqvJQUH-A -O path/to/ckpt
```

Optionally, you may download **LocoTrack**, although it is not used in our pipelines:

```bash
wget -P path/to/ckpt https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_base.ckpt
```

---

## AllTracker

Official repository:  
👉 [AllTracker](https://github.com/aharley/alltracker)

We use the **full AllTracker model**.

Note that AllTracker is a **dense tracker**, meaning the model must be run from scratch for frames where queries start. As a result, the computational cost increases with the number of unique query frames.

Download the checkpoint:

```bash
wget -P path/to/ckpt https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth
```