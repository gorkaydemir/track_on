# Track-On2: Enhancing Online Point Tracking with Memory

## [**Webpage**](https://kuis-ai.github.io/track_on2) | [**Track-On (ICLR’25)**](https://arxiv.org/abs/2501.18487) | **[**Track-On2**](https://arxiv.org/abs/2509.19115)**

This repository contains the official implementation of both versions:  
> **Track-On: Transformer-based Online Point Tracking with Memory**  
> [Görkay Aydemir](https://gorkaydemir.github.io), Xiongyi Cai, [Weidi Xie](https://weidixie.github.io), [Fatma Güney](https://mysite.ku.edu.tr/fguney/)  
> *International Conference on Learning Representations (ICLR), 2025*

> **Track-On2: Enhancing Online Point Tracking with Memory**  
> [Görkay Aydemir](https://gorkaydemir.github.io), [Weidi Xie](https://weidixie.github.io), [Fatma Güney](https://mysite.ku.edu.tr/fguney/)  
> *Under submission*

---

## Overview

**Track-On** is an efficient **online point tracking** model that processes videos **frame-by-frame** with a compact transformer memory—no future frames, no windows. **Track-On2** builds on this with improved accuracy and efficiency.

<p align="center">
  <img src="media/teaser.png" alt="Track-On Overview" width="800" />
</p>

---

## 🚀 Installation

### Clone the repository
```bash
git clone https://github.com/gorkaydemir/track_on.git
cd track_on
```

### Set up the environment
Use `mamba` or `conda`:
```bash
mamba create -n trackon2 python=3.12
mamba activate trackon2
mamba install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install -r requirements.txt
```

---

## 🔑 Pretrained models

We provide two pretrained **Track-On2** checkpoints, each using a different backbone:

- **Track-On2 with DINOv3**  
  [Download here](https://huggingface.co/gorkaydemir/track_on2/resolve/main/trackon2_dinov3_checkpoint.pt?download=true)  
  This checkpoint uses the **DINOv3** visual backbone.  
  - To use it, you must separately obtain the official pretrained DINOv3 weights of [dinov3-vits16plus](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m) by requesting access through Hugging Face.  
  - Our released checkpoints **do not include** backbone weights in order to comply with DINOv3’s licensing and distribution policy.

- **Track-On2 with DINOv2**  
  [Download here](https://huggingface.co/gorkaydemir/track_on2/resolve/main/trackon2_dinov2_checkpoint.pt?download=true)  
  No additional permissions or downloads are needed.  
  - It offers competitive, often comparable (or stronger) performance to the DINOv3 variant.  
  - Recommended if you want a quick setup without external dependencies.
---

## 🎬 Demo

You can track points on a video using the **`Predictor`** class.

### Minimal example
```python
import torch
from model.trackon_predictor import Predictor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize
model = Predictor(args, checkpoint_path="path/to/checkpoint.pth").to(device).eval()

# Inputs
# video:   (1, T, 3, H, W) in range 0-255
# queries: (1, N, 3) with rows = (t, x, y) in pixel coordinates
#          or use None to enable the model's uniform grid querying
video = ...          # e.g., torchvision.io.read_video -> (T, H, W, 3) -> (T, 3, H, W) -> add batch dim
queries = ...        # e.g., torch.tensor([[0, 190, 190], [0, 200, 190], ...]).unsqueeze(0).to(device)

# Inference
traj, vis = model(video, queries)

# Outputs
# traj: (1, T, N, 2)  -> per-point (x, y) in pixels
# vis:  (1, T, N)     -> per-point visibility probability in [0, 1]
```

### Using `demo.py`

A ready-to-run script ([`demo.py`](demo.py)) handles loading, preprocessing, inference, and visualization.

Given:
- `$video_path`: Path to the input video file (e.g., `.mp4`)
- `$config_path`: Config file of the model with `yaml` extension (default: `./config/test.yaml`)
- `$ckpt_path`: Path to the Track-On2 checkpoint (`.pth`)
- `$output_path`: Path to save the rendered tracking video (e.g., `demo_output.mp4`)
- `$use_grid`: Whether to use a uniform grid of queries (`true` or `false`)

you can run the demo by
```bash
python demo.py \
--video $video_path \
--config $config_path \
--ckpt $ckpt_path \
--output $output_path \
--use-grid $use_grid
```

Running the model with uniform grid queries on the video at `media/sample.mp4` produces the visualization shown below.

<p align="center">
  <img src="media/demo_output.gif" alt="Sample Tracking" width="300" />
</p>

---

## 📦 Datasets

### Training dataset
We use the TAP-Vid Kubric Movi-F split from [CoTracker3](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid). You can download it from [Hugging Face](https://huggingface.co/datasets/facebook/CoTracker3_Kubric). After downloading to `$movi_f_root`, **extract** all archives so each sample is in its own folder.

### Evaluation datasets
- [TAP-Vid DAVIS](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid#downloading-tap-vid-davis-and-tap-vid-rgb-stacking)
- [TAP-Vid Kinetics](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid#downloading-and-processing-tap-vid-kinetics)
- [RoboTAP](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid#downloading-robotap)
- [Dynamic Replica](https://github.com/facebookresearch/dynamic_stereo#download-the-dynamic-replica-dataset)
- [PointOdyssey](https://github.com/y-zheng18/point_odyssey#download) (use the `test` split)

After downloading, you can directly use these datasets for evaluation as described in the **Evaluation** section.

---

## ⚖️ Evaluation

Given:
- `$model_config_path`: Path to the model config (`config/test.yaml` for DINOv3, `config/test_dinov2.yaml` for DINOv2)
- `$dataset_name`: One of `davis`, `kinetics`, `robotap`, `point_odyssey`, `dynamic_replica`
- `$dataset_path`: Path to the selected evaluation dataset
- `$model_checkpoint_path`: Path to the downloaded checkpoint

run:
```bash
torchrun --master_port=12345 --nproc_per_node=1 -m evaluation.eval \
--model_config_path $model_config_path \
--dataset_name $dataset_name \
--dataset_path $dataset_path \
--model_checkpoint_path $model_checkpoint_path
```

This should reproduce the paper’s results ($\delta_{avg}^x$) when configured correctly:

| Model Backbone | DAVIS | Kinetics | RoboTAP | Dynamic Replica | PointOdyssey |
| :---: | :---: | :---: | :---: | :---: | :---: |
| DINOv3 | 79.9 | 69.3 | 80.5 | 74.5 | 45.1 |
| DINOv2 | 79.8 | 69.1 | 80.0 | 74.6 | 47.4 |

### Benchmarking

Compute inference statistics (GPU memory and throughput) on DAVIS.

Given:
- `$davis_path`: Path to TAP-Vid DAVIS
- `$N_sqrt`: √(number of points) (e.g., `8` → 64 points)
- `$memory_size`: Inference-time memory size

run:
```bash
torchrun --master_port=12345 --nproc_per_node=1 -m evaluation.benchmark \
--davis_path $davis_path \
--model_checkpoint_path $model_checkpoint_path
--N_sqrt $N_sqrt \
--memory_size $memory_size
```

---

## 🛠️ Training

Given:
- `$config_path`: Config file of the model (`.yaml`), default `./config/train.yaml`
- `$movi_f_root`: Path to TAP-Vid Kubric dataset
- `$tapvid_root`: Path to TAP-Vid DAVIS dataset (used for evaluation after each epoch)
- `$model_save_path`: Folder to save last and best checkpoints

you can train the model by running
```bash
torchrun --master_port=12345 --nproc_per_node=#gpus main.py \
--config_path $config_path \
--movi_f_root $movi_f_root \
--tapvid_root $tapvid_root \
--model_save_path $model_save_path
```

You can continue training by passing the latest checkpoint via `--checkpoint_path`.  
Training saves the latest checkpoint after each epoch.

---

## 📜 Previous Version

Track-On2 is recommended for both performance and efficiency.  
For convenience, the original Track-On code is available in the `track-on` branch.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Aydemir2025TrackOn,
  title     = {{Track-On}: Transformer-based Online Point Tracking with Memory},
  author    = {Aydemir, G\"orkay and Cai, Xiongyi and Xie, Weidi and G\"uney, Fatma},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025}
}
```

```bibtex
@article{Aydemir2025TrackOn2,
  title={{Track-On2}: Enhancing Online Point Tracking with Memory},
  author={Aydemir, G\"orkay and Xie, Weidi and G\"uney, Fatma},
  journal={arXiv preprint arXiv:2509.19115},
  year={2025}
}
```

---

## Acknowledgments

This repository incorporates code from public works including [CoTracker](https://github.com/facebookresearch/co-tracker), [TAPNet](https://github.com/google-deepmind/tapnet), [DINOv2](https://github.com/facebookresearch/dinov2), [ViT-Adapter](https://github.com/czczup/ViT-Adapter), and [SPINO](https://github.com/robot-learning-freiburg/SPINO). We thank the authors for making their code available.
