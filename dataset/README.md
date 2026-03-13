# 📦 Datasets

This repository uses several datasets for **pretraining**, **verifier training**, **real-world fine-tuning**, and **evaluation**.

---

## Track-On2 Pretraining Dataset

We use the **TAP-Vid Kubric Movi-F** split from [CoTracker3](https://huggingface.co/datasets/facebook/CoTracker3_Kubric).

Download the dataset from [Hugging Face](https://huggingface.co/datasets/facebook/CoTracker3_Kubric) and place it under:

```
/path/to/movi-f/dataset
```

After downloading, **extract all archives** so that each sample resides in its own folder.

Note: the dataset contains **unused depth maps**, which can be safely deleted to reduce disk usage.

---

## Verifier Training Dataset

We use the **K-Epic dataset** from [EgoPoints](https://ahmaddarkhalil.github.io/EgoPoints/).

Download and extract it under any local directory:

```
path/to/k_epic
```

After extraction, the directory should contain both datasets:

```
path/to/k_epic/k_epic/
path/to/k_epic/ego_points/
```

The archive also contains **EgoPoints** (see evaluation datasets below).  
For verifier training, we use the **K-Epic** subset. Set `epic_k_path` in the config to the `train` subdirectory:

```
epic_k_path: path/to/k_epic/k_epic/train
```

---

## Track-On Real-World Fine-Tuning Dataset

For real-world fine-tuning we use the [TAO dataset](https://taodataset.org), which contains videos originally collected for object tracking.

Dataset (Hugging Face):  
[TAO-Amodal](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal)

You must accept the dataset license before downloading.

Download the dataset under:

```
/path/to/real-world/dataset
```

Additionally, we incorporate videos from:

- **[OVIS](https://codalab.lisn.upsaclay.fr/competitions/4763)**
- **[VSPW](https://github.com/VSPW-dataset/VSPW-dataset-download)**

These datasets should be inserted into the TAO-style directory structure as additional **sources** under the `train` split.

---

### Real-World Dataset Structure

The real-world dataset should follow the structure below:

```
/path/to/real-world/dataset/
└── frames/
    ├── train/
    │   ├── source_name_1/
    │   │   ├── video_id_0001/
    │   │   │   ├── 000000.jpg
    │   │   │   ├── 000001.jpg
    │   │   │   └── ...
    │   │   └── video_id_0002/
    │   │       └── ...
    │   └── source_name_2/
    │       └── ...
    │
    ├── val/
    │   └── source_name_1/
    │       └── ...
    │
    └── test/
        └── ...
```

Each frame is stored as:

```
parent/frames/{split}/{source}/{video_id}/{frame_idx}.jpg
```

where:

- `{split}` – dataset split (`train`, `val`, or `test`)
- `{source}` – dataset source (e.g., AVA, BDD, Charades, OVIS, VSPW)
- `{video_id}` – unique identifier of the video
- `{frame_idx}` – frame index within the video

Example:

```
/path/to/real-world/dataset/frames/train/AVA/000123/000045.jpg
```

This layout is used by the data loader during **real-world fine-tuning**.  
You can also add your own real-world videos following this structure.

---

## Evaluation Datasets

The following datasets are used for evaluation:

- **[TAP-Vid DAVIS](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid#downloading-tap-vid-davis-and-tap-vid-rgb-stacking)**
- **[TAP-Vid Kinetics](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid#downloading-and-processing-tap-vid-kinetics)**
- **[RoboTAP](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid#downloading-robotap)**
- **[Dynamic Replica](https://github.com/facebookresearch/dynamic_stereo#download-the-dynamic-replica-dataset)** (use the `validation` split)
- **[PointOdyssey](https://github.com/y-zheng18/point_odyssey#download)** (use the `test` split)
- **[EgoPoints](https://www.dropbox.com/scl/fo/tfvctluqu3cr17jr6q0td/AA6h6GlV-x6QeuupmeLejzA?rlkey=r0q12vbi6wour6qsteklivb6p&e=1&st=1e4b4dnn&dl=0)**

After downloading, these datasets can be used directly for evaluation as described in:

👉 **[Evaluation](../evaluation/README.md)**