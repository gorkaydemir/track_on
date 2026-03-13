# ⚖️ Evaluation

In addition to the **Track-On** models, this module allows evaluating the **verifier** and other point tracking models using a unified evaluation pipeline.

## Parameters

### Required

- `--dataset_name`: Evaluation dataset. One of:  
  `davis`, `kinetics`, `robotap`, `point_odyssey`, `dynamic_replica`, `ego_points`

- `--dataset_path`: Path to the selected evaluation dataset.

- `--model_names`: One or more model names to evaluate. Supported values:  
  `trackon2`, `tapir`, `bootstapir`, `tapnext`, `bootstapnext`, `cotracker3_video`, `cotracker3_window`, `locotrack`, `anthro_locotrack`, `alltracker`, `verifier`, `oracle`, `random`

  Multiple models can be listed simultaneously. When `verifier` is included, it re-ranks the predictions of the other listed models.  
  `oracle` computes the ensemble upper bound by selecting the best candidate per point (no checkpoint needed).  
  `random` selects a candidate model uniformly at random as a lower-bound baseline (no checkpoint needed).  
  `cotracker3_video` and `cotracker3_window` download their checkpoints automatically on first use.  
  All other models require their official checkpoint to be downloaded. See [`ensemble/README.md`](../ensemble/README.md) for instructions.

### Track-On

- `--trackon_config_path`: Path to the Track-On config file (e.g., `config/test.yaml`).  
  Required if `trackon2` is included in `--model_names` and you want custom model parameters. If omitted, built-in defaults are used.

- `--trackon_checkpoint_path`: Path to the Track-On checkpoint.  
  Required if `trackon2` is included in `--model_names`.

### Verifier

- `--verifier_config_path`: Path to the verifier config file.  
  If omitted, built-in defaults are used.

- `--verifier_checkpoint_path`: Path to the verifier checkpoint.  
  Required if `verifier` is included in `--model_names`.

### Other model checkpoints

Required only for the corresponding model when listed in `--model_names`.

- `--tapir_checkpoint_path`: Path to the TAPIR checkpoint.
- `--bootstapir_checkpoint_path`: Path to the BootsTAPIR checkpoint.
- `--tapnext_checkpoint_path`: Path to the TAPNext checkpoint.
- `--bootstapnext_checkpoint_path`: Path to the BootsTAPNext checkpoint.
- `--locotrack_checkpoint_path`: Path to the LocoTrack checkpoint.
- `--anthro_locotrack_checkpoint_path`: Path to the Anthro-LocoTrack checkpoint.
- `--alltracker_checkpoint_path`: Path to the AllTracker checkpoint.

### Prediction caching

- `--cache_predictions`: Flag. When set, model predictions are cached to disk.  
  Highly recommended when evaluating the **verifier**, so teacher predictions are not recomputed on every run.

- `--cache_dir`: Directory where cached predictions are stored (default: `./cached_predictions`).  
  Only used when `--cache_predictions` is set.

---

## Example Usage

### Example 1

Evaluate **Track-On2** and **BootsTAPNext** on **DAVIS**.

```bash
torchrun --master_port=12345 --nproc_per_node=1 -m evaluation.eval \
--dataset_name "davis" \
--dataset_path /path/to/davis \
--model_names "trackon2" "bootstapnext" \
--trackon_config_path config/test.yaml \
--trackon_checkpoint_path /path/to/trackon/ckpt \
--bootstapnext_checkpoint_path /path/to/bootstapnext/ckpt
```

### Example 2

Evaluate **Track-On2**, **Anthro-LocoTrack**, **AllTracker**, **CoTracker3 (window)**, and **Verifier** on Kinetics.
The verifier re-ranks the predictions of the preceding models as candidate trajectories.

```bash
torchrun --master_port=12345 --nproc_per_node=1 -m evaluation.eval \
--dataset_name "kinetics" \
--dataset_path /path/to/kinetics \
--model_names "trackon2" "anthro_locotrack" "alltracker" "cotracker3_window" "verifier" \
--trackon_config_path config/test.yaml \
--trackon_checkpoint_path /path/to/trackon/ckpt \
--anthro_locotrack_checkpoint_path /path/to/anthrolocotrack/ckpt \
--alltracker_checkpoint_path /path/to/alltracker/ckpt \
--verifier_checkpoint_path /path/to/verifier/ckpt
```

