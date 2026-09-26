# DeChunk

Official implementation of **DeChunk**: a unified single-forward flow-matching
Vision-Language-Action (VLA) model with auxiliary next-chunk prediction.

DeChunk trains the action expert to denoise **two action chunks in one forward
pass**: the *current chunk* (executed at inference) and a longer *next chunk*
(a training-only auxiliary objective). A block-causal attention mask guarantees
that base tokens never attend to next-chunk tokens, so at inference the
next-chunk tokens are simply never appended — the deployed policy is
**bit-identical** to the masked training forward, with zero extra cost.

## Method at a glance

Sequence layout during training:

```
[current_actions (A), proprio, registers, task_cond, obs tokens, next_actions (A')]
```

with independent, asymmetric lengths (e.g. `A=4`, `A'=12` on MetaWorld) and
independent flow-matching times `t` / `t_next`.

- **Block-causal mask** (`models/vla_model_fm_unified_2.py`, `_build_unified_mask`):
  - base tokens **never** attend to next-chunk tokens (always);
  - next-chunk tokens attend to current actions only if
    `model.next_attend_current: true` (default `false`, blocks the
    current→next shortcut).
- **Next-chunk tokens** get their own type embedding, a *continuous* positional
  encoding (honest absolute positions `A..A+A'-1`), an optional separate output
  projection, and an independent denoising time `t_next`.
- **Loss**: `mse_current + lambda_next_chunk * mse_next`.
- **Inference**: leave `t_next=None, noisy_actions_next=None` — the sequence
  contains only base tokens. Attention is the only cross-token operation, so
  the current-chunk output is exactly the same as in training.

Backbone: frozen **DINOv3 ViT-L/16** features fused through Perceiver-style
adapters; action expert: adaLN-zero DiT with per-token time conditioning.

## Repository structure

```
├── train_dinov3_unified_2.py     # pre-training entry point
├── eval_metaworld_parallel.py    # MetaWorld evaluation (serial / batched-parallel)
├── configs/
│   └── metaworld_unified_2.yaml  # MetaWorld MT50 config (current 4 + next 12)
├── models/
│   ├── vla_model_fm_unified_2.py     # DeChunk model + unified FM loss
│   ├── model_runner_unified_2.py     # model factory + training wrapper
│   └── robotwin_infer_unified_2.py   # inference agent (Euler ODE sampling)
├── dataloader/
│   └── dataset_unified_2.py      # MetaWorld HDF5 dataset
├── task_cond/
│   └── metaworld/                # per-task conditioning vectors (MT50, dim 1024)
└── utils/
    ├── train_utils.py
    └── stat-metaworld.json       # action/state normalization stats
```

## Installation

```bash
conda create -n dechunk python=3.10 -y
conda activate dechunk
pip install -r requirements.txt
```

Tested environment: Python 3.10, torch 2.7.1 (cu128), torchvision 0.22.1,
transformers 5.0.0rc0, mujoco 3.3.0, gymnasium 1.3.0, metaworld 3.1.1.

**DINOv3 weights.** Download the frozen backbone
[`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
and place it at:

```
./dinov3_pretrain/dinov3-vitl16-pretrain-lvd1689m
```

(or edit `model.vision_encoder.checkpoint_path` in the config).

## Data preparation

The dataloader expects one folder per MetaWorld task of HDF5 demonstrations:

```
<dataset_dir>/<task>/demo_clean/data/*.hdf5
```

Set `dataset.dataset_dir` in `configs/metaworld_unified_2.yaml` accordingly.
MetaWorld envs clip actions to [-1, 1] internally; `dataset.action_clip: 1.0`
keeps training targets aligned with execution and matches the shipped
`utils/stat-metaworld.json`.

The task-condition vectors for all 50 MT50 tasks are shipped in this repo
under `task_cond/metaworld/<task>/task_cond.npy` (loaded when
`model.use_task_cond: true`, the default).

## Training

```bash
python train_dinov3_unified_2.py \
    --config configs/metaworld_unified_2.yaml \
    --save_dir ./checkpoints_vla
```

Useful flags: `--tasks assembly-v3 push-v3 ...` (train on a task subset;
omit for all 50 tasks), `--resume <ckpt>` (restore model + optimizer +
scheduler + epoch).

Checkpoints are written to `./checkpoints_vla/unified_2_all_<timestamp>/`
together with the resolved `config.yaml`.

## Evaluation

```bash
# Full MT50 suite, 50 parallel envs per task, 50 episodes
python eval_metaworld_parallel.py \
    --checkpoint ./checkpoints_vla/unified_2_all_<timestamp>/checkpoint_epoch_30.pt \
    --task all --mode parallel --num-envs 50 --num-episodes 50

# Single task
python eval_metaworld_parallel.py \
    --checkpoint <ckpt> --task door-lock-v3 --num-episodes 50
```

Results (per-task and mean success rates) are logged to `log/eval/`.
Headless rendering uses EGL by default (`--mujoco-gl egl|osmesa|glfw`).
Add `--save-video` to record rollout MP4s.

## Key configuration

| Config key | Meaning | Default |
|---|---|---|
| `common.action_chunk_size` | current chunk length `A` (used at inference) | 4 |
| `common.next_chunk_size` | next chunk length `A'` (training-only) | 12 |
| `training.use_next_chunk_pred` | enable the auxiliary next-chunk loss | true |
| `training.lambda_next_chunk` | weight of the next-chunk loss | 0.9 |
