<h1 align="center">Local Experiment Notes: Multiplex Testing</h1>

This private snapshot contains the local Multiplex Thinking experiments run on
Hyak/Slurm. The core theme was compute-efficient reasoning by sharing branches
or memory across candidate traces, then comparing that against independent
chain-of-thought sampling and RL-trained baselines.

## What Was Run

| Experiment family | Goal | Main scripts | Result folders |
| --- | --- | --- | --- |
| AIME train100 branch-sharing ablations | Compare fixed independent sampling with shared-every-2/4/8/16/32 branch budgets and repair/top-up runs. | `scripts/aggregate_aime_train100_ablation4096.py`, `scripts/plot_branch_ablation_budget_sweep.py`, repair/top-up submit scripts | `final_eval_outputs/aime-train100-ablation4096-r3-stable3b-20260501-103533` |
| Fixed plus shared mixed sources | Combine fixed independent traces with shared-memory traces and aggregate pass@k/cost tradeoffs. | `scripts/build_mixed_memory_match_sources.py`, `scripts/aggregate_aime_train100_fixed_shared_mix4096.py` | `final_eval_outputs/aime-train100-fixed-shared-mix4096-r3-mix4096-20260501-221133` |
| Short training reproductions | Reproduce discrete RL vs Multiplex Thinking training on small DeepScaleR subsets. | `scripts/run_six_hour_multiplex_repro_h200.sh`, `scripts/plot_training_logs.py` | `final_eval_outputs/five-hour-multiplex-35226849-20260514-024457` |
| Long-budget scaling | Train/evaluate discrete RL, trained Multiplex Thinking, shared4 base, and shared4 plus shared2 top-ups on AIME 2024. | `scripts/run_twelve_hour_long_budget_scaling_h200.sh`, `scripts/aggregate_four_hour_scaling.py`, `scripts/plot_eval_bootstrap_comparisons.py` | `final_eval_outputs/twelve-hour-long-budget-scaling-retry2` |
| Shared4 objective smoke suite | Test shared4 joint/thinking/answer objectives and soft-HF rollout compatibility. | `scripts/run_shared4_objective_suite_h200.sh`, `scripts/smoke_soft_thinking_generation.py` | `final_eval_outputs/smoke-soft-hf-4x4-a40b-20260522-0212` |

## Key Results

- The AIME train100 ablation showed that shared branching can reduce token cost
  while preserving much of pass@k. For example, the stable ablation summary has
  fixed k=32 around `0.747` pass@k, while shared-every-2 k=32 reached about
  `0.740` at substantially lower average token cost.
- The mixed fixed+shared aggregate reached about `0.740` mean accuracy at
  k=32, using fixed and shared components together. This was useful as a budget
  tradeoff, but it did not make shared traces universally dominate fixed
  independent sampling.
- The short five-hour reproduction was mostly a systems/procedure check. On a
  tiny 32-example training subset, both discrete RL and Multiplex Thinking evals
  were near zero on the 20-prompt AIME sweep.
- The twelve-hour long-budget run was the most informative training result:
  trained Multiplex Thinking reached mean pass@8 `0.289` versus discrete RL
  trained at `0.189`; compute-normalized shared4 plus shared2 top-up reached
  mean pass@8 `0.333` with larger effective k and higher cost.
- The shared4 objective smoke suite verified that the local shared4 objective
  paths and soft-HF/SGLang compatibility could execute, but the configured
  smoke run was intentionally only one training step.

## What Was Learned

- The shared-branch idea is viable as a cost-control knob, but it needs top-ups
  or hybrid fixed/shared sampling to recover pass@k at high budgets.
- Small RL reproductions are too noisy to judge algorithm quality. They are
  useful for finding runtime, checkpoint, and rollout failures.
- Dependency management was a large part of the work. The scripts now bootstrap
  a scratch-managed runtime overlay to stabilize `verl`, `sglang`, CUDA,
  FlashInfer, and Transformers on cluster nodes.
- Checkpoints, final eval folders, and Slurm logs are very large and are kept as
  local artifacts. The repo keeps source patches, scripts, configs, and this
  ledger so results can be regenerated.

## Local Artifact Policy

The following are intentionally ignored for GitHub:

- `final_eval_outputs/` - checkpoints, generated traces, plots, logs, and eval
  artifacts.
- `slurm_logs/` - scheduler stdout/stderr.
- `**/checkpoints/`, `**/outputs/`, `**/wandb/` - training outputs.

The upstream project README starts below for environment and usage context.

---

<div align="center">

<h1 style="display: flex; align-items: center; justify-content: center; gap: 10px; margin: 0;">
  <img src="figs/logo_clip.png" alt="logo" height="50" style="display: block;" />
  <span>Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge</span>
</h1>

</div>

[![Paper](https://img.shields.io/badge/arXiv-2601.08808-B31B1B.svg)](https://arxiv.org/abs/2601.08808)
[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-FFD21E)](https://huggingface.co/Multiplex-Thinking)
[![Website](https://img.shields.io/badge/Website-Online-2ea44f)](https://gmlr-penn.github.io/Multiplex-Thinking/)

<div align="center">
  <img src="figs/teaser.png" alt="teaser" width="750" />
</div>

<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)  -->


## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started-)
  - [Environment Setup](#environment-setup)
  - [Base Docker Image](#base-docker-image)
  - [Dependencies](#dependencies)
  - [Setup](#setup)
- [Training and evaluation](#training-and-evaluation)
- [Implementation Credits](#implementation-credits)
- [Checkpoints](#-checkpoints)

## Overview

This repository contains the **official implementation** of **Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge**.

Multiplex Thinking proposes a **token-wise branch-and-merge reasoning mechanism**, enabling efficient and expressive multi-pat reasoning while maintaining a compact token representation.

The codebase is built upon several high-quality open-source projects. We sincerely thank the original authors and contributors for their outstanding work.

---

## Getting Started 🚀

### Environment Setup

We recommend using Docker to ensure a consistent and reproducible environment. If you prefer Conda, we also provide an environment specification in `conda_env.yaml`.

### Base Docker Image

We suggest starting from the official **verl SGLang worker** Docker image:

- https://github.com/volcengine/verl/blob/325cbc770bfe32ef022f1cd67feab1a23bba9e42/docker/verl0.5-cu126-torch2.7-fa2.7.4/Dockerfile.app.sglang0.4.9.post6.mcore0.13

For general system configuration, please refer to the official documentation of verl:

- https://verl.readthedocs.io/en/latest/workers/sglang_worker.html

### Dependencies

Please ensure the following package versions are installed:

- `sglang == 0.4.9.post6`
- `transformers == 4.54.0`

### Setup

Run the setup script:

```bash
bash setup.sh
```
    The `setup.sh` script handles the installation of required dependencies and ensures the correct versions of our customized libraries are active by running:
    * `pip install sglang-0.4.9.post6`
    * `pip install transformers-4.54.0`

## Training and evaluation

Train and evaluate by running:

```
bash scripts/train.sh \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --exp_name your_exp_name \
  --enable_unweighting True \ # True for average embedding; False for weighted embedding
  --total_training_steps 300 \
  --train_batch_size 128 \
  --max_token_len_per_gpu 32768 \
  --loss_mode multiplex_thinking \
  --multiplex_width 3 \
  --n_gpus_per_node 8 \
  --max_response_length 4096 \
  --val_rollout_n 4 \
  --val_dataset math \
  --val_batch_size 1024
```

Or run evaluation:

`bash scripts/eval.sh`

## Implementation Credits
This codebase is built upon and inspired by the exceptional work from the following projects:
* **Training & RL Framework**: [verl](https://github.com/volcengine/verl) & [DeepScaleR](https://github.com/agentica-project/DeepScaleR)
* **Inference Engine**: [sglang](https://github.com/sgl-project/sglang)
* **Code Inspiration & Adaptations**: [Soft Thinking](https://github.com/eric-ai-lab/Soft-Thinking)


## 📁 Checkpoints
Model weights are available on Hugging Face:
👉 [**Multiplex-Thinking-HF-Checkpoints**](https://huggingface.co/Multiplex-Thinking)

# ✍️ Citation 
If you find this work useful for your research, please cite our paper as:
```
@article{tang2026multiplexthinking,
  title   = {Multiplex Thinking: Reasoning via Token-wise Branch-and-Merge},
  author  = {Tang, Yao and Dong, Li and Hao, Yaru and Dong, Qingxiu and Wei, Furu and Gu, Jiatao},
  journal = {arXiv preprint arXiv:2601.08808},
  year    = {2026}
}
```
