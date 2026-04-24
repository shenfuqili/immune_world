# immune_world

Reference implementation of ImmuneWorld — a trajectory-aware foundation model for tumour-immune
microenvironment simulation, as described in *Foundation Model-Based World Simulator Predicts
Immune Microenvironment Evolution from Single-Cell Trajectories* (Nature Communications
submission; see §Citation).

## 1. Overview

ImmuneWorld treats immune-microenvironment dynamics as a world-modelling problem: an
autoregressive, trajectory-aware transformer pretrained on 12.4 M immune-enriched single cells
that predicts the next cell state given the cell's trajectory so far and an optional therapeutic
perturbation context. A shared 68 M-parameter backbone serves four downstream heads
(perturbation prediction, trajectory reconstruction, cell-type deconvolution, immunotherapy
response prediction). The architecture composes seven design elements: Gene2Vec-initialised
embeddings (Eq. 2), a learnable per-head temporal-decay attention bias (Eq. 4), SwiGLU feed-forward
blocks (Eq. 5), a gradient-reversal cross-cancer transfer head (Eq. 6), a mean-pooled
perturbation-context embedder (Eq. 7) injected via cross-attention (Eq. 8), and a three-branch
composite objective (Eq. 9). The authoritative derivation of every field mentioned below lives in
`docs/project-context.md`; implementation ↔ equation mapping lives in `docs/implementation-map.md`.

## 2. Installation

### 2a. pip (user environment, recommended for development)
```
git clone <repo-url> immune_world
cd immune_world
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```
On a CUDA host, additionally:
```
pip install -e ".[gpu]"        # installs flash-attn 2.6 (matched to torch 2.3 / CUDA 12.1)
```

### 2b. conda
```
conda env create -f environment.yml
conda activate immune_world
pip install -e .               # picks up editable install inside the env
```

### 2c. Docker (mirrors the paper's compute target)
```
docker build -t immune_world:dev .
docker run --gpus all -it immune_world:dev
```
The Dockerfile is pinned to `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` +
torch 2.3.x (CUDA 12.1 wheels) + flash-attn 2.6.x.

## 3. Data

All datasets are publicly available; no private / restricted data. Per-dataset SHA-256 manifests
are written to `${IMMUNE_WORLD_DATA_DIR:-data/processed}/manifests/<dataset>.sha256` by
`scripts/prepare_data.sh`.

| Dataset | Source URL | Version / Accession | License | Cells / Patients | On-disk (approx.) |
|---|---|---|---|---|---|
| Norman Perturb-Seq   | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344 | GSE133344 | GEO public (≈CC0) | 91 205 cells / 284 perts | ~1.5 GB |
| Adamson Perturb-Seq  | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546  | GSE90546  | GEO public | 68 603 cells / 87 perts | ~1.0 GB |
| Replogle K562        | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132080 | GSE132080 | GEO public | 162 751 cells / 1 093 perts | ~3.0 GB |
| Replogle RPE1        | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132080 | GSE132080 | GEO public | 162 733 cells / 1 544 perts | ~3.0 GB |
| Pancreas             | https://scvelo.readthedocs.io/en/stable/scvelo.datasets.html | Bastidas-Ponce 2019 | inherited (public) | 3 696 cells | ~15 MB |
| Dentate Gyrus        | https://scvelo.readthedocs.io/en/stable/scvelo.datasets.html | Hochgerner 2018 | inherited | 2 930 cells | ~15 MB |
| Bone Marrow          | https://scvelo.readthedocs.io/en/stable/scvelo.datasets.html | Setty 2019 | inherited | 5 780 cells | ~25 MB |
| ICBatlas (14 cohorts) | http://bioinfo.life.hust.edu.cn/ICBatlas/ | Tang 2023 [24] | academic-open | 2 834 patients (7 cancers) | ~400 MB |
| Human Cell Atlas (immune subset) | https://www.humancellatlas.org/ | 2024 immune release | HCA Data Release Policy | ~5.2 M cells | ~45 GB |
| CellxGene (immune)    | https://cellxgene.cziscience.com/ | 2024 snapshot | CC-BY-4.0 metadata, per-deposit licences | ~4.8 M cells | ~40 GB |
| GEO Perturb-Seq (immune) | https://www.ncbi.nlm.nih.gov/geo/ | aggregated 2020–2024 | GEO public | ~2.4 M cells | ~25 GB |

Required preprocessing (runs via `bash scripts/prepare_data.sh`):

```
# 1. Fetch + cache every dataset listed above (honours $IMMUNE_WORLD_DATA_DIR)
bash scripts/prepare_data.sh

# 2. Subset each source to the shared top-2000 HVG vocabulary and write SHA-256 manifests
python -m immune_world.cli.prepare_data all
```

Every dataset's ethics is recorded in `docs/project-context.md §6`; see the Ethics Declaration in
the paper ("This study is purely computational, using only publicly available, de-identified
datasets").

## 4. Training

```
# Primary pretraining run (paper's main result)
torchrun --nproc_per_node=4 -m immune_world.cli.train experiment=main

# Per-task fine-tuning (run after pretraining; uses the pretrained checkpoint)
python -m immune_world.cli.train experiment=perturbation_norman   --phase finetune
python -m immune_world.cli.train experiment=perturbation_adamson  --phase finetune
python -m immune_world.cli.train experiment=perturbation_k562     --phase finetune
python -m immune_world.cli.train experiment=perturbation_rpe1     --phase finetune
python -m immune_world.cli.train experiment=trajectory_pancreas   --phase finetune
python -m immune_world.cli.train experiment=trajectory_dentate_gyrus --phase finetune
python -m immune_world.cli.train experiment=trajectory_bone_marrow --phase finetune
python -m immune_world.cli.train experiment=deconvolution_pancreas --phase finetune
python -m immune_world.cli.train experiment=icb_response           --phase finetune
```

Expected wall-clock on 4× A100 80 GB (from §2.1 / §2.8 of the paper, reported as-is):

| Phase | Wall-clock | GPU-hours |
|---|---|---|
| Pretraining (200 epochs, 12.4 M cells)   | 72 h  | 288 |
| Fine-tuning (per task, single A100)     | 2 – 6 h | 2 – 6 |
| Inference (Table 5, batch 256, A100)    | 14 200 cells / s | — |

Ablation (Table 3) and cross-cancer / cohort-size configs each reuse the same pretrained
checkpoint; only the fine-tune step re-runs:

```
python -m immune_world.cli.train experiment=ablation_no_traj_attention         --phase finetune
python -m immune_world.cli.train experiment=ablation_no_cross_cancer_transfer  --phase finetune
python -m immune_world.cli.train experiment=ablation_no_perturbation_engine    --phase finetune
python -m immune_world.cli.train experiment=ablation_no_gene_embed_pretrain    --phase finetune
python -m immune_world.cli.train experiment=ablation_no_temporal_encoding      --phase finetune
python -m immune_world.cli.train experiment=ablation_no_traj_no_pert           --phase finetune
python -m immune_world.cli.train experiment=ablation_reduced_corpus            --phase finetune
python -m immune_world.cli.train experiment=ablation_loco_cancer               --phase finetune
python -m immune_world.cli.train experiment=cohort_size_scan                   --phase finetune
```

Hyperparameter-sensitivity sweeps (Table S3):
```
python -m immune_world.cli.train experiment=sensitivity_lr
python -m immune_world.cli.train experiment=sensitivity_alpha
python -m immune_world.cli.train experiment=sensitivity_d
python -m immune_world.cli.train experiment=sensitivity_H
python -m immune_world.cli.train experiment=sensitivity_L
```

## 5. Evaluation

Each table maps to a single evaluation command; every quoted figure is the paper-reported value ±
its reported tolerance.

```
# Table 1 — 4-task headline metrics
python -m immune_world.cli.eval experiment=main --output-json metrics/table1.json
# Expected (seeds {42, 123, 456}):
#   Norman   Pearson r  = 0.914 ± 0.008
#   Adamson  Pearson r  = 0.821 ± 0.011
#   K562     Pearson r  = 0.712 ± 0.015
#   RPE1     Pearson r  = 0.783 ± 0.013
#   CBDir (3-set mean) = 0.881
#   DTW   (3-set mean) = 1.387
#   Deconv  F1         = 0.730 ± 0.019
#   ICB    AUC (14 co.) = 0.891 ± 0.014

# Table 2 — ICB AUC by cancer type + deconvolution F1 by cell type
python -m immune_world.cli.eval experiment=icb_response           --output-json metrics/table2_icb.json
python -m immune_world.cli.eval experiment=deconvolution_pancreas --output-json metrics/table2_deconv.json

# Table 3 — ablation matrix
python -m immune_world.cli.eval experiment=ablation_no_traj_attention ...

# Table 4 — leave-one-cancer-out + training-cohort scan
python -m immune_world.cli.eval experiment=ablation_loco_cancer
python -m immune_world.cli.eval experiment=cohort_size_scan

# Table 5 — compute cost benchmark
python -m immune_world.cli.eval experiment=compute_benchmark

# Supplementary S1 – S8
python scripts/failure_analysis.py   --results metrics/table1.json    --output-csv metrics/table_s4.csv
python scripts/training_dynamics.py  --output-csv metrics/table_s6.csv
```

All statistics reported follow Table S8: paired t-tests over 3 seeds for Pearson r / F1,
DeLong for AUC comparisons, 1 000-iteration bootstrap for CBDir / DTW (95 % CI).

## 6. Compute budget

Numbers are reproduced exactly as reported in the paper (§2.1 L220, §2.8 Table 5, §4.6 L692–704):

| Phase | GPU | Count | VRAM / GPU | Wall-clock | GPU-hours | Storage |
|---|---|---|---|---|---|---|
| Pretraining | NVIDIA A100 (80 GB) | 4 | ~60 – 65 GB mixed-precision | 72 h | 288 | ≥ 250 GB working disk |
| Fine-tuning (per task) | NVIDIA A100 (80 GB) | 1 | < 20 GB | 2 – 6 h | 2 – 6 | — |
| Inference (14 200 cells / s) | NVIDIA A100 or RTX 3090 | 1 | 6.2 GB peak | — | — | — |

We do **not** ship a reduced-budget "quick-start" alternative. The closest thing is
`configs/experiment/_smoke.yaml`, which is strictly for pytest smoke tests.

## 7. Checkpoints

Released after acceptance. Per §Code Availability of the paper (verbatim):

> Code for ImmuneWorld, including model weights, training scripts, and evaluation pipelines,
> will be made available upon acceptance at github.

Placeholder layout (to be filled in once the Zenodo / HuggingFace release is minted):

| Checkpoint | Venue | SHA-256 |
|---|---|---|
| `immune_world_pretrain_12p4m.ckpt` | Zenodo (DOI TBD) | `<sha256>` |
| `immune_world_finetune_perturbation_norman.ckpt` | Zenodo (DOI TBD) | `<sha256>` |
| `immune_world_finetune_icb_response.ckpt` | HuggingFace (TBD) | `<sha256>` |

Until release, checkpoints are available upon reasonable request from the corresponding author.


## Ethics

This is a purely computational, retrospective analysis on publicly available, de-identified
datasets. No new human or animal data were collected and no ethics approval was required for this
study. Immunotherapy-response predictions are evaluated with leave-one-cohort-out cross-validation
only and do not constitute prospective clinical validity.
