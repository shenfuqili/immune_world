# Repo Plan

Final directory tree, module-level responsibilities, pinned dependencies, and expected test coverage for the `immune_world` project. This file is the acceptance criterion for the scaffolding stage.

## 1. Final directory tree

```
immune_world/
├── README.md                           # reviewer-facing (8 sections per protocol)
├── LICENSE                             # MIT (standard; no clinical-use restriction beyond the paper's ethics frame)
├── pyproject.toml                      # build system, ruff, black, isort, mypy, pytest config
├── requirements.txt                    # pinned runtime deps
├── environment.yml                     # conda-equivalent
├── Dockerfile                          # CUDA 12.1 + PyTorch 2.3 + flash-attn 2.6
├── Makefile                            # `make test`, `make lint`, `make docker`
├── .gitignore
├── .pre-commit-config.yaml             # ruff + black + isort + mypy
├── .github/
│   └── workflows/
│       └── ci.yml                      # ruff + mypy + pytest on _smoke.yaml (CPU)
│
├── configs/
│   ├── model/
│   │   └── base.yaml                   # L=12, d=512, H=8, SwiGLU, TrajAttn
│   ├── data/
│   │   ├── norman.yaml                 # GSE133344, 91,205 cells, 284 perts
│   │   ├── adamson.yaml                # GSE90546, 68,603 cells, 87 perts
│   │   ├── replogle_k562.yaml          # GSE132080 K562 branch, 162,751 cells, 1,093 perts
│   │   ├── replogle_rpe1.yaml          # GSE132080 RPE1 branch, 162,733 cells, 1,544 perts
│   │   ├── pancreas.yaml               # scVelo Pancreas, 3,696 cells
│   │   ├── dentate_gyrus.yaml          # scVelo DG, 2,930 cells
│   │   ├── bone_marrow.yaml            # scVelo BM, 5,780 cells
│   │   ├── icbatlas.yaml               # 14 cohorts / 7 cancers / 2,834 patients
│   │   ├── pretrain_corpus.yaml        # HCA 5.2M + CellxGene 4.8M + GEO 2.4M = 12.4M
│   │   └── deconv_synthetic_bulk.yaml  # 350/50/100 Dirichlet mixtures
│   ├── train/
│   │   ├── base.yaml                   # optim + scheduler + logging defaults
│   │   ├── pretrain.yaml               # lr=1e-4 / bs=512 / 200ep / 4×A100 / 72h
│   │   └── finetune.yaml               # lr=3e-4 / bs=256 / patience=10 / seeds {42,123,456}
│   └── experiment/
│       ├── main.yaml                   # the paper's primary result
│       ├── pretrain.yaml               # the pretraining run
│       ├── perturbation_norman.yaml
│       ├── perturbation_adamson.yaml
│       ├── perturbation_k562.yaml
│       ├── perturbation_rpe1.yaml
│       ├── trajectory_pancreas.yaml
│       ├── trajectory_dentate_gyrus.yaml
│       ├── trajectory_bone_marrow.yaml
│       ├── deconvolution_pancreas.yaml
│       ├── icb_response.yaml
│       ├── ablation_no_traj_attention.yaml
│       ├── ablation_no_cross_cancer_transfer.yaml
│       ├── ablation_no_perturbation_engine.yaml
│       ├── ablation_no_gene_embed_pretrain.yaml
│       ├── ablation_no_temporal_encoding.yaml
│       ├── ablation_no_traj_no_pert.yaml
│       ├── ablation_reduced_corpus.yaml
│       ├── ablation_loco_cancer.yaml
│       ├── cohort_size_scan.yaml
│       ├── sensitivity_lr.yaml         # Table S3 lr row
│       ├── sensitivity_alpha.yaml      # Table S3 α row
│       ├── sensitivity_d.yaml          # Table S3 d row
│       ├── sensitivity_H.yaml          # Table S3 H row
│       ├── sensitivity_L.yaml          # Table S3 L row
│       ├── compute_benchmark.yaml      # Table 5 throughput / memory / params
│       ├── supplementary_failure_analysis.yaml   # Table S4
│       └── _smoke.yaml                 # pytest only; NOT for reporting
│
├── src/
│   └── immune_world/
│       ├── __init__.py                 # __version__, get_logger()
│       ├── __main__.py                 # routes to cli.train
│       ├── py.typed                    # PEP 561 marker
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── anndata_io.py           # AnnData / HDF5 loaders
│       │   ├── preprocessing.py        # HVG (top-2000), normalisation, log1p
│       │   ├── pseudotime.py           # diffusion pseudotime + entropy filter
│       │   ├── trajectories.py         # trajectory windows; scVelo dataset wrappers
│       │   ├── perturbation.py         # Perturb-Seq dataset; perturbation-identity split
│       │   ├── icb.py                  # ICB bulk RNA-seq dataset; LOOCV split
│       │   ├── synthetic_bulk.py       # Dirichlet synthetic bulk
│       │   ├── pretrain_corpus.py      # HCA + CellxGene + GEO assembly
│       │   └── collate.py              # batch collation + MLM gene masking
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── gene_embedding.py       # Eq. 2
│       │   ├── trajectory_attention.py # Eq. 4
│       │   ├── swiglu_ffn.py           # Eq. 5
│       │   ├── transformer_layer.py    # Eq. 3
│       │   ├── transformer.py          # L=12 stack with causal mask
│       │   ├── cross_cancer_transfer.py# Eq. 6 + GRL
│       │   ├── perturbation_engine.py  # Eq. 7 + Eq. 8
│       │   ├── world_simulator.py      # top-level ImmuneWorld (Eq. 1)
│       │   └── heads/
│       │       ├── __init__.py
│       │       ├── perturbation.py     # linear R^{d×G}
│       │       ├── trajectory.py       # trajectory-aware decoder
│       │       ├── deconvolution.py    # softmax-normalised linear
│       │       └── icb.py              # 2-layer MLP, dropout 0.1
│       │
│       ├── losses/
│       │   ├── __init__.py
│       │   ├── trajectory.py           # L_traj = MSE on next-state
│       │   ├── reconstruction.py       # L_recon = BCE on masked genes
│       │   ├── perturbation.py         # L_pert
│       │   ├── composite.py            # Eq. 9
│       │   └── cross_cancer.py         # GRL-based adversarial loss
│       │
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── perturbation.py         # Pearson r, MAE, prec / recall / spec
│       │   ├── trajectory.py           # CBDir, DTW
│       │   ├── deconvolution.py        # F1 at 5% threshold
│       │   ├── icb.py                  # AUC + 95% CI
│       │   └── statistics.py           # DeLong / paired-t / bootstrap 1000
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── pretrain.py             # pretraining orchestrator
│       │   ├── finetune.py             # per-task fine-tune orchestrator
│       │   ├── optim.py                # AdamW + cosine warm-up builder
│       │   ├── distributed.py          # DDP init / barrier / cleanup
│       │   ├── amp.py                  # GradScaler / bf16 setup
│       │   ├── checkpoint.py           # atomic save / load with seed persistence
│       │   └── trainer.py              # unified Trainer class
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── runner.py               # top-level evaluator dispatching to task-specific evals
│       │   ├── perturbation_eval.py    # Table 1 perturbation columns + Table S1
│       │   ├── trajectory_eval.py      # Table 1 CBDir/DTW
│       │   ├── deconvolution_eval.py   # Table 1/2 F1
│       │   ├── icb_eval.py             # Table 1/2 AUC
│       │   ├── ablation_eval.py        # Table 3
│       │   ├── ablation_interaction.py # Table S2
│       │   ├── cross_cancer_eval.py    # Table 4
│       │   ├── compute_benchmark.py    # Table 5
│       │   ├── extended_metrics.py     # Table S1
│       │   └── extra_baselines.py      # Table S7
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── seeding.py              # set_seed()
│       │   ├── logging.py              # stdlib logging config (never bare print)
│       │   ├── config.py               # Hydra / OmegaConf loader + validator
│       │   ├── registry.py             # datasets / models / heads registry
│       │   └── paths.py                # dataset path resolver with env overrides
│       │
│       └── cli/
│           ├── __init__.py
│           ├── train.py                # `python -m immune_world.cli.train`
│           ├── eval.py
│           ├── infer.py
│           ├── export_onnx.py
│           └── prepare_data.py
│
├── scripts/
│   ├── launch_train.sh                 # torchrun + srun snippets
│   ├── launch_eval.sh
│   ├── prepare_data.sh                 # wraps cli.prepare_data
│   ├── failure_analysis.py             # Table S4
│   ├── training_dynamics.py            # Table S6
│   ├── benchmark_compute.py            # Table 5
│   └── generate_figures.py             # Fig. 1, 5, 6, 7, 8
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # pytest fixtures: small adata, mock configs
│   ├── test_package_import.py          # sanity: package imports; __version__ is str
│   ├── data/
│   │   ├── test_anndata_io.py
│   │   ├── test_preprocessing.py
│   │   ├── test_pseudotime.py
│   │   ├── test_trajectories.py
│   │   ├── test_perturbation.py
│   │   ├── test_icb.py
│   │   ├── test_synthetic_bulk.py
│   │   ├── test_pretrain_corpus.py
│   │   └── test_collate.py
│   ├── models/
│   │   ├── test_gene_embedding.py
│   │   ├── test_trajectory_attention.py
│   │   ├── test_swiglu_ffn.py
│   │   ├── test_transformer_layer.py
│   │   ├── test_transformer.py
│   │   ├── test_cross_cancer_transfer.py
│   │   ├── test_perturbation_engine.py
│   │   ├── test_world_simulator.py
│   │   └── heads/
│   │       ├── test_perturbation_head.py
│   │       ├── test_trajectory_head.py
│   │       ├── test_deconvolution_head.py
│   │       └── test_icb_head.py
│   ├── losses/
│   │   ├── test_trajectory_loss.py
│   │   ├── test_reconstruction_loss.py
│   │   ├── test_perturbation_loss.py
│   │   ├── test_composite_loss.py
│   │   └── test_cross_cancer_loss.py
│   ├── metrics/
│   │   ├── test_perturbation_metrics.py
│   │   ├── test_trajectory_metrics.py  # CBDir, DTW against synthetic ordering
│   │   ├── test_deconvolution_metrics.py
│   │   ├── test_icb_metrics.py
│   │   └── test_statistics.py          # DeLong vs reference value, bootstrap CI width
│   ├── training/
│   │   ├── test_optim.py
│   │   ├── test_checkpoint.py          # round-trip; seed restores; atomic replace
│   │   └── test_training_smoke.py      # E2E 2 steps on _smoke.yaml, loss decreases
│   ├── evaluation/
│   │   └── test_runner_shape.py        # metrics-dict schema matches paper tables
│   └── cli/
│       └── test_cli_entrypoints.py     # import + argparse no-op
│
├── docs/
│   ├── project-context.md              # scaffolding stage
│   ├── implementation-map.md           # scaffolding stage
│   ├── repo-plan.md                    # scaffolding stage
│   ├── deviations.md                   # stub; populated during implementation
│   ├── data.md                         # companion to README §Data
│   ├── compute.md                      # companion to README §Compute
│   └── checkpoints.md                  # Zenodo / HF placeholders + SHA-256 schema
│
└── assets/                             # figures produced by scripts/generate_figures.py
    └── .gitkeep
```

---

## 2. Module-level responsibilities

| Package | Responsibility | Public API surface |
|---|---|---|
| `data/` | Convert raw scRNA-seq / bulk RNA-seq / Perturb-Seq → tensors with correct trajectory ordering, cohort labelling, perturbation context | `PerturbSeqDataset`, `TrajectoryDataset`, `ICBDataset`, `SyntheticBulkDataset`, `PretrainCorpus`, `diffusion_pseudotime`, `collate_trajectory_batch` |
| `models/` | Paper's architecture (§4.2–§4.5) — gene embedding, trajectory-aware transformer, cross-cancer transfer, perturbation engine, 4 task heads | `ImmuneWorld`, `TrajectoryAwareTransformer`, `TrajAttention`, `SwiGLU_FFN`, `CrossCancerHead`, `PerturbationEngine` |
| `losses/` | Composite objective (Eq. 9) + its branches + adversarial cross-cancer loss | `CompositeObjective`, `TrajectoryMSELoss`, `MaskedGeneBCELoss`, `PerturbationPredictionLoss`, `CrossCancerAdversarialLoss` |
| `metrics/` | Quantitative evaluation primitives matching paper tables + stats tests | `pearson_r`, `cbdir`, `dtw`, `f1_at_threshold`, `auc_loco`, `delong_test`, `bootstrap_ci` |
| `training/` | Pretraining + fine-tuning orchestration with DDP / AMP / checkpoint / schedulers | `Trainer`, `Pretrainer`, `FineTuner`, `save_checkpoint`, `load_checkpoint`, `build_optimizer`, `build_scheduler` |
| `evaluation/` | Reproduce every reported table / ablation from model outputs | `run_full_evaluation`, `run_ablation_matrix`, `run_loco_cancer`, `benchmark_compute` |
| `utils/` | Seed management, logging, config loading, path resolution, registry | `set_seed`, `get_logger`, `load_config`, `register`, `resolve_dataset_path` |
| `cli/` | User-facing entry points (train / eval / infer / export_onnx / prepare_data) | one `main(cfg)` per sub-CLI; dispatched from `__main__` |

---

## 3. Pinned dependencies

### `requirements.txt` (runtime)

```
# Core ML
torch==2.3.1
flash-attn==2.6.3

# Config
hydra-core==1.3.2
omegaconf==2.3.0

# Single-cell
anndata==0.10.7
scanpy==1.10.1
scvelo==0.3.2

# Numerics
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
scikit-learn==1.4.2

# Metrics helpers
lifelines==0.28.0            # optional — Cox partial likelihood
dtw-python==1.3.1            # DTW metric
pingouin==0.5.4              # paired-t + effect size

# I/O
h5py==3.11.0
zarr==2.18.0
fsspec==2024.3.1
tqdm==4.66.2

# Logging
rich==13.7.1
```

### `requirements-dev.txt`

```
pytest==8.2.0
pytest-cov==5.0.0
pytest-xdist==3.6.0
ruff==0.4.4
black==24.4.2
isort==5.13.2
mypy==1.10.0
pre-commit==3.7.0
types-pyyaml==6.0.12.20240311
types-tqdm==4.66.0.20240417
```

### `environment.yml` (conda superset)

Mirrors `requirements.txt` + `requirements-dev.txt`; pins `python=3.11`, `cuda-toolkit=12.1`, `cudnn=8.9`.

### `Dockerfile` base image

```
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
```
— matches PyTorch 2.3 + flash-attn 2.6 compiled wheels.

---

## 4. Expected test coverage

| Package | Lines target | Key invariants tested |
|---|---|---|
| `data/*` | ≥ 85 % | Shape of tensors / split determinism / mask-rate / pseudo-time entropy filter cuts ≥ 0 cells |
| `models/*` | ≥ 90 % | Forward-shape / causal-mask correctness / gradient-flow through GRL / FlashAttention fallback path / param-count ≈ 68 M within ±5 % |
| `losses/*` | ≥ 95 % | Non-negativity / scale invariance / λ_1=0.5, λ_2=0.3 wiring |
| `metrics/*` | ≥ 95 % | Pearson r on identical vectors = 1 / CBDir on reversed ordering = −1 / DTW on identical series = 0 / F1 at threshold against sklearn baseline |
| `training/*` | ≥ 80 % | Checkpoint round-trip / seed restoration / 2-step smoke on `_smoke.yaml` shows loss decrease |
| `evaluation/*` | ≥ 80 % | Metrics-dict schema matches the column set of Tables 1 / 2 / 3 / 4 / 5 / S1–S8 |
| `utils/*` | ≥ 95 % | `set_seed` restores torch/np/random/cuda RNG; `get_logger` returns stdlib Logger |
| `cli/*` | smoke only | `python -m immune_world.cli.train --help` exits 0 |

CI runs `pytest -q tests/` on CPU-only `_smoke.yaml`. GPU-dependent tests (flash-attn forward, mixed-precision smoke) are marked `@pytest.mark.gpu` and skipped in CI but reachable via `pytest -m gpu`.

---

## 5. Style / toolchain

| Tool | Config location | Scope |
|---|---|---|
| `ruff` | `[tool.ruff]` in `pyproject.toml` | lint + import-sort; rules: `E,F,I,N,UP,B,A,C4,SIM,TCH,RUF` |
| `black` | `[tool.black]` in `pyproject.toml` | 100-col line length |
| `isort` | `[tool.isort]` in `pyproject.toml` | black-compatible |
| `mypy` | `[tool.mypy]` in `pyproject.toml` | `strict = true` scoped to `src/immune_world/**` |
| `pytest` | `[tool.pytest.ini_options]` in `pyproject.toml` | `testpaths = ["tests"]`, `addopts = "-ra -q"`, markers `gpu` / `slow` |
| `pre-commit` | `.pre-commit-config.yaml` | ruff + black + isort + mypy + check-yaml + end-of-file-fixer |

---

## 6. Acceptance gate

`docs/repo-plan.md` is accepted at the scaffolding stage. Scaffolding succeeds when:

1. The full tree above is materialised (files may be empty or stub-only; types must be declared).
2. `pip install -e .[dev]` succeeds.
3. `pytest -q` passes (at least one sanity test; others will be `pytest.mark.skip("stub")` until implementation lands).
4. `ruff check .` is clean.
5. `mypy --strict src/immune_world` is clean on stubs.
6. `docker build -t immune_world:dev .` succeeds.
