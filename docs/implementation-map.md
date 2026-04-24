# Implementation Map — paper → code

One-to-one mapping from every numbered equation, algorithm artefact (there are zero algorithm boxes — the paper specifies methods via prose + Eq. 1–9), figure, and table to the file / module that implements or reproduces it. Each row below will be realised in subsequent commits; this document is the acceptance criterion for "done" (scientific fidelity).

## Legend
- **§** = section number in the source manuscript
- **S#** = supplementary table / figure
- **Eq.** = numbered equation
- Module paths are relative to repo root `~/immune_world/`

---

## A. Methods (§4) — equations → core modules

| § | Eq. | File | Class / function | Notes |
|---|---|---|---|---|
| 4.1 | Eq. 1 `x̂_{t+1} = f_θ(x_{≤t}, c)` | `src/immune_world/models/world_simulator.py` | `ImmuneWorld.forward(x_seq, c) -> Tensor` | Top-level autoregressive transition; `c = ∅` → trajectory mode, `c ≠ ∅` → counterfactual |
| 4.2 | Eq. 2 `z_t = LayerNorm(x_t^T E + p_t)` | `src/immune_world/models/gene_embedding.py` | `GeneEmbedding(n_genes=2000, d_model=512)` | `E` initialised from Gene2Vec, retrained end-to-end; `p_t` = continuous sinusoidal (paper p.27 L622) — **not** discrete index positional |
| 4.3 | Eq. 3 (residual transformer layer) | `src/immune_world/models/transformer_layer.py` | `TransformerLayer(d_model, n_heads)` | `h^{(ℓ)} = TrajAttn(LN(h^{(ℓ-1)})) + h^{(ℓ-1)}`; `h^{(ℓ)} ← FFN(LN(h^{(ℓ)})) + h^{(ℓ)}`; pre-norm residual — standard |
| 4.3 | Eq. 4 `TrajAttn(Q,K,V) = softmax(QKᵀ/√d_k + B_temp) V`, `B_temp[s,t] = −α·|t_s − t_t|` | `src/immune_world/models/trajectory_attention.py` | `TrajAttention(d_model, n_heads=8)` with per-head learnable `α` (shape `[H]`, init 0.01 per Table S3 †) | Causal mask on `≤ t`; FlashAttention-2 path when available |
| 4.3 | Eq. 5 `FFN(h) = (W_1 h ⊙ SiLU(W_g h)) W_2`, `W_{1,g} ∈ R^{d×4d}`, `W_2 ∈ R^{4d×d}` | `src/immune_world/models/swiglu_ffn.py` | `SwiGLU_FFN(d_model, ffn_mult=4)` | Shazeer 2020 (ref [33]); `⊙` element-wise |
| 4.4 | Eq. 6 `h^shared = g_φ(h_i)`, `L_transfer = L_task − λ_adv · L_disc` | `src/immune_world/models/cross_cancer_transfer.py` | `CrossCancerHead(d_model)` with `GradientReversalFn` (`λ_adv=0.1`) | 2-layer MLP projection head; GRL via `torch.autograd.Function`; Ganin et al. 2016 (ref [34]) |
| 4.5 | Eq. 7 `c = (1/k) Σ e_{g_j} + W_c [dose_j; type_j]` | `src/immune_world/models/perturbation_engine.py` | `PerturbationEmbedder(shared_gene_E, n_pert_types=2)` | Mean-pool perturbed-gene embeddings + dose / type modulation (`type ∈ {KO, overexpression}` per p.28 L664–665) |
| 4.5 | Eq. 8 `h̃^{(ℓ)} = CrossAttn(h^{(ℓ)}, c) + h^{(ℓ)}` | `src/immune_world/models/perturbation_engine.py` | `PerturbationInjection(d_model, n_heads)` | Cross-attention: query=`h^{(ℓ)}`, key/value=`c`; injected per layer |
| 4.6 | Eq. 9 `L_total = L_traj + λ_1 L_recon + λ_2 L_pert`, λ_1=0.5, λ_2=0.3 | `src/immune_world/losses/composite.py` | `CompositeObjective(lambda_recon=0.5, lambda_pert=0.3)` | Three-branch MTL; per-branch scalars fixed from paper |
| 4.6 | `L_traj = (1/(T-1)) Σ ‖x̂_{t+1} − x_{t+1}‖²` | `src/immune_world/losses/trajectory.py` | `TrajectoryMSELoss` | MSE on next-state; T = trajectory length |
| 4.6 | `L_recon = −(1/G) Σ [x_g log x̂_g + (1−x_g) log(1−x̂_g)]` | `src/immune_world/losses/reconstruction.py` | `MaskedGeneBCELoss` | Applied only to masked gene positions (MLM-style) |
| 4.6 | `L_pert` on cells with known perturbation outcomes | `src/immune_world/losses/perturbation.py` | `PerturbationPredictionLoss` | MSE on ground-truth post-perturbation expression |
| 4.6 | Diffusion pseudotime on 50 PCs; drop cells with DPT entropy > 0.9 | `src/immune_world/data/pseudotime.py` | `diffusion_pseudotime(adata, n_pcs=50, entropy_cut=0.9)` | Wolf+Haghverdi DPT (ref [30]); used for trajectory datasets lacking ground-truth time |
| 4.6 | Dirichlet synthetic bulk, 500 cells/sample, F1 threshold 0.05 | `src/immune_world/data/synthetic_bulk.py` | `make_dirichlet_bulk(cells_per_mix=500)` | Paper p.30 L718–726 |
| 4.6 | Mean-pool cell embeddings per patient → 2-layer MLP, dropout 0.1 | `src/immune_world/models/heads/icb.py` | `ICBResponseHead(d_model, dropout=0.1)` | Aggregation + binary classification; §4.6 L699–700 |
| 4.6 | Linear projection `W_pert ∈ R^{d×G}` for perturbation head | `src/immune_world/models/heads/perturbation.py` | `PerturbationHead(d_model, n_genes)` | §4.6 L696 |
| 4.6 | Trajectory-aware decoder head | `src/immune_world/models/heads/trajectory.py` | `TrajectoryHead(d_model)` | §4.6 L697 |
| 4.6 | Softmax-normalised linear layer for cell-type proportion | `src/immune_world/models/heads/deconvolution.py` | `DeconvolutionHead(d_model, n_cell_types=8)` | §4.6 L722–726 |

---

## B. Results (§2) — tables → evaluation code

| § | Table | File | Class / function | Evaluated experiment config |
|---|---|---|---|---|
| 2.2–2.5 | **Table 1** — 4-task main comparison (Pearson r, CBDir, DTW, F1, AUC) | `src/immune_world/evaluation/runner.py` | `run_full_evaluation(cfg)` | `configs/experiment/main.yaml` |
| 2.4–2.5 | **Table 2** — per-cancer-type AUC + per-cell-type F1 | `src/immune_world/evaluation/icb_eval.py`, `deconvolution_eval.py` | `evaluate_icb_by_cancer(...)`, `evaluate_deconv_by_celltype(...)` | `main.yaml` (with per-cancer breakdown enabled) |
| 2.6 | **Table 3** — ablation (7 component drops + corpus reduction) | `src/immune_world/evaluation/ablation_eval.py` | `run_ablation_matrix(cfg)` | 8 `configs/experiment/ablation_*.yaml` |
| 2.7 | **Table 4** — leave-one-cancer-out + cohort-size scan | `src/immune_world/evaluation/cross_cancer_eval.py` | `run_loco_cancer(cfg)`, `cohort_size_scan(cfg)` | `ablation_loco_cancer.yaml`, `cohort_size_scan.yaml` |
| 2.8 | **Table 5** — compute cost (params, GPU-h, throughput, memory) | `src/immune_world/evaluation/compute_benchmark.py` | `benchmark_compute(model, batch=256, device)` | `compute_benchmark.yaml` |

---

## C. Supplementary Information (pp. 36–52) — SI tables

| Table | File | Function |
|---|---|---|
| **S1** Extended metrics (prec / recall / spec / MAE + per-dataset CBDir/DTW) | `src/immune_world/evaluation/extended_metrics.py` | `compute_extended_metrics(preds, targets, datasets)` |
| **S2** Two-way component-interaction ablation (Δ_obs − Δ_exp synergy) | `src/immune_world/evaluation/ablation_interaction.py` | `compute_two_way_interactions(results)` |
| **S3** Hparam sensitivity (lr × α × d × H × L × 7-point grid) | `configs/experiment/sensitivity_{lr,alpha,d,H,L}.yaml` | n/a — drives multiple training runs |
| **S4** Failure analysis (worst 10 samples Norman/K562/ICB) | `scripts/failure_analysis.py` | CLI; produces CSV table matching SI format |
| **S5** Dataset statistical summary (cells/patients/splits/perts/platform) | `docs/data.md` | documentation only — numbers sourced from code's dataset registry |
| **S6** Training dynamics (corpus-size × epoch grid) | `scripts/training_dynamics.py` | CLI; repeated pretraining on {3.1M, 6.2M, 9.3M, 12.4M} × checkpoint at {10,25,…,200} epochs |
| **S7** Extra baselines (Monocle3 / Slingshot / RNA-ODE / MuSiC / BisqueRNA / SCENIC+ / Linear / MLP / TIDE / PD-L1 / TMB) | `src/immune_world/evaluation/extra_baselines.py` | Thin wrappers calling upstream libs where available; analytical floors for marker-based TIDE / PD-L1 / TMB |
| **S8** Statistical significance tests (DeLong / paired t / bootstrap 1,000) | `src/immune_world/metrics/statistics.py` | `delong_test(y_true, score_a, score_b)`, `paired_t_test(a, b)`, `bootstrap_ci(metric_fn, a, b, n=1000)` |

---

## D. Figures — reproduction recipes

Figures are regenerated from evaluation outputs (never hand-drawn). `scripts/generate_figures.py` dispatches per-figure routines; each writes a PDF + PNG to `assets/`.

| Fig. | Description | Script entry point | Data source |
|---|---|---|---|
| 1 | TIME composition + ICB cohort breakdown + data pipeline | `generate_figure1(cfg)` | aggregates dataset registry + ICB metadata |
| 2 | Graphical abstract / three-core-modules schematic | hand-drawn in paper — **no code** | — |
| 3 | Architecture diagram (one transformer layer fully expanded) | hand-drawn — **no code** | — |
| 4 | Study-design / validation framework | hand-drawn — **no code** | — |
| 5 | Benchmark (age-scatter / UMAP / PCA / SHAP / volcano / Manhattan / corpus-size / gene-coex) | `generate_figure5(results)` | ICB evaluation + SHAP + gene-expression preprocessing |
| 6 | CD8+ T-cell state transitions / cohort-size scan / training dynamics / per-cancer AUC CI / trajectory metrics | `generate_figure6(results)` | trajectory eval + `cohort_size_scan` + `training_dynamics` |
| 7 | Multi-task benchmarking (Perturb-Seq violins / cell-type composition / log2FC / ablation waterfall) | `generate_figure7(results)` | full evaluation runner outputs |
| 8 | Methodological analysis (feature importance / training-loss surface / method ranking) | `generate_figure8(results)` | ICB eval + training history + bootstrap summaries |
| 9 | Mathematical formulation schematic | hand-drawn — **no code** | — |

---

## E. Training procedure (§4.6) — orchestration

| Phase | File | Class / function | Key hparams (see `docs/project-context.md §8`) |
|---|---|---|---|
| Pretraining | `src/immune_world/training/pretrain.py` | `Pretrainer(cfg)` | AdamW lr=1e-4, 5k-step linear warmup, cosine decay to 1e-6, 200 epochs, batch 512, 4× A100 80 GB |
| Fine-tuning (per task) | `src/immune_world/training/finetune.py` | `FineTuner(cfg)` | AdamW lr=3e-4, cosine annealing, weight decay 0.01, batch 256, patience 10, seeds {42,123,456} |
| Optim + scheduler | `src/immune_world/training/optim.py` | `build_optimizer(params, cfg)`, `build_scheduler(opt, cfg)` | AdamW + cosine with linear warmup |
| DDP init | `src/immune_world/training/distributed.py` | `init_distributed()`, `cleanup()` | `torch.distributed.init_process_group` |
| AMP | `src/immune_world/training/amp.py` | `build_grad_scaler(cfg)` | bf16 preferred on A100; fp16 fallback |
| Checkpoint | `src/immune_world/training/checkpoint.py` | `save_checkpoint(state, path)` (atomic via `os.replace`), `load_checkpoint(path)` | Stores seed + RNG state |
| Trainer orchestration | `src/immune_world/training/trainer.py` | `Trainer(model, loaders, optim, scheduler, ckpt_mgr)` | Unified entry for pretrain + fine-tune |

---

## F. Data pipeline (§2.1 / §4.6 / Table S5) — loaders

| Task | File | Class | Source |
|---|---|---|---|
| AnnData I/O + HVG selection (top 2000 across all sources) | `src/immune_world/data/anndata_io.py`, `preprocessing.py` | `load_anndata(path)`, `select_hvgs(adata, n=2000)` | §4.6 L688–689 |
| Perturb-Seq (Norman / Adamson / Replogle K562 / RPE1) | `src/immune_world/data/perturbation.py` | `PerturbSeqDataset`, `perturbation_identity_split(ratios=[0.8,0.1,0.1])` | §2.1 L197–198, Table S5 |
| Trajectory (Pancreas / Dentate Gyrus / Bone Marrow) | `src/immune_world/data/trajectories.py` | `TrajectoryDataset(via_scvelo)` | §2.1 L200–201 |
| ICB cohorts (14 cohorts, 7 cancers) | `src/immune_world/data/icb.py` | `ICBDataset`, `leave_one_cohort_out_split` | §2.1 L201–205, §4.6 L727–734 |
| Synthetic bulk (Dirichlet) | `src/immune_world/data/synthetic_bulk.py` | `SyntheticBulkDataset` | §4.6 L718–726 |
| Pretraining corpus assembly | `src/immune_world/data/pretrain_corpus.py` | `PretrainCorpus(hca, cellxgene, geo)` | §4.6 L684–687 |
| Collation (masked tokens, trajectory windows) | `src/immune_world/data/collate.py` | `collate_trajectory_batch`, `apply_gene_mlm_mask(rate=0.15)` | masking rate not stated — see `deviations.md` |

---

## G. CLI

| Entry point | File | Purpose |
|---|---|---|
| `python -m immune_world.cli.train experiment=<name>` | `src/immune_world/cli/train.py` | Drives pretrain + fine-tune runs |
| `python -m immune_world.cli.eval experiment=<name>` | `src/immune_world/cli/eval.py` | Runs evaluation, writes metrics JSON + LaTeX-table row |
| `python -m immune_world.cli.infer checkpoint=<path> input=<anndata>` | `src/immune_world/cli/infer.py` | Cell-state transition / counterfactual inference |
| `python -m immune_world.cli.export_onnx checkpoint=<path>` | `src/immune_world/cli/export_onnx.py` | ONNX export of main model graph |
| `bash scripts/prepare_data.sh` | `src/immune_world/cli/prepare_data.py` (main logic) | Downloads / caches datasets; writes SHA-256 manifests |

---

## H. Paper items NOT implemented in code (descriptive only)

- Figures 2, 3, 4, 9 are schematic / hand-drawn (no code mapping required).
- Introduction (§1) and Discussion (§3) are prose — no code mapping.
- Acknowledgements, Author Contributions, Competing Interests, Ethics Declaration (pp. 35–36) are mirrored into `README.md` §Citation / §Ethics.

---

## I. Acceptance gate

`docs/implementation-map.md` is accepted at the scaffolding stage. Subsequent completion requires every row in A–G to have:
1. the file and class/function created,
2. a unit test that exercises at least one shape / value / gradient path,
3. the top-of-file docstring citing the paper section (e.g., `"""Ref: Sec. 4.3, Eq. (4)."""`).

`tests/` column is populated 1-for-1 with this table in `docs/repo-plan.md`.
