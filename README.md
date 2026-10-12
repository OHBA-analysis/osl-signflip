# osl-signflip

This repository contains the scripts and results for reproducing the analyses in our work on **new sign-flipping algorithms for the dipole sign-ambiguity problem** in parcellated M/EEG source reconstruction.

🙋‍♂️ Please email SungJun Cho at sungjun.cho@ndcn.ox.ac.uk or simply raise GitHub Issues if you have any questions or concerns.

## Table of Contents

- [Descriptions](#-descriptions)
- [Requirements](#-requirements)
- [Getting Started](#️-getting-started)
- [Project Structure](#-project-structure)
- [License](#-license)

## 💡 Descriptions

Group analyses of source-reconstructed brain networks require the arbitrary per-parcel dipole sign of each subject to be made consistent, a problem equivalent to **Z₂ (sign) synchronization**. The existing approach is a slow, stochastic random search. Here we add three principled alternatives — **spectral synchronization**, **loopy belief propagation (BP)**, and a **semidefinite programming (SDP)** — that share a common channel affinity and an exact-metric greedy polish.

All four methods are evaluated on
- LEMON (EEG, rest)
- Cam-CAN (MEG, rest)
- Wakeman-Henson (MEG, task)

across static and dynamic (TDE-HMM) downstream features.

The headline result is an at least **~160× speed-up with full reproducibility at equal-or-better quality**, with SDP as the best solver.

The methods are available in two forms:
- **`osl-signflip`**, a lightweight standalone Python package (NumPy and tqdm only) whose source lives in [`src/osl_signflip/`](src/osl_signflip) (see [`src/README.md`](src/README.md) for usage), and
- [`osl-ephys`](https://github.com/OHBA-analysis/osl-ephys) (see `osl_ephys/source_recon/sign_flipping.py`), which was used to produce the results in the paper.

The two implementations give identical results, except that random search in `osl-signflip` also applies the global sign convention, so its sign assignments can differ from those of `osl-ephys` by a global sign (checked by `tests/test_equivalence_osl_ephys.py`). This repository additionally holds the experiment, analysis, and figure-generation scripts.

To install `osl-signflip` in a fresh conda environment:

```bash
conda create -n oslsf python=3.12
conda activate oslsf && pip install osl-signflip
```

To install it from this repository instead (e.g., for development):

```bash
pip install -e ".[mne,dev]"   # install osl-signflip from this repository
pytest                        # run its test suite
```

## 🎯 Requirements

The analyses rely on two OHBA software packages, each with its own conda environment:

| Package | Environment | Used by |
| :------ | :---------- | :------ |
| [`osl-ephys`](https://github.com/OHBA-analysis/osl-ephys) | `osle` (v2.4.0) | Source reconstruction and the sign-flipping solvers (scripts `01`, `09`, `10`) |
| [`osl-dynamics`](https://github.com/OHBA-analysis/osl-dynamics) | `osld` (v3.5.1) | Static/dynamic feature extraction, TDE-HMM, and visualization (scripts `03`–`08`, `11`) |

Please follow the installation guide [here](https://osl-ephys.readthedocs.io/en/latest/install.html) and [here](https://osl-dynamics.readthedocs.io/en/latest/install.html) to set up the environments. Once complete, clone this repository to your preferred directory and you're ready to begin.

> [!NOTE]
> All the code within this repository was executed on the Oxford Biomedical Research Computing (BMRC) servers. While individual jobs were allocated varying CPUs and GPUs, general information about the BMRC resources can be found at [_Using the BMRC Cluster with Slurm_](https://www.medsci.ox.ac.uk/for-staff/resources/bmrc/using-the-bmrc-cluster-with-slurm) and [_GPU Resources_](https://www.medsci.ox.ac.uk/for-staff/resources/bmrc/gpu-resources).

## ⚡️ Getting Started

The scripts in `scripts/` are numbered and meant to be run in order. Each takes a dataset name (`lemon`, `camcan`, `wakeman_henson`) and, where relevant, a sign-flipping method (`random`, `spectral`, `sdp`, `belief_propagation`). Please find an example below:

```bash
# 1. Apply sign flipping to the source-reconstructed data (one run per method)
python scripts/01_sign_flip.py camcan sdp

# 2. Collect after-flip metrics and recover the applied flip decisions
python scripts/02_collect_results.py camcan

# 3. Generate the performance and flip-decision figures
python scripts/03_plot_comparison.py

# 4-5. Static feature extraction (per method) and cross-method comparison
python scripts/04_static_analysis.py camcan sdp --n-workers 8
python scripts/05_static_comparison.py

# 6-8. Train TDE-HMMs (per method x run), aggregate, and compare dynamics
python scripts/06_dynamic_analysis.py camcan sdp 0
python scripts/07_dynamic_features.py camcan
python scripts/08_dynamic_comparison.py

# 9-11. Benchmark compute cost, measure seed stability, and plot both
python scripts/09_benchmark_cost.py camcan
python scripts/10_seed_stability.py camcan
python scripts/11_cost_stability_comparison.py
```

> [!NOTE]
> The raw MEG/EEG recordings and the large source-reconstructed outputs are not distributed with this repository. The datasets used are publicly available: [LEMON](https://www.nature.com/articles/sdata2018308), [Cam-CAN](https://cam-can.mrc-cbu.cam.ac.uk/dataset/), and [Wakeman-Henson](https://www.nature.com/articles/sdata20151).

## 📚 Project Structure

```
osl-signflip/
├── scripts/
│   ├── 01_sign_flip.py                 # Apply sign flipping (random/spectral/BP/SDP) to source data
│   ├── 02_collect_results.py           # Collect after-flip metrics + recover flip decisions
│   ├── 03_plot_comparison.py           # Fig 1, 3, S1-S3: performance & flip-decision agreement
│   ├── 04_static_analysis.py           # Per-method static features (PSD/power/AEC + TDE covariance)
│   ├── 05_static_comparison.py         # Fig 4, S4-S5: static-feature comparison across methods
│   ├── 06_dynamic_analysis.py          # Train one TDE-HMM run per (dataset, method, run_id)
│   ├── 07_dynamic_features.py          # Aggregate best HMM run; match states to SDP
│   ├── 08_dynamic_comparison.py        # Fig 5, S6-S7, Table S1: dynamic-feature comparison
│   ├── 09_benchmark_cost.py            # Wall-clock + analytic FLOP cost benchmark
│   ├── 10_seed_stability.py            # Seed-to-seed variability of the randomized methods
│   ├── 11_cost_stability_comparison.py # Fig 2, S8: compute cost and seed stability
│   └── utils.py                        # Shared plotting, statistics, and I/O helpers
│
├── src/osl_signflip/                   # Standalone sign-flipping package (pip-installable)
├── tests/                              # osl-signflip test suite (pytest)
├── pyproject.toml                      # osl-signflip packaging metadata
├── CITATION.cff
│
├── {camcan, lemon, wakeman_henson}/    # Per-dataset directory (data; not tracked in git)
│   ├── src_{random, spectral, sdp, belief_propagation}/
│   │   ├── <subject>/                  # Source-reconstructed, sign-flipped parcel data
│   │   │   ├── parc/lcmv-parc-raw.fif              # Pre-flip parcel time courses
│   │   │   └── <subject>_sflip_lcmv-parc-raw.fif   # Sign-flipped parcel time courses
│   │   ├── src_report/                 # Per-subject before/after metrics from the sign-flip wrapper
│   │   └── hmm/run<id>/                # Trained TDE-HMM runs (from 06)
│   └── data/                           # Aggregated .npz results + summary_stats.txt
│
├── LICENSE
└── README.md
```

## 🪪 License

Copyright (c) 2026 [SungJun Cho](https://github.com/scho97) and [OHBA Analysis Group](https://github.com/OHBA-analysis). `osl-signflip` is a free and open-source software licensed under the [MIT License](https://github.com/OHBA-analysis/osl-signflip/blob/main/LICENSE).
