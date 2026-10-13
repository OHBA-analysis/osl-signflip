# Reproducing the Analyses

This directory contains the scripts for reproducing the analyses and figures in our work on **new sign-flipping algorithms for the dipole sign-ambiguity problem** in parcellated M/EEG source reconstruction. For installing and using the `osl-signflip` package itself, see the [main README](../README.md).

## Table of Contents

- [Descriptions](#-descriptions)
- [Requirements](#-requirements)
- [Getting Started](#️-getting-started)
- [Project Structure](#-project-structure)
- [License](#-license)

## 💡 Descriptions

Group analyses of source-reconstructed brain networks require the arbitrary per-parcel dipole sign of each session to be made consistent, a problem equivalent to **Z₂ (sign) synchronization**. The existing approach is a slow, stochastic random search. Here we add three principled alternatives — **spectral synchronization**, **semidefinite programming (SDP)**, and **loopy belief propagation (BP)** — that share a common channel affinity matrix and an exact-metric greedy polish.

All four methods are evaluated on

- LEMON (EEG, rest)
- Cam-CAN (MEG, rest)
- Wakeman-Henson (MEG, task)

across static and dynamic (TDE-HMM) downstream features.

The headline result is an at least **~160× speed-up with full reproducibility at equal-or-better quality**, with SDP as the best method.

## 🎯 Requirements

All analysis scripts run in a single conda environment, `osld`, which contains two OHBA software packages:

| Package | Used by |
| :------ | :------ |
| [`osl-dynamics`](https://github.com/OHBA-analysis/osl-dynamics) (v3.5.1) | Static/dynamic feature extraction, TDE-HMM, and visualization (scripts `03`–`08`, `11`) |
| [`osl-signflip`](../src/osl_signflip) (installed from this repository) | Sign flipping, cost benchmark, and seed stability (scripts `01`, `09`, `10`) |

To set up the environment:

1. **Install `osl-dynamics`.** Create the `osld` conda environment by following the `osl-dynamics` [installation guide](https://osl-dynamics.readthedocs.io/en/latest/install.html).

2. **Install `osl-signflip` into the same environment.** Clone this repository and install `osl-signflip` from its root directory:

   ```bash
   git clone https://github.com/OHBA-analysis/osl-signflip.git
   cd osl-signflip
   conda activate osld
   pip install -e ".[mne]"
   ```

   This installs `osl-signflip` from the cloned source code (in editable mode) rather than from PyPI, so the scripts always use the version of the package that matches them. The `[mne]` option adds the MNE-Python helpers that the scripts use to read and write `.fif` files.

> [!IMPORTANT]
> The scripts locate the data through `BASE_DIR`, an absolute path to the root of this repository on our servers. Before running them, set `BASE_DIR` to the location of your clone. The data are expected in the per-dataset directories shown in [Project Structure](#-project-structure).

> [!NOTE]
> All the code within this repository was executed on the Oxford Biomedical Research Computing (BMRC) servers. While individual jobs were allocated varying CPUs and GPUs, general information about the BMRC resources can be found at [_Using the BMRC Cluster with Slurm_](https://www.medsci.ox.ac.uk/for-staff/resources/bmrc/using-the-bmrc-cluster-with-slurm) and [_GPU Resources_](https://www.medsci.ox.ac.uk/for-staff/resources/bmrc/gpu-resources).

## ⚡️ Getting Started

The scripts in this directory are numbered and meant to be run in order. Each takes a dataset name (`lemon`, `camcan`, `wakeman_henson`) and, where relevant, a sign-flipping method (`random`, `spectral`, `sdp`, `belief_propagation`). Please find an example below:

```bash
# 1. Apply sign flipping to the source-reconstructed data (one run per method)
python scripts/01_sign_flip.py camcan sdp

# 2. Collect after-flip metrics and recover the applied flip decisions
python scripts/02_collect_results.py camcan

# 3. Generate the figures on sign-flipping performance and flip-decision
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
│   ├── README.md                       # This file: analyses and paper reproduction
│   ├── 01_sign_flip.py                 # Apply sign flipping (random/spectral/BP/SDP) to source data
│   ├── 02_collect_results.py           # Collect after-flip metrics + recover flip decisions
│   ├── 03_plot_comparison.py           # Fig 1, 2, S1-S3: performance & flip-decision agreement
│   ├── 04_static_analysis.py           # Per-method static features (PSD/power/AEC + TDE covariance)
│   ├── 05_static_comparison.py         # Fig 4, S4-S5: static-feature comparison across methods
│   ├── 06_dynamic_analysis.py          # Train one TDE-HMM run per (dataset, method, run_id)
│   ├── 07_dynamic_features.py          # Aggregate best HMM run; match states to SDP
│   ├── 08_dynamic_comparison.py        # Fig 5, S6-S7, Table S3: dynamic-feature comparison
│   ├── 09_benchmark_cost.py            # Wall-clock + analytic FLOP cost benchmark
│   ├── 10_seed_stability.py            # Seed-to-seed variability of the randomized methods
│   ├── 11_cost_stability_comparison.py # Fig 3, S8: compute cost and seed stability
│   └── utils.py                        # Shared plotting, statistics, and I/O helpers
│
├── src/osl_signflip/                   # Standalone sign-flipping package (pip-installable)
├── tests/                              # osl-signflip test suite (pytest)
├── pyproject.toml                      # osl-signflip packaging metadata
├── CITATION.cff
│
├── {camcan, lemon, wakeman_henson}/    # Per-dataset directory (data; not tracked in git)
│   ├── src_{random, spectral, sdp, belief_propagation}/
│   │   ├── <session>/                  # Source-reconstructed, sign-flipped parcel data
│   │   │   ├── parc/lcmv-parc-raw.fif              # Pre-flip parcel time courses
│   │   │   └── <session>_sflip_lcmv-parc-raw.fif   # Sign-flipped parcel time courses
│   │   ├── src_report/                 # Per-session before/after metrics (written by 01)
│   │   └── hmm/run<id>/                # Trained TDE-HMM runs (from 06)
│   └── data/                           # Aggregated .npz results + summary_stats.txt
│
├── LICENSE
└── README.md                           # The osl-signflip package (also the PyPI page)
```

## 🪪 License

These scripts are part of `osl-signflip` and are licensed under the [MIT License](../LICENSE).
