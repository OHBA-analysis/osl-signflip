# osl-signflip

**Fast, reproducible sign-flipping for source-reconstructed M/EEG data.**

Source reconstruction of MEG/EEG data with unconstrained or data-driven source orientations leaves
each parcel time course with an arbitrary sign that can differ between recording sessions. `osl-signflip`
aligns these signs to a common template session so that group-level analyses (averaging, functional
connectivity, network models) are not distorted.

It casts sign-flipping as a **Z₂ synchronization** problem (equivalent to finding the ground state of an
Ising model) and solves it with:

| Method | `method=` | Notes |
| :-- | :-- | :-- |
| Semidefinite programming relaxation | `"sdp"` (default) | Recommended. Seeded and insensitive to the seed. |
| Spectral synchronization | `"spectral"` | Deterministic. |
| Loopy belief propagation | `"belief_propagation"` | Deterministic. |
| Random search | `"random"` | The stochastic baseline used by existing pipelines. |

The first three are followed by greedy refinement of the exact objective and run roughly 160 to
more than 2,000 times faster than random search while giving equally good or better solutions.

## 📌 Installation

```bash
pip install osl-signflip            # core (NumPy and tqdm only)
pip install "osl-signflip[mne]"     # + helpers for MNE-Python Raw/Epochs and .fif files
```

## ⚡️ Quick start

```python
import osl_signflip as oslsf

# data: list of (n_samples, n_parcels) arrays, one per session
flips, metrics, template = oslsf.align_signs(data, n_embeddings=15, method="sdp")
data_flipped = [oslsf.apply_flips(x, f) for x, f in zip(data, flips)]
```

`flips[s]` is a vector of ±1 per parcel for session `s`, and `metrics[s]` tracks the correlation between
the session and template covariances (first entry: before flipping; last entry: after).

To work directly with covariances, e.g. when aligning new sessions to an existing template:

```python
cov = oslsf.compute_covariance(x, n_embeddings=15)
flips, metrics = oslsf.find_flips(cov, template_cov, n_embeddings=15, method="sdp")
```

With MNE-Python objects (parcel channels named `parcel_X`, as written by `osl-ephys`):

```python
from osl_signflip import io

covs = io.load_covariances(["sub-01_parc-raw.fif", "sub-02_parc-raw.fif"], n_embeddings=15)
template = oslsf.find_template(covs, diag_offset=15)
flips, _ = oslsf.find_flips(covs[0], covs[template], n_embeddings=15)
raw_flipped = io.apply_flips(raw, flips)
```

## A note on the global sign

Sign-flipping uses covariances, which cannot distinguish a flip vector `f` from its global negation `-f`
(flipping every parcel of a session leaves its covariance unchanged). `osl-signflip` therefore fixes the
global sign by convention: it flips the fewer parcels (the flips sum to ≥ 0) and, if exactly half the
parcels would be flipped, it keeps the first parcel unflipped. This makes the output reproducible.

## Relationship to osl-ephys

The same methods are available in [`osl-ephys`](https://github.com/OHBA-analysis/osl-ephys) through
`osl_ephys.source_recon.fix_sign_ambiguity`, with identical hyperparameter names, defaults and results. The one difference is that `osl-signflip` also applies the global sign convention to random search, which `osl-ephys` returns as found; the two can therefore differ by a global sign for that method only.
`osl-signflip` is a lightweight, pipeline-independent alternative that works on NumPy arrays.

## Citation

If you use `osl-signflip`, please cite the accompanying paper (citation to be added).

## License

Copyright (c) 2026 [SungJun Cho](https://github.com/scho97) and [OHBA Analysis Group](https://github.com/OHBA-analysis). `osl-signflip` is a free and open-source software licensed under the [MIT License](https://github.com/OHBA-analysis/Cho2026_SignFlip/blob/main/LICENSE).
