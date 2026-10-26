# Continuous laminar BOLD deveining (experimental)

CLI: `ffs_laminar_tdv` (on main since 2026-10-25), or
`python -m fastfuncstuff.cli.laminar_tdv` before reinstalling entry points.

This implementation passes physiological and synthetic recovery tests, but **does
not improve GE–SE agreement on the first four-subject real-data experiment**.
It is an experimental inverse, not a validated preprocessing recommendation.

## What it estimates

The existing Havlicek–Uludag model (`laminar/forward.py`) supplies the venule and
ascending-vein state equations and GE observation equation. Exact Jacobians at
rest give a continuous vascular state-space model. The unknown inputs are
relative CBF changes at each vascular depth, with metabolism coupled through the
model's n-ratio. The neuronal and NVC stages are bypassed: fast neuronal activity
is not needed to correct slow BOLD transport.

For frequency f, the forward transfer is `H(f) = C (i 2πf I − A)⁻¹ B`. The local
target keeps the diagonal of H, retaining each depth's response to its own CBF,
including its own ascending-vein contribution. The microvascular target uses only
the venule contribution to the observation equation, keeping the original GE
normalization H0. Neither target modifies or disconnects the flow equations.

The inverse estimates continuous CBF histories with a ridge penalty increasing
as `1+(f/cutoff)^4`. The correction is `observed − predicted(full−target)`, so
unexplained residuals remain in the output. Input means and the specified
polynomial subspace are preserved. This can still remove true local activity if
the physiological or spatial model is wrong.

This is a **linearization around rest**, with band-limited vascular inputs. It is
not the full nonlinear Variational-Laplace inversion, and does not estimate
vascular physiology automatically. Defaults reuse `laminar.pipeline.reference_p0`;
`-vascular_json` accepts baseline overrides (`p0`) and scalar parameter
log-deviations (`parameters`). Regional calibration/posterior uncertainty is still
needed before reliable per-column application can be claimed.

## Spatial model and regularization

An explicit `-columns` label map is preferred. Otherwise the tool partitions GM
into 5 mm patches reached through connected GM voxels. The segmentation fallback
cannot resolve bridges between touching banks or identify individual vessels.

Depth interpolation and averaging are represented explicitly by matrices R and S.
The forward model sees `S R H`; known shared-voxel noise has covariance `S Sᵀ`,
with 5% diagonal shrinkage for stable whitening. This describes the sampling
weights, not the true subvoxel PSF or NORDIC's full noise covariance. A patch must
sample every model depth or it passes through unchanged, marked invalid.

Default: five vascular CBF input depths; these are not five independent neuronal
layers. Extra sampled bins cannot create independent spatial information.

With repeated runs that have identical stimulus timing, choose a regularization
value per patch by leave-one-GE-run-out full-response prediction. SE is withheld
from selection. For rest or unmatched tasks, give one `-ridge` value; this
avoids cross-validating against unrelated activity. Fixed-strength runs currently
also need equal lengths (process unequal runs separately).

Reflection padding reduces FFT wraparound. Inspect or exclude boundaries; the
benchmark discards the first and last 60 s. The `noise_gain` map is a conditional
white-noise gain at sampled depth profiles, copied across its patch. It is not a
voxelwise uncertainty interval. `-transfer_mode static` replaces every transfer
frequency by the DC transfer as an amplitude-only control with the same inverse
regularization framework.

## Usage

Surfaces are the usual input (2026-10-26): columns are geodesic midthickness
patches through each voxel's owning vertex, built by the same
`processing/hrftdv.py:ribbon_patches` that `ffs_hrftdv` uses, with equivolume
depth. `-te`/`-b0` come from the BIDS sidecar or must be given. `ffs_laminar_tdv
-help` documents every flag, including what `-ridge` (formerly `-strengths`) means.

```sh
ffs_laminar_tdv -input ge_run-0{1,2,3}.nii.gz -fs_subj $SUBJECTS_DIR/sub-01 \
  -repeated_runs -prefix ltdv/sub-01
```

Segmentation depth maps still work instead of surfaces; then depth order is
mandatory (the ds003427 maps are 0=pial, 1=white):

```sh
ffs_laminar_tdv -input ge_run-0{1,2,3}.nii.gz \
  -depth depth.nii.gz -gmfrac gmfrac.nii.gz -depth_order pial_to_white \
  -repeated_runs -te 0.023 -b0 7 -device cpu,4 -prefix tdv
```

Outputs:

- `_corrected_run-NN`: corrected BOLD in original signal units;
- `_removed_psc_run-NN`: estimated transported contribution in PSC;
- `_fit_psc_run-NN` (`-save_fit`): full-model reconstruction, a smoothing-only
  control (was `_smooth_psc`, always written);
- `_ltdv` (was `_diagnostics`): ridge, GE prediction R², profile noise gain,
  removed RMS, valid mask, patch ID, and depth (0 white, 1 pial);
- `_metadata.json`: parameters, conventions, input files and coverage.

The fitted-model control omits voxel residuals and is spatially pooled. Any gain
from it cannot be attributed solely to vascular correction.

## ds003427 external validation, 2026-10-24

Source: `/mnt/belegost/openneuro_data/ds003427/derivatives/autoproc/sub-XX_nordicA`.
These NORDIC runs already occupy the same GE/SE grid. The segmentation depths were
rebuilt with aligned MP2RAGE (`presurfer_al`) after the alignment/frame fixes;
older maps in `old_pre1014` are not used. No registration was repeated.

Three GE and three SE runs per subject, TR 2.5 s, 102 volumes, 0.8 mm voxels,
15 s checkerboard blocks. Each run is converted to PSC and detrended separately,
then the three SE runs are averaged. GE-only lambda selection uses three runs.
TE comes from each GE acquisition sidecar. Shared coverage, brain/GM fraction,
brightness floors, and slab-edge exclusion are fixed before scoring. Both all-GM
and GE-positive (uncorrected t>3) selections are reported. Positive-mask results
are a descriptive comparison, not independent activation inference.

Median matched-voxel temporal correlation with averaged SE, GE-positive valid GM:

| Subject | Original GE | Local target | Microvascular target | Model smoothing |
|---|---:|---:|---:|---:|
| 03 | .100 | .083 | .067 | .134 |
| 04 | .094 | .082 | .067 | .118 |
| 05 | .117 | .096 | .077 | .158 |
| 07 | .088 | .080 | .069 | .101 |

Spatial correlation of block-beta maps with SE also decreases overall. A small
outer-depth gain occurs in sub-07 (local target: spatial .332→.347), but the other
three subjects deteriorate there. Sub-03 checks at 3/7 depths and ascending-vein
slopes .2/.8 do not change the direction. The static control also loses
(sub-03 temporal .100→.081, versus .083 for dynamic correction). Float32 and
float64 summary temporal correlations agree within 6e-8 in a sub-03 check.
These are sensitivity checks, not
hyperparameters selected against SE.

Interpretation: the implementation can undo transport in its own physiological
regime, including independently integrated nonlinear test trajectories. With
fixed physiology and coarse segmentation patches, however, subtracting the
predicted transported component reduces real shared task signal. The experiment
does not establish whether physiological misspecification, spatial pooling,
residual noise, or their combination dominates. It does reject treating a
flatter laminar profile as sufficient evidence of successful deveining.

The next substantive experiment is to calibrate constrained vascular parameters
on independent strong responses, propagate parameter uncertainty, and use
surface-defined columns/overlap where available. The current model also lacks a
lateral pial-vein network, so it cannot trace remote pial sources.

Reproduce:

```sh
python scripts/bench_laminar_tdv_se.py -output /tmp/laminar_tdv/results -save_means
python scripts/plot_laminar_tdv_se.py /tmp/laminar_tdv/results/report.json \
  /tmp/laminar_tdv/results/depth_validation.png
# GE-only choices; SE is the referee in all variants:
python scripts/bench_laminar_tdv_se.py -subjects 03 -targets local -depths 3 \
  -output /tmp/laminar_tdv/K3
python scripts/bench_laminar_tdv_se.py -subjects 03 -targets local -s_d 0.8 \
  -output /tmp/laminar_tdv/slope08
```

Physics tests check stable dynamics, upward-only leakage, microvascular locality,
nonlinear equilibrium and time-varying recovery, deepest-depth preservation,
identity when transport is absent, and noise-only run splits. Volume tests check
GM connectivity, residual/baseline handling, explicit depth conventions and the
end-to-end CLI.

## Verification and saved artifacts

All 11 new tests pass; Ruff and ty pass for the changed modules/scripts. The first
fast-tier attempt had 7,770 passes plus test-package import and sandbox IPC errors.
All 27 affected files passed on rerun with repo PYTHONPATH and IPC access (172
passes). No GPU tests were run; CUDA was unavailable in this execution environment.

Saved averaged PSC volumes, paired voxel scores, JSON summaries and the depth
figure live at
`/mnt/belegost/openneuro_data/ds003427/derivatives/laminar_tdv/2026-10-24/`.
The working scratch directory is `/tmp/laminar_tdv/`.

## Events-based physiology cross-validation (2026-10-26)

With `-events`/`-onsets`, `-cv_p0 NAME V ...` (default `s_d 0.2 0.4 0.8 × t0v
0.5 1 2`) scores fixed physiologies by leave-one-run-out: each run's events,
through the model's own linearised stimulus→CBF response (`cbf_kernel`, peak
2.0 s), give each depth a CBF source with free per-depth, per-condition
amplitudes; the transfer matrix predicts the held-out run (schedules may
differ). The best pooled held-out R² is used; the correction itself stays
event-free. Simulation (`ident.py`-style, K=5): t0v separates strongly (R²
0.50 vs 0.41/0.34); s_d barely (≤0.003 R² even at TR 1, low noise), because free
per-depth amplitudes absorb drainage strength when local and drained shapes
are alike — only pooling over many patches resolves it. Caveat: the CBF shape
is fixed, so on real data t0v also absorbs HRF-shape mismatch — a chosen t0v
is fit-to-shape, not measured transit time. Events cannot choose `-ridge`.

## Per-voxel transport gains (2026-10-26)

Why: on MindsEye 7T (sub-ME462566, 0.8 mm, TR 2.5, 6 runs, `-ridge 0.8`) the
column-level correction removed ~10% of a ribbon voxel's PSC SD, and 0.77%
against 8.5% in the top-1% |Coef| voxels; 46.5% of those voxels lie outside the
ribbon. A column correction gives every voxel at a depth the same removal, so a
vein voxel keeps its excess. Physiology cannot fix that: across s_d 0.1–1.6 the
model's pial leak share only moves 0.54→0.59.

What (`-removal voxel`, default): each voxel regresses its detrended PSC on its
column's local BOLD at its depth (free), transported BOLD (gain, ridged toward
1) and d(transport)/dt (shift, toward 0); ridge strength per voxel by
leave-one-run-out over `-gain_lambdas` (1e4 ≈ column model). Removed = gain ×
transport + shift × derivative. A shell `-extend_mm` (2 mm) past the ribbon
inherits each nearest ribbon voxel's column and depth (pial veins), and gets
gains but never builds a column.

Trap found building it: veins also corrupt the column. Unweighted, three 5×
vein voxels inflate the pial bin; the inverse explains the excess as LOCAL
signal, so normal voxels' local regressor carries transport and their gain
falls to 0.4. Cleaning the excess linearly and refitting is a fixed point
(normal voxels' negative excess exactly cancels the veins'). Precision-weighting
the profiles, min(1, (bin median SD / voxel SD)²), fixes it: gains 4.85 / 0.89
against truths 5 / 1 (`test_voxel_gains_remove_a_veins_excess_and_keep_local_signal`).

Caveats: the voxel's own data enters its column profile (mostly the local term);
a genuine local response that correlates with deep activity can be taken as
transport where the local regressor does not explain it — VASO is the referee.
