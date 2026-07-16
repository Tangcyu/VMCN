# TensorQ

TensorQ now uses one shared package architecture:

- `tensorq.common`: YAML, dataset, CV featurization, lagged indexing, and rate helpers.
- `tensorq.next_hit`: next-hit committor training, inference, plotting, and rate estimation.
- `tensorq.pairwise`: pair-wise committor training, inference, plotting, and rate estimation.
- `tensorq.gradpath`: direct-channel gradient pathway shooting, weighted clustering, and plotting.
- `tensorq.voronoi_merge`: Voronoi shared-segment alignment and iterative KLD diagnostics.
- `tensorq.MSMlabel`: MSM/PCCA+ macrostate discovery and TensorQ core-label dataset export.

The shared dataset is produced by `scripts/label.py` from the `TENSORQ_LABEL` config section. Both committor families consume that same `.pt` or `.npz` dataset.

## Staged MSM-to-committor-vector workflow

`run.py` is the single dispatcher for the four-stage workflow. The default
`config.yaml` points to one focused configuration file per stage:

- `configs/0.MSMcorelabel.yaml`: weighted MSM, PCCA+, diagnostics, and core labels.
- `configs/1.Committorvector.yaml`: next-hit training, inference, plotting, rate constants, and optional rate fitting.
- `configs/2.Gradpath.yaml`: gradient pathfinding, weighted path clustering, and optional Voronoi merging.
- `configs/3.Relabel.yaml`: diagnostics plus entropy (`H`) and Gini (`G`) relabeling.

Run from the repository root:

```bash
python run.py --step 0       # MSM -> PCCA+ -> core-label dataset
python run.py --step 1       # next-hit committor vector
python run.py --step 2       # gradpath and Voronoi workflow
python run.py --step 3       # diagnose, H relabel, G relabel
```

`--step` also accepts the names `msmcorelabel`, `committorvector`, `gradpath`,
`relabel`, and `all`. Use a direct stage YAML when desired:

```bash
python run.py --step 1 --config configs/1.Committorvector.yaml
python run.py --step 1 --substep rate_constant
python run.py --step 2 --substep merging
python run.py --step 3 --substep gini
```

The next-hit notation is `q_j(x)` for destination-state probabilities. Rate
estimation is named `rate_constant` (not `rate_contant`). Gradpath's native
runner performs gradient shooting and weighted path clustering in one call;
the dispatcher reports the clustering substep as complete after pathfinding.
Voronoi merging is controlled by `VORONOI_MERGE.enabled` and is disabled in the
example until current and previous sample files are supplied.

Relabel diagnostics use normalized committor entropy `H`; `relabel_gini` uses
normalized Gini impurity `G = 1 - sum_j q_j^2` through the existing
`relabel_test_G.py` implementation. Both methods use the same dataset, model,
lag-safe kinetic checks, and threshold sections, while writing to separate
output directories.

## Commands

## For generating datasets:
```bash
python scripts/label.py --config configs/example.yaml
```
## For training:
```bash
python scripts/next_hit_train.py --config configs/example.yaml
```
## Plotting and building kinetic models:
```bash
python scripts/next_hit_infer.py --config configs/example.yaml
python scripts/next_hit_plot.py --config configs/example.yaml
python scripts/next_hit_rate.py --config configs/example.yaml
```

## Path finding from committor vector gradient
```bash
python scripts/gradpath.py --config configs/gradpath.example.yaml
python scripts/gradpath_plot.py --config configs/gradpath.example.yaml
python scripts/gradpath.mergy.py --config configs/voronoi_merge.example.yaml # Merging based on Voronoi expansion
```
## Old pairwise committor related
```bash
python scripts/pairwise_train.py --config configs/example.yaml
python scripts/pairwise_infer.py --config configs/example.yaml
python scripts/pairwise_plot.py --config configs/example.yaml
python scripts/pairwise_rate.py --config configs/example.yaml
```

## MSM/PCCA+ core labels
```bash
PYTHONPATH=src python -m tensorq.MSMlabel.cli all src/tensorq/MSMlabel/config.template.yaml
```

For inference, plotting, and rate estimation, prefer `*_checkpoint.pt` models because they preserve the model input metadata.
