# TensorQ

TensorQ now uses one shared package architecture:

- `tensorq.common`: YAML, dataset, CV featurization, lagged indexing, and rate helpers.
- `tensorq.next_hit`: next-hit committor training, inference, plotting, and rate estimation.
- `tensorq.pairwise`: pair-wise committor training, inference, plotting, and rate estimation.
- `tensorq.gradpath`: direct-channel gradient pathway shooting, weighted clustering, and plotting.
- `tensorq.voronoi_merge`: Voronoi shared-segment alignment and iterative KLD diagnostics.

The shared dataset is produced by `scripts/label.py` from the `TENSORQ_LABEL` config section. Both committor families consume that same `.pt` or `.npz` dataset.

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

For inference, plotting, and rate estimation, prefer `*_checkpoint.pt` models because they preserve the model input metadata.
