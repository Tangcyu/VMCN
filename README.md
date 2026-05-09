# TensorQ

TensorQ now uses one shared package architecture:

- `tensorq.common`: YAML, dataset, CV featurization, lagged indexing, and flux helpers.
- `tensorq.next_hit`: next-hit committor training, inference, plotting, and rate estimation.
- `tensorq.pairwise`: pair-wise committor training, inference, plotting, and rate estimation.

The shared dataset is produced by `scripts/label.py` from the `TENSORQ_LABEL` config section. Both committor families consume that same `.pt` or `.npz` dataset.

## Commands

```bash
python scripts/label.py --config configs/example.yaml
python scripts/next_hit_train.py --config configs/example.yaml
python scripts/next_hit_infer.py --config configs/example.yaml
python scripts/next_hit_plot.py --config configs/example.yaml
python scripts/next_hit_rate.py --config configs/example.yaml
python scripts/pairwise_train.py --config configs/example.yaml
python scripts/pairwise_infer.py --config configs/example.yaml
python scripts/pairwise_plot.py --config configs/example.yaml
python scripts/pairwise_rate.py --config configs/example.yaml
```

For inference, plotting, and rate estimation, prefer `*_checkpoint.pt` models because they preserve the model input metadata.
