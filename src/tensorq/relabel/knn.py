from __future__ import annotations

import numpy as np

def _torch_knn_distances(query, reference, k, cfg, *, exclude_self=False):
    backend = str(cfg.get("knn_backend", "auto")).lower()
    if backend in {"sklearn", "cpu", "none"}:
        return None
    if backend not in {"auto", "torch", "cuda"}:
        raise ValueError("relabel.knn_backend must be one of: auto, torch, cuda, sklearn.")

    try:
        import torch
    except Exception:
        if backend == "auto":
            return None
        raise

    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    n_query = int(query.shape[0])
    n_reference = int(reference.shape[0])
    k = int(k)
    if n_query == 0 or n_reference == 0 or k <= 0:
        return (
            np.empty((n_query, 0), dtype=np.float32),
            np.empty((n_query, 0), dtype=np.int64),
            {"knn_backend": "torch", "knn_device": str(cfg.get("knn_device", "cuda:0"))},
        )

    max_pairs = int(cfg.get("torch_knn_auto_max_pairs", 1_000_000_000))
    if backend == "auto" and (n_query * n_reference) > max_pairs:
        return None

    device_str = str(cfg.get("knn_device", cfg.get("device", "cuda:0")))
    device = torch.device(device_str)
    if backend == "auto" and device.type != "cuda":
        return None
    if device.type == "cuda" and not torch.cuda.is_available():
        if backend == "auto":
            return None
        raise RuntimeError(f"Requested relabel.knn_device={device_str!r}, but CUDA is not available.")

    query_batch = max(1, int(cfg.get("torch_knn_query_batch", 4096)))
    reference_batch = max(1, int(cfg.get("torch_knn_reference_batch", 32768)))
    dtype_name = str(cfg.get("torch_knn_dtype", "float32")).lower()
    dtype = torch.float64 if dtype_name in {"float64", "double"} else torch.float32
    k = min(k, n_reference - (1 if exclude_self and n_query == n_reference else 0))
    if k <= 0:
        return (
            np.empty((n_query, 0), dtype=np.float32),
            np.empty((n_query, 0), dtype=np.int64),
            {"knn_backend": "torch", "knn_device": str(device)},
        )

    try:
        reference_t = torch.as_tensor(reference, dtype=dtype, device=device)
        distances_out = np.empty((n_query, k), dtype=np.float32)
        indices_out = np.empty((n_query, k), dtype=np.int64)
        with torch.no_grad():
            for start in range(0, n_query, query_batch):
                stop = min(start + query_batch, n_query)
                query_t = torch.as_tensor(query[start:stop], dtype=dtype, device=device)
                best_dist = torch.full((stop - start, k), float("inf"), dtype=dtype, device=device)
                best_idx = torch.full((stop - start, k), -1, dtype=torch.long, device=device)
                for ref_start in range(0, n_reference, reference_batch):
                    ref_stop = min(ref_start + reference_batch, n_reference)
                    dist = torch.cdist(query_t, reference_t[ref_start:ref_stop])
                    if exclude_self and n_query == n_reference:
                        overlap_start = max(start, ref_start)
                        overlap_stop = min(stop, ref_stop)
                        if overlap_start < overlap_stop:
                            local_q = torch.arange(
                                overlap_start - start,
                                overlap_stop - start,
                                device=device,
                            )
                            local_r = torch.arange(
                                overlap_start - ref_start,
                                overlap_stop - ref_start,
                                device=device,
                            )
                            dist[local_q, local_r] = float("inf")

                    local_k = min(k, dist.shape[1])
                    local_dist, local_idx = torch.topk(dist, k=local_k, dim=1, largest=False)
                    local_idx = local_idx + ref_start
                    merged_dist = torch.cat([best_dist, local_dist], dim=1)
                    merged_idx = torch.cat([best_idx, local_idx], dim=1)
                    best_dist, order = torch.topk(merged_dist, k=k, dim=1, largest=False)
                    best_idx = torch.gather(merged_idx, 1, order)

                distances_out[start:stop] = best_dist.detach().cpu().numpy().astype(np.float32, copy=False)
                indices_out[start:stop] = best_idx.detach().cpu().numpy().astype(np.int64, copy=False)
    except RuntimeError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if backend == "auto":
            return None
        raise

    return distances_out, indices_out, {
        "knn_backend": "torch",
        "knn_device": str(device),
        "torch_knn_query_batch": query_batch,
        "torch_knn_reference_batch": reference_batch,
        "torch_knn_auto_max_pairs": max_pairs,
    }

def _knn_component_labels(z, frame_idx, k, cfg):
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix

    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    if frame_idx.size == 0:
        return 0, np.zeros(0, dtype=np.int64), {"knn_backend": "none", "knn_device": ""}
    if frame_idx.size == 1:
        return 1, np.zeros(1, dtype=np.int64), {"knn_backend": "none", "knn_device": ""}

    graph_neighbors = min(max(1, int(k)), frame_idx.size - 1)
    points = z[frame_idx]
    torch_knn = _torch_knn_distances(points, points, graph_neighbors, cfg, exclude_self=True)
    if torch_knn is None:
        from sklearn.neighbors import kneighbors_graph

        graph = kneighbors_graph(
            points,
            n_neighbors=graph_neighbors,
            mode="connectivity",
            include_self=False,
        )
        knn_meta = {"knn_backend": "sklearn", "knn_device": ""}
    else:
        _, neighbor_idx, knn_meta = torch_knn
        row_idx = np.repeat(np.arange(frame_idx.size, dtype=np.int64), neighbor_idx.shape[1])
        col_idx = neighbor_idx.reshape(-1)
        data = np.ones(row_idx.size, dtype=np.int8)
        graph = csr_matrix((data, (row_idx, col_idx)), shape=(frame_idx.size, frame_idx.size))

    graph = graph.maximum(graph.T)
    n_components, component_labels = connected_components(graph, directed=False)
    return int(n_components), np.asarray(component_labels, dtype=np.int64), knn_meta

