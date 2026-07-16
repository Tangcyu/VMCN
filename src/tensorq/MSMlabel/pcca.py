from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.linalg import eig
try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None
import pandas as pd

from .checkpoint import exists, load_npz, save_npz
from .config import ensure_dir, stage_path
from .msm import count_matrix, trajectory_label_weight_arrays, transition_matrix


def spectral_memberships(T: np.ndarray, m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if KMeans is None:
        raise SystemExit("Need scikit-learn for the spectral PCCA fallback. Install with: pip install scikit-learn")
    vals, vecs = eig(T, left=False, right=True)
    order = np.argsort(np.real(vals))[::-1]
    Y = np.real(vecs[:, order[:m]])
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    macro = KMeans(n_clusters=m, n_init=20, random_state=seed).fit_predict(Y)
    memberships = np.zeros((T.shape[0], m), dtype=np.float64)
    memberships[np.arange(T.shape[0]), macro] = 1.0
    return memberships, macro.astype(np.int64)


def try_deeptime_pcca(T: np.ndarray, m: int):
    try:
        from deeptime.markov.msm import MarkovStateModel
        from deeptime.markov.tools.analysis import pcca_memberships
    except Exception:
        return None
    msm = MarkovStateModel(T)
    memberships = np.asarray(pcca_memberships(msm.transition_matrix, m), dtype=np.float64)
    macro = np.argmax(memberships, axis=1).astype(np.int64)
    return memberships, macro


def macro_transition(T: np.ndarray, pi: np.ndarray, memberships: np.ndarray) -> np.ndarray:
    chi = memberships
    weights = pi[:, None] * chi
    denom = np.maximum(weights.sum(axis=0), 1e-300)
    return (weights.T @ T @ chi) / denom[:, None]


def residence_escape_times(P: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    stay = np.asarray(np.diag(P), dtype=np.float64)
    residence = np.zeros_like(stay, dtype=np.float64)
    escape = np.zeros_like(stay, dtype=np.float64)

    middle = (stay > 0.0) & (stay < 1.0)
    residence[middle] = -float(lag) / np.log(stay[middle])
    escape[middle] = float(lag) / (1.0 - stay[middle])

    escaped = stay <= 0.0
    residence[escaped] = 0.0
    escape[escaped] = float(lag)

    trapped = stay >= 1.0
    residence[trapped] = np.inf
    escape[trapped] = np.inf
    return residence, escape


def active_microstates_from_weights(table: pd.DataFrame, labels: np.ndarray, weight_column: str) -> np.ndarray:
    if weight_column in table.columns:
        weights = pd.to_numeric(table[weight_column], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    else:
        weights = np.ones(len(table), dtype=np.float64)
    active = np.unique(labels[weights > 0.0])
    return active.astype(np.int64)


def restrict_msm_to_active_microstates(C_full: np.ndarray, active_microstates: np.ndarray, pseudocount: float) -> tuple[np.ndarray, np.ndarray]:
    C_active = C_full[np.ix_(active_microstates, active_microstates)]
    T_active = transition_matrix(C_active, reversible=False, pseudocount=pseudocount)
    pi_active = np.sum(C_active, axis=1)
    if float(pi_active.sum()) <= 0.0:
        pi_active = stationary_distribution_from_transition(T_active)
    else:
        pi_active = pi_active / float(pi_active.sum())
    return T_active, pi_active


def stationary_distribution_from_transition(T: np.ndarray) -> np.ndarray:
    vals, vecs = eig(T.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.abs(np.real(vecs[:, idx]))
    return pi / max(float(pi.sum()), 1e-300)


def ck_test(
    seqs: List[np.ndarray],
    macro_by_micro: np.ndarray,
    m: int,
    lag: int,
    multiples: List[int],
    pseudocount: float,
    weight_seqs: List[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    macro_seqs = [macro_by_micro[seq] for seq in seqs]
    C1 = count_matrix(macro_seqs, lag=lag, n_states=m, weight_seqs=weight_seqs)
    P1 = transition_matrix(C1, reversible=False, pseudocount=pseudocount)
    direct = []
    predicted = []
    rmsd = []
    for k in multiples:
        Ck = count_matrix(macro_seqs, lag=lag * int(k), n_states=m, weight_seqs=weight_seqs)
        Pk = transition_matrix(Ck, reversible=False, pseudocount=pseudocount)
        Ppred = np.linalg.matrix_power(P1, int(k))
        direct.append(Pk)
        predicted.append(Ppred)
        rmsd.append(float(np.sqrt(np.mean((Pk - Ppred) ** 2))))
    return np.asarray(direct), np.asarray(predicted), np.asarray(rmsd)


def configured_m_values(cfg: Dict) -> List[int]:
    pcca_cfg = cfg["pcca"]
    single_m = pcca_cfg.get("single_m", pcca_cfg.get("only_m", None))
    if single_m is not None:
        return [int(single_m)]
    m_values = pcca_cfg.get("m_values", [])
    if isinstance(m_values, int):
        return [int(m_values)]
    return [int(x) for x in m_values]


def analyze_macrostates(cfg: Dict, table: pd.DataFrame, micro: Dict[str, np.ndarray], msms: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
    force = bool(cfg["project"].get("force", False))
    lag = int(cfg["pcca"].get("selected_lag", cfg["msm"]["lags"][0]))
    if lag not in msms:
        raise SystemExit(f"pcca.selected_lag={lag} was not built. Add it to msm.lags.")
    msm = msms[lag]
    T = np.asarray(msm["transition_matrix"], dtype=np.float64)
    pi = np.asarray(msm["stationary_distribution"], dtype=np.float64)
    C = np.asarray(msm["count_matrix"], dtype=np.float64)
    labels = np.asarray(micro["labels"], dtype=np.int64)
    use_weights = bool(cfg["msm"].get("use_weights", True))
    mask_zero_weight_origins = bool(cfg["msm"].get("mask_zero_weight_origins", True))
    weight_column = cfg["data"].get("weight_column", "weight")
    seqs, weight_seqs = trajectory_label_weight_arrays(
        table,
        labels,
        weight_column=weight_column,
        use_weights=use_weights,
        mask_zero_weight_origins=mask_zero_weight_origins,
    )
    out: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {lag: {}}
    ensure_dir(stage_path(cfg, "04_pcca"))
    exclude_zero_weight_microstates = bool(cfg["pcca"].get("exclude_zero_weight_microstates", True))
    active_microstates = active_microstates_from_weights(table, labels, weight_column) if exclude_zero_weight_microstates else np.arange(T.shape[0])

    for m in configured_m_values(cfg):
        out_npz = stage_path(cfg, "04_pcca", f"lag_{lag}", f"m_{m}", "pcca.npz")
        if exists(out_npz, force=force):
            print(f"[reuse] PCCA m={m}: {out_npz}")
            out[lag][m] = load_npz(out_npz)
            continue

        if active_microstates.size < m:
            raise SystemExit(
                f"PCCA m={m} needs at least {m} active microstates, but only "
                f"{active_microstates.size} have positive-weight support."
            )
        if exclude_zero_weight_microstates:
            T_pcca, pi_pcca = restrict_msm_to_active_microstates(C, active_microstates, float(cfg["msm"].get("pseudocount", 1e-8)))
        else:
            T_pcca, pi_pcca = T, pi

        pcca_result = try_deeptime_pcca(T_pcca, m) if bool(cfg["pcca"].get("use_deeptime", True)) else None
        if pcca_result is None:
            memberships_active, macro_active = spectral_memberships(T_pcca, m=m, seed=int(cfg["project"].get("seed", 2026)))
            method = "spectral_kmeans_fallback"
        else:
            memberships_active, macro_active = pcca_result
            method = "deeptime_pcca_plus"

        memberships = np.zeros((T.shape[0], m), dtype=np.float64)
        macro_by_micro = np.full(T.shape[0], -1, dtype=np.int64)
        memberships[active_microstates, :] = memberships_active
        macro_by_micro[active_microstates] = macro_active

        Pmacro = macro_transition(T_pcca, pi_pcca, memberships_active)
        residence, escape = residence_escape_times(Pmacro, lag=lag)
        multiples = [int(x) for x in cfg["ck_test"].get("multiples", [1, 2, 3, 4])]
        direct, predicted, rmsd = ck_test(
            seqs,
            macro_by_micro=macro_by_micro,
            m=m,
            lag=lag,
            multiples=multiples,
            pseudocount=float(cfg["msm"].get("pseudocount", 1e-8)),
            weight_seqs=weight_seqs,
        )
        save_npz(
            out_npz,
            memberships=memberships,
            macro_by_micro=macro_by_micro,
            macro_transition=Pmacro,
            residence_times=residence,
            escape_times=escape,
            ck_direct=direct,
            ck_predicted=predicted,
            ck_rmsd=rmsd,
            ck_multiples=np.asarray(multiples, dtype=np.int64),
            active_microstates=active_microstates,
            manifest={
                "stage": "pcca",
                "lag": lag,
                "m": m,
                "method": method,
                "exclude_zero_weight_microstates": exclude_zero_weight_microstates,
                "n_active_microstates": int(active_microstates.size),
                "ck_uses_weights": use_weights,
                "ck_masks_zero_weight_origins": mask_zero_weight_origins,
                "weight_column": weight_column if use_weights else None,
            },
        )
        print(f"[ok] PCCA m={m}: {out_npz} ({method})")
        out[lag][m] = load_npz(out_npz)
    return out
