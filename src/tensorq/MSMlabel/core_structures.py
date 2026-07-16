from __future__ import annotations

import glob
import os
import re
import struct
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ensure_dir, stage_path


def find_matching(root: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))


def trajectory_stem(filename: str) -> str:
    for suffix in (".colvars.traj", ".dcd"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return os.path.splitext(filename)[0]


def pairing_tag(stem: str, cre: re.Pattern) -> Optional[str]:
    matches = list(cre.finditer(stem))
    if not matches:
        return None
    match = matches[-1]
    return match.group(1) if match.lastindex else match.group(0)


def collect_pairing_candidates(
    files: List[str],
    root: str,
    tag_re: str,
) -> Tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], List[Tuple[Tuple[str, str], str]]]]:
    by_stem: Dict[Tuple[str, str], str] = {}
    by_tag: Dict[Tuple[str, str], List[Tuple[Tuple[str, str], str]]] = {}
    cre = re.compile(tag_re)
    for path in files:
        rel = os.path.relpath(path, root)
        subdir = os.path.dirname(rel)
        stem = trajectory_stem(os.path.basename(path))
        stem_key = (subdir, stem)
        by_stem[stem_key] = path
        tag = pairing_tag(stem, cre)
        if tag is not None:
            by_tag.setdefault((subdir, tag), []).append((stem_key, path))
    return by_stem, by_tag


def find_pairs_dcd_colvars(
    roots: List[str],
    match_dcd: str,
    match_colvars: str,
    tag_re: str = r"([ABM])",
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for root in roots:
        dcds = find_matching(root, f"*{match_dcd}*.dcd")
        cols = find_matching(root, f"*{match_colvars}*.colvars.traj")
        if not dcds:
            print(f"[warn] No DCD files under {root} matching '*{match_dcd}*.dcd'")
            continue
        if not cols:
            print(f"[warn] No colvars files under {root} matching '*{match_colvars}*.colvars.traj'")
            continue

        d_by_stem, d_by_tag = collect_pairing_candidates(dcds, root, tag_re)
        c_by_stem, c_by_tag = collect_pairing_candidates(cols, root, tag_re)

        common_stems = sorted(set(d_by_stem) & set(c_by_stem))
        local_pairs = [(d_by_stem[key], c_by_stem[key]) for key in common_stems]
        used_d = set(common_stems)
        used_c = set(common_stems)

        for tag_key in sorted(set(d_by_tag) & set(c_by_tag)):
            d_items = [(key, path) for key, path in d_by_tag[tag_key] if key not in used_d]
            c_items = [(key, path) for key, path in c_by_tag[tag_key] if key not in used_c]
            if not d_items or not c_items:
                continue
            if len(d_items) == 1 and len(c_items) == 1:
                d_key, d_path = d_items[0]
                c_key, c_path = c_items[0]
                local_pairs.append((d_path, c_path))
                used_d.add(d_key)
                used_c.add(c_key)
            else:
                print(
                    f"[warn] Ambiguous tag-only pairing for {tag_key} under {root}: "
                    f"{len(d_items)} DCD and {len(c_items)} colvars files remain."
                )

        pairs.extend(local_pairs)

    if not pairs:
        raise FileNotFoundError("No matching (dcd, colvars) pairs found for core structure export.")
    return sorted(pairs)


def read_colvars_traj(path: str) -> pd.DataFrame:
    colnames = None
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                tokens = stripped.lstrip("#").strip().split()
                if len(tokens) >= 2 and all("=" not in token for token in tokens):
                    colnames = tokens
            else:
                break
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if colnames is not None and len(colnames) == df.shape[1]:
        df.columns = colnames
    else:
        df.columns = [f"col{i}" for i in range(df.shape[1])]
    return df


def load_core_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".pt", ".pth"}:
        try:
            import torch
        except Exception as exc:
            raise SystemExit("Reading core_structures.dataset_path='.pt' requires torch.") from exc
        try:
            pack = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            pack = torch.load(path, map_location="cpu")
        cv = pack["cv"]
        meta_state = pack["meta_state"]
        dist = pack["dist_to_centroid"]
        if hasattr(cv, "detach"):
            cv = cv.detach().cpu().numpy()
        if hasattr(meta_state, "detach"):
            meta_state = meta_state.detach().cpu().numpy()
        if hasattr(dist, "detach"):
            dist = dist.detach().cpu().numpy()
        meta = pack.get("meta", {})
        cv_headers = list(meta.get("cv_headers", [])) if isinstance(meta, dict) else []
    elif suffix == ".npz":
        pack = np.load(path, allow_pickle=True)
        cv = np.asarray(pack["cv"])
        meta_state = np.asarray(pack["meta_state"])
        dist = np.asarray(pack["dist_to_centroid"])
        cv_headers = []
        if "meta_yaml" in pack:
            try:
                import yaml

                meta = yaml.safe_load(str(pack["meta_yaml"][0])) or {}
                cv_headers = list(meta.get("cv_headers", []))
            except Exception:
                cv_headers = []
    else:
        raise SystemExit("core_structures.dataset_path must point to dataset.pt or dataset.npz.")

    cv = np.asarray(cv, dtype=np.float64)
    if cv.ndim == 1:
        cv = cv.reshape(-1, 1)
    meta_state = np.asarray(meta_state, dtype=np.int64).reshape(-1)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    if cv.shape[0] != meta_state.shape[0]:
        raise SystemExit(f"Dataset cv rows ({cv.shape[0]}) != meta_state rows ({meta_state.shape[0]}).")
    if dist.shape[0] != cv.shape[0]:
        raise SystemExit(f"Dataset dist_to_centroid rows ({dist.shape[0]}) != cv rows ({cv.shape[0]}).")
    if len(cv_headers) != cv.shape[1]:
        cv_headers = [f"cv_{idx}" for idx in range(cv.shape[1])]
    return cv, meta_state, [str(name) for name in cv_headers], dist


def quantized_key(values: Iterable[float], tolerance: float) -> Tuple[int, ...]:
    vals = np.asarray(list(values), dtype=np.float64)
    return tuple(np.rint(vals / tolerance).astype(np.int64).tolist())


def build_target_index(
    cv: np.ndarray,
    meta_state: np.ndarray,
    cv_headers: List[str],
    match_cvs: List[str],
    tolerance: float,
) -> Dict[Tuple[int, ...], deque]:
    idx_by_name = {name: idx for idx, name in enumerate(cv_headers)}
    missing = [name for name in match_cvs if name not in idx_by_name]
    if missing:
        raise SystemExit(f"core_structures.match_cvs missing from dataset cv headers: {missing}")
    cols = [idx_by_name[name] for name in match_cvs]
    targets: Dict[Tuple[int, ...], deque] = defaultdict(deque)
    for dataset_row in np.where(meta_state >= 0)[0]:
        key = quantized_key(cv[dataset_row, cols], tolerance)
        targets[key].append(
            {
                "dataset_row": int(dataset_row),
                "core_state": int(meta_state[dataset_row]),
                "cv": cv[dataset_row, cols].astype(np.float64),
            }
        )
    return targets


def match_colvars_rows(
    colvars_path: str,
    match_cvs: List[str],
    targets: Dict[Tuple[int, ...], deque],
    tolerance: float,
    stride: int,
    skip_first_colvars: bool,
    colvars_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    if colvars_df is not None:
        df = colvars_df
    else:
        df = read_colvars_traj(colvars_path)
        if stride != 1:
            df = df.iloc[::stride].reset_index(drop=True)
    missing = [name for name in match_cvs if name not in df.columns]
    if missing:
        raise SystemExit(f"core_structures.match_cvs missing from {colvars_path}: {missing}")

    values = df[match_cvs].to_numpy(dtype=np.float64)
    row_offset = 1 if skip_first_colvars else 0
    matched: List[Dict[str, Any]] = []
    used_dataset_rows = set()

    for row_idx, row_values in enumerate(values):
        key = quantized_key(row_values, tolerance)
        queue = targets.get(key)
        if not queue:
            continue
        best_pos = None
        best_dist = np.inf
        for pos, target in enumerate(queue):
            if target["dataset_row"] in used_dataset_rows:
                continue
            dist = float(np.max(np.abs(row_values - target["cv"])))
            if dist <= tolerance and dist < best_dist:
                best_pos = pos
                best_dist = dist
        if best_pos is None:
            continue

        target = queue[best_pos]
        del queue[best_pos]
        used_dataset_rows.add(int(target["dataset_row"]))
        matched.append(
            {
                "dataset_row": int(target["dataset_row"]),
                "core_state": int(target["core_state"]),
                "colvars_row": int(row_idx),
                "dcd_frame": int((row_idx - row_offset) * stride),
                "max_abs_cv_delta": best_dist,
            }
        )
    return [item for item in matched if item["dcd_frame"] >= 0]


def _preload_pairs_data(
    pairs: List[Tuple[str, str]],
    stride: int,
    md: Any,
    top_path: str,
    allow_skip_first: bool,
) -> List[Dict[str, Any]]:
    """Pre-load colvars DataFrames and count DCD frames for all pairs in parallel."""
    t0 = time.perf_counter()

    def _load_one(idx: int, dcd_path: str, colvars_path: str) -> Optional[Dict[str, Any]]:
        n_dcd = strided_length(trajectory_n_frames(md, dcd_path, top_path), stride)
        n_colvars = strided_length(count_colvars_rows(colvars_path), stride)
        if n_colvars == n_dcd:
            skip_first = False
        elif allow_skip_first and n_colvars == n_dcd + 1:
            skip_first = True
        else:
            print(
                f"[warn] skip mismatched pair after stride={stride}: "
                f"dcd={n_dcd} colvars={n_colvars} {dcd_path}"
            )
            return None
        df = read_colvars_traj(colvars_path)
        if stride != 1:
            df = df.iloc[::stride].reset_index(drop=True)
        return {
            "idx": idx,
            "dcd": dcd_path,
            "colvars": colvars_path,
            "df": df,
            "skip_first": skip_first,
        }

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor() as ex:
        futures = {ex.submit(_load_one, i, d, c): i for i, (d, c) in enumerate(pairs)}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                results.append(result)

    results.sort(key=lambda r: r["idx"])
    elapsed = time.perf_counter() - t0
    n_skipped = len(pairs) - len(results)
    msg = f"[time] preloaded {len(results)}/{len(pairs)} colvars+DCD pairs in {elapsed:.1f}s"
    if n_skipped:
        msg += f" (skipped {n_skipped} mismatched)"
    print(msg, flush=True)
    return results


def match_targets_to_pairs(
    pairs: List[Tuple[str, str]],
    match_cvs: List[str],
    targets: Dict[Tuple[int, ...], deque],
    tolerance: float,
    stride: int,
    allow_skip_first_colvars: bool,
    md: Any,
    top_path: str,
    preloaded: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if preloaded is not None:
        for pdata in preloaded:
            pair_index = pdata["idx"]
            dcd_path = pdata["dcd"]
            colvars_path = pdata["colvars"]
            pair_matches = match_colvars_rows(
                colvars_path,
                match_cvs,
                targets,
                tolerance,
                stride,
                pdata["skip_first"],
                colvars_df=pdata["df"],
            )
            for item in pair_matches:
                item["pair_index"] = int(pair_index)
                item["dcd"] = dcd_path
                item["colvars"] = colvars_path
                rows.append(item)
            remaining = sum(len(queue) for queue in targets.values())
            print(f"[info] matched {len(pair_matches)} frames in {colvars_path}; remaining targets={remaining}", flush=True)
            if remaining == 0:
                break
    else:
        for pair_index, (dcd_path, colvars_path) in enumerate(pairs):
            n_dcd = strided_length(trajectory_n_frames(md, dcd_path, top_path), stride)
            n_colvars = strided_length(count_colvars_rows(colvars_path), stride)
            if n_colvars == n_dcd:
                skip_first = False
            elif allow_skip_first_colvars and n_colvars == n_dcd + 1:
                skip_first = True
            else:
                print(
                    f"[warn] skip mismatched pair after stride={stride}: "
                    f"dcd={n_dcd} colvars={n_colvars} {dcd_path}"
                )
                continue
            pair_matches = match_colvars_rows(
                colvars_path,
                match_cvs,
                targets,
                tolerance,
                stride,
                skip_first,
            )
            for item in pair_matches:
                item["pair_index"] = int(pair_index)
                item["dcd"] = dcd_path
                item["colvars"] = colvars_path
                rows.append(item)
            remaining = sum(len(queue) for queue in targets.values())
            print(f"[info] matched {len(pair_matches)} frames in {colvars_path}; remaining targets={remaining}", flush=True)
            if remaining == 0:
                break
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["core_state", "pair_index", "dcd_frame", "dataset_row"]).reset_index(drop=True)


def count_colvars_rows(path: str) -> int:
    n_rows = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                n_rows += 1
    return n_rows


def strided_length(n_frames: int, stride: int) -> int:
    if n_frames <= 0:
        return 0
    return (int(n_frames) + int(stride) - 1) // int(stride)


def trajectory_n_frames(md: Any, dcd_path: str, top_path: str) -> int:
    with open(dcd_path, "rb") as handle:
        header = handle.read(12)
    if len(header) == 12 and header[4:8] in (b"CORD", b"VELD"):
        n_frames = struct.unpack("<i", header[8:12])[0]
        if n_frames > 0:
            return int(n_frames)
        n_frames = struct.unpack(">i", header[8:12])[0]
        if n_frames > 0:
            return int(n_frames)

    try:
        with md.open(dcd_path) as handle:
            try:
                return int(len(handle))
            except Exception:
                if hasattr(handle, "n_frames"):
                    return int(handle.n_frames)
    except Exception:
        pass
    traj = md.load(dcd_path, top=top_path)
    return int(traj.n_frames)


def _element_from_atom_name(name: str) -> str:
    """Derive the element symbol from a PDB/CHARMM atom name.

    Strips leading digits (e.g. "1H" → "H").  For C/N/O the *second*
    character is checked against the known protein-atom suffix set so that
    e.g. "CD" → C (delta‑carbon) while "CL" → Cl (chlorine ion) and
    "NA" → Na (sodium ion).  H, S, and P have no ambiguous two‑letter
    counterparts in biomolecular simulations and always map to the first
    character.  Everything else falls through to a two‑letter element
    lookup that handles cofactors and ions (Fe, Zn, Mg, …).
    """
    stripped = name.lstrip("0123456789 ")
    if not stripped:
        return "C"

    first = stripped[0].upper()
    second = stripped[1].upper() if len(stripped) >= 2 else ""

    # Protein carbon suffixes: CA, CB, CG, CD, CE, CZ, CH, CH2
    if first == "C" and (not second or second in "ABGDEZH"):
        return "C"
    # Protein nitrogen suffixes: ND1, NE1, NE2, NZ, NH1, NH2
    if first == "N" and (not second or second in "DEZH"):
        return "N"
    # Protein oxygen suffixes: OD1, OD2, OE1, OE2, OG, OG1, OH, OXT
    if first == "O" and (not second or second in "DEGHX"):
        return "O"
    # H, S, P have no two‑letter element conflicts in biomolecular contexts
    if first in "HSP":
        return first

    two_letter = {
        "FE", "ZN", "MG", "MN", "CL", "NA", "BR", "CU", "CO",
        "NI", "PT", "AU", "AG", "HG", "SE", "CR", "MO",
        "SR", "BA", "LI", "BE", "AL", "SI", "KR", "XE",
        "CA",
    }
    if len(stripped) >= 2 and stripped[:2].upper() in two_letter:
        return stripped[:2].upper()
    return first


def _build_pdb_atom_table(topology) -> List[Dict[str, Any]]:
    """Extract per-atom metadata from an mdtraj topology for strict PDB formatting."""
    atoms: List[Dict[str, Any]] = []
    for atom in topology.atoms:
        name = str(atom.name)
        chain_idx = atom.residue.chain.index if atom.residue.chain is not None else 0
        atoms.append(
            {
                "name": name,
                "res_name": str(atom.residue.name)[:3],
                "chain_id": chain_idx,
                "res_seq": int(atom.residue.resSeq),
                "element": _element_from_atom_name(name),
            }
        )
    return atoms


def _format_pdb_atom_line(
    serial: int,
    info: Dict[str, Any],
    x: float,
    y: float,
    z: float,
    occupancy: float = 1.0,
    temp_factor: float = 0.0,
) -> str:
    """Return an 80-column PDB ATOM record with strict column alignment.

    Columns  (official PDB v3.3 / wwPDB):
      1- 6   "ATOM  " (or "HETATM")
      7-11   atom serial number            right-justified
     12       space
     13-16   atom name                     see PDB convention below
     17       alternate location indicator  (space if none)
     18-20   residue name                  right-justified
     21       space
     22       chain identifier
     23-26   residue sequence number       right-justified
     27       insertion code               (space if none)
     28-30   space
     31-38   x coordinate (8.3f)           right-justified
     39-46   y coordinate (8.3f)           right-justified
     47-54   z coordinate (8.3f)           right-justified
     55-60   occupancy    (6.2f)           right-justified
     61-66   temp factor  (6.2f)           right-justified
     67-76   space
     77-78   element symbol                right-justified
     79-80   charge                        (space if none)
    """
    name = info["name"]
    elem = info["element"]

    # Atom name placement (columns 13-16).
    # Single-letter elements: name starts at column 14.
    # Two-letter elements:   name starts at column 13.
    if len(elem) == 1:
        if len(name) == 1:
            pdb_name = f" {name}  "
        elif len(name) == 2:
            pdb_name = f" {name} "
        elif len(name) == 3:
            pdb_name = f" {name}"
        else:
            pdb_name = name[:4]
    else:
        if len(name) <= 2:
            pdb_name = f"{name:<4}"
        elif len(name) == 3:
            pdb_name = f"{name} "
        else:
            pdb_name = name[:4]

    chain_id = chr(65 + info["chain_id"]) if 0 <= info["chain_id"] <= 25 else " "

    return (
        f"ATOM  "
        f"{serial:>5} "
        f"{pdb_name}"
        f" "
        f"{info['res_name'][:3]:>3}"
        f" "
        f"{chain_id}"
        f"{info['res_seq']:>4} "
        f"   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{temp_factor:6.2f}"
        f"          "
        f"{elem:>2}"
        f"  \n"
    )


def _write_pdb_cryst1(handle, lengths, angles) -> None:
    """Write CRYST1 record if unit-cell parameters are available.

    mdtraj stores lengths in nm internally; PDB expects Angstroms.
    """
    if lengths is None or angles is None:
        return
    a, b, c = float(lengths[0]) * 10.0, float(lengths[1]) * 10.0, float(lengths[2]) * 10.0
    alpha, beta, gamma = float(angles[0]), float(angles[1]), float(angles[2])
    handle.write(
        f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1\n"
    )


def save_matched_pdbs(
    *,
    frame_table: pd.DataFrame,
    out_dir: str,
    top_path: str,
    chunk_size: int,
    atomselect: Optional[str],
) -> pd.DataFrame:
    try:
        import mdtraj as md
    except Exception as exc:
        raise SystemExit("Writing core structures requires mdtraj.") from exc

    atom_indices = None
    topology = md.load_topology(top_path)
    if atomselect:
        atom_indices = topology.select(str(atomselect))
        if atom_indices.size == 0:
            raise SystemExit(f"core_structures.atomselect selected zero atoms: {atomselect}")
        topology = topology.subset(atom_indices)

    atom_table = _build_pdb_atom_table(topology)

    summary_rows: List[Dict[str, Any]] = []
    for dcd_path, dcd_group in frame_table.groupby("dcd", sort=False):
        dcd_group = dcd_group.sort_values("dcd_frame")
        needed = np.asarray(sorted(dcd_group["dcd_frame"].unique().tolist()), dtype=np.int64)
        by_frame = {int(frame): block.copy() for frame, block in dcd_group.groupby("dcd_frame", sort=False)}
        last_needed = int(needed[-1])
        cursor = 0
        for chunk in md.iterload(dcd_path, chunk=chunk_size, top=top_path, atom_indices=atom_indices):
            start = cursor
            stop = start + int(chunk.n_frames)
            cursor = stop
            hits = needed[(needed >= start) & (needed < stop)]
            for dcd_frame in hits.tolist():
                xyz = chunk.xyz[int(dcd_frame) - start]  # (n_atoms, 3)
                uc_len = chunk.unitcell_lengths[int(dcd_frame) - start] if chunk.unitcell_lengths is not None else None
                uc_ang = chunk.unitcell_angles[int(dcd_frame) - start] if chunk.unitcell_angles is not None else None
                for _, row in by_frame[int(dcd_frame)].iterrows():
                    state = int(row["core_state"])
                    pdb_path = os.path.join(out_dir, f"core_state_{state:03d}.pdb")
                    with open(pdb_path, "w") as fh:
                        _write_pdb_cryst1(fh, uc_len, uc_ang)
                        fh.write(f"MODEL        1\n")
                        for i, info in enumerate(atom_table):
                            fh.write(
                                _format_pdb_atom_line(
                                    serial=i + 1,
                                    info=info,
                                    x=float(xyz[i, 0]) * 10.0,
                                    y=float(xyz[i, 1]) * 10.0,
                                    z=float(xyz[i, 2]) * 10.0,
                                )
                            )
                        fh.write("TER\n")
                        fh.write("ENDMDL\n")
                        fh.write("END\n")
                    summary_rows.append(
                        {
                            "core_state": state,
                            "dataset_row": int(row["dataset_row"]),
                            "pdb": pdb_path,
                            "model_index": 1,
                            "dcd": dcd_path,
                            "dcd_frame": int(dcd_frame),
                            "max_abs_cv_delta": float(row["max_abs_cv_delta"]),
                        }
                    )
            if cursor > last_needed:
                break
    return pd.DataFrame(summary_rows)


def export_core_structures_from_dataset(cfg: Dict[str, Any]) -> str:
    struct_cfg = cfg.get("core_structures", {}) or {}
    if not bool(struct_cfg.get("enabled", True)):
        print("[skip] core_structures.enabled=false")
        return ""

    dataset_path = struct_cfg.get("dataset_path", None)
    if dataset_path is None:
        lag = int(struct_cfg.get("lag", cfg.get("pcca", {}).get("selected_lag", 0)))
        m = int(struct_cfg.get("m", cfg.get("pcca", {}).get("single_m", cfg.get("pcca", {}).get("only_m", 0))))
        save_format = str(cfg.get("core_labeling", {}).get("save_format", "npz")).lower()
        filename = "dataset.pt" if save_format == "pt" else f"dataset.{save_format}"
        dataset_path = stage_path(cfg, "06_core_labels", f"lag_{lag}", f"m_{m}", filename)
    dataset_path = str(dataset_path)
    if not os.path.exists(dataset_path):
        raise SystemExit(f"core_structures.dataset_path not found: {dataset_path}")

    out_dir = str(struct_cfg.get("out_dir", stage_path(cfg, "07_core_structures")))
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(cfg["project"]["out_dir"], out_dir) if not out_dir.startswith(cfg["project"]["out_dir"]) else out_dir
    out_dir = ensure_dir(out_dir)

    top_path = str(struct_cfg.get("top", struct_cfg.get("topology", "")))
    if not top_path:
        raise SystemExit("core_structures.top must point to the topology used by the DCD files.")
    try:
        import mdtraj as md
    except Exception as exc:
        raise SystemExit("Writing core structures requires mdtraj.") from exc
    folders = [str(path) for path in struct_cfg.get("folders", [])]
    pairs = find_pairs_dcd_colvars(
        folders,
        str(struct_cfg.get("match_dcd", "")),
        str(struct_cfg.get("match_colvars", "")),
        str(struct_cfg.get("tag_regex", r"([ABM])")),
    )
    stride = int(struct_cfg.get("stride", 1))
    tolerance = float(struct_cfg.get("tolerance", struct_cfg.get("cv_tolerance", 1e-4)))
    if tolerance <= 0.0:
        raise SystemExit("core_structures.tolerance must be positive.")

    cv, meta_state, cv_headers, dist_to_centroid = load_core_dataset(dataset_path)
    match_cvs = [str(name) for name in struct_cfg.get("match_cvs", cv_headers)]
    if not match_cvs:
        raise SystemExit("core_structures.match_cvs could not be inferred; set it explicitly.")
    targets = build_target_index(cv, meta_state, cv_headers, match_cvs, tolerance)
    n_targets = sum(len(queue) for queue in targets.values())
    if n_targets == 0:
        raise SystemExit("core_structures found no dataset rows with meta_state >= 0.")
    print(f"[info] matching {n_targets} core frames from {dataset_path}", flush=True)

    preloaded = _preload_pairs_data(
        pairs,
        stride,
        md,
        top_path,
        bool(struct_cfg.get("allow_skip_first_colvars", True)),
    )
    if not preloaded:
        raise SystemExit("core_structures: no valid (dcd, colvars) pairs after length alignment.")

    frame_table = match_targets_to_pairs(
        pairs,
        match_cvs,
        targets,
        tolerance,
        stride,
        bool(struct_cfg.get("allow_skip_first_colvars", True)),
        md,
        top_path,
        preloaded=preloaded,
    )
    if frame_table.empty:
        raise SystemExit("core_structures did not match any dataset core frames to colvars rows.")

    # Keep only the core-center frame (minimum dist_to_centroid) per core_state.
    frame_table["dist_to_centroid"] = dist_to_centroid[frame_table["dataset_row"].to_numpy(dtype=np.int64)]
    center_mask = frame_table.groupby("core_state")["dist_to_centroid"].idxmin()
    frame_table = frame_table.loc[center_mask].reset_index(drop=True)
    print(f"[info] selected {len(frame_table)} core-center frames (min dist_to_centroid per state)", flush=True)

    frames_csv = os.path.join(out_dir, "frames.csv")
    frame_table.to_csv(frames_csv, index=False)

    written = save_matched_pdbs(
        frame_table=frame_table,
        out_dir=out_dir,
        top_path=top_path,
        chunk_size=int(struct_cfg.get("chunk_size", 256)),
        atomselect=struct_cfg.get("atomselect", struct_cfg.get("atom_selection", None)),
    )
    summary_csv = os.path.join(out_dir, "summary.csv")
    written.to_csv(summary_csv, index=False)

    counts = written.groupby("core_state").size().reset_index(name="n_pdbs")
    counts.to_csv(os.path.join(out_dir, "counts.csv"), index=False)
    remaining = sum(len(queue) for queue in targets.values())
    print(f"[ok] wrote {len(written)} PDBs to {out_dir}; unmatched targets={remaining}", flush=True)
    return out_dir
