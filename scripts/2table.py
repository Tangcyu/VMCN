#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def fmt_sci(x, precision=4):
    """Format a number as LaTeX scientific notation: 1.234 \times 10^{-5}."""
    x = float(x)

    if math.isnan(x):
        return r"\mathrm{nan}"
    if math.isinf(x):
        return r"\infty" if x > 0 else r"-\infty"
    if x == 0:
        return "0"

    exponent = math.floor(math.log10(abs(x)))
    mantissa = x / (10**exponent)

    decimals = max(precision - 1, 0)
    mantissa_str = f"{mantissa:.{decimals}f}"

    if float(mantissa_str) >= 10:
        mantissa /= 10
        exponent += 1
        mantissa_str = f"{mantissa:.{decimals}f}"

    return rf"{mantissa_str} \times 10^{{{exponent}}}"


def fmt_sci_with_std(value, std, precision=4):
    r"""Format as (m \pm s) \times 10^{e} — shared exponent, no repetition."""
    value = float(value)
    std = float(std)

    if math.isnan(value):
        return r"\mathrm{nan}"
    if math.isinf(value):
        return r"\infty" if value > 0 else r"-\infty"
    if value == 0:
        return "0"
    if math.isnan(std) or math.isinf(std):
        return fmt_sci(value, precision)

    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10**exponent)
    std_mantissa = std / (10**exponent)

    decimals = max(precision - 1, 0)

    # If mantissa rounds to >= 10, shift exponent
    if float(f"{mantissa:.{decimals}f}") >= 10:
        mantissa /= 10
        std_mantissa /= 10
        exponent += 1

    return rf"({mantissa:.{decimals}f} \pm {std_mantissa:.{decimals}f}) \times 10^{{{exponent}}}"


def fmt_rate(value, std=None, precision=4):
    if std is None:
        return fmt_sci(value, precision)
    return fmt_sci_with_std(value, std, precision)


def _csv_to_matrix(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")

    first_col = df.columns[0]
    first_values = pd.to_numeric(df[first_col], errors="coerce")
    remaining = df.drop(columns=[first_col])

    # Matrix CSV with an index-label column: state_i,0,1,2,...
    if len(remaining.columns) == len(df) and first_values.notna().all():
        return remaining.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)

    return df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)


def read_matrix(path: Path) -> np.ndarray:
    """Read a square matrix from .csv or .npy."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path)
    elif suffix == ".csv":
        matrix = _csv_to_matrix(path)
    else:
        raise ValueError(f"Unsupported matrix file extension for {path}; use .csv or .npy.")

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{path} must contain a square matrix; got shape {matrix.shape}.")
    return matrix


def read_first_matrix(rate_dir: Path, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        p = rate_dir / name
        if p.exists():
            return read_matrix(p)
    return None


def read_rate_dir(rate_dir: Path) -> dict:
    """Read all rate outputs from a rate_out directory.

    Returns a dict with keys:
        k_mfpt      — (n, n) array of k_mfpt values
        k_mfpt_std  — (n, n) array of k_mfpt standard errors, if available
        k_direct    — (n, n) array of direct kij values, if available
        k_direct_std — (n, n) array of direct kij standard errors, if available
        p_jump      — (n, n) array of P(i -> j) jump probabilities
        pi          — (n,) array of equilibrium populations
        n_states    — int
        rate_unit   — str or None (e.g. "1/ps")
    """
    rate_dir = Path(rate_dir)

    data: dict = {"rate_unit": None}

    data["k_mfpt"] = read_first_matrix(rate_dir, ("k_mfpt.csv", "k_mfpt.npy", "Kij.csv", "Kij.npy"))
    if data["k_mfpt"] is None:
        raise FileNotFoundError(f"No k_mfpt matrix found in {rate_dir}")

    data["k_mfpt_std"] = read_first_matrix(rate_dir, ("k_mfpt_std.csv", "k_mfpt_std.npy", "Kij_std.csv", "Kij_std.npy"))
    data["k_direct"] = read_first_matrix(
        rate_dir,
        ("k_direct.csv", "k_direct.npy", "k_matrix.csv", "k_matrix.npy", "Kij.csv", "Kij.npy"),
    )
    data["k_direct_std"] = read_first_matrix(
        rate_dir,
        ("k_direct_std.csv", "k_direct_std.npy", "k_matrix_std.csv", "k_matrix_std.npy", "Kij_std.csv", "Kij_std.npy"),
    )

    # --- P_jump ---
    data["p_jump"] = read_first_matrix(rate_dir, ("P_jump.csv", "P_jump.npy"))

    # --- pi (populations) ---
    pop_path = rate_dir / "populations.csv"
    if pop_path.exists():
        df = pd.read_csv(pop_path)
        data["pi"] = df["pi"].to_numpy(dtype=np.float64)
    else:
        pi_npy = rate_dir / "pi.npy"
        if pi_npy.exists():
            data["pi"] = np.load(pi_npy)
        else:
            data["pi"] = None

    # --- rate unit from rate_constants.csv ---
    rc_path = rate_dir / "rate_constants.csv"
    if rc_path.exists():
        df = pd.read_csv(rc_path)
        if "k_unit" in df.columns and len(df) > 0:
            data["rate_unit"] = str(df["k_unit"].iloc[0])

    data["n_states"] = data["k_mfpt"].shape[0]
    return data


def latex_directed_rate_table(
    rate_dir: Path,
    reference_dir: Path | None = None,
    output_file: str = "transition_pair_table.tex",
    caption: str = "Directed direct transition rates from committor vector.",
    label: str = "tab:directed_transition_rates",
    precision: int = 3,
) -> None:
    """Build a LaTeX table of directed direct kij rates, sorted by state index."""

    text = render_directed_direct_rate_table(
        rate_dir=rate_dir,
        output_file=output_file,
        caption=caption,
        label=label,
        precision=precision,
    )
    Path(output_file).write_text(text)
    print(f"Wrote LaTeX table to: {output_file}")


def render_directed_direct_rate_table(
    rate_dir: Path,
    output_file: str = "transition_pair_table.tex",
    caption: str = "Directed direct transition rates from committor vector.",
    label: str = "tab:directed_transition_rates",
    precision: int = 3,
) -> str:
    """Render directed direct kij rates in natural (i, j) index order."""

    data = read_rate_dir(rate_dir)
    n = data["n_states"]
    k_direct = data["k_direct"]
    p_jump = data["p_jump"]
    pi = data["pi"]
    rate_unit = data["rate_unit"] or ""
    if k_direct is None:
        raise FileNotFoundError(f"No direct kij matrix found in {rate_dir}; expected k_direct.csv or Kij.csv.")

    # Build rows: one per directed reaction i -> j (i != j)
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            entry = {
                "i": i,
                "j": j,
                "k_ij": k_direct[i, j],
            }
            if p_jump is not None:
                entry["p_ij"] = p_jump[i, j]
            else:
                entry["p_ij"] = None
            if pi is not None:
                entry["pi_i"] = pi[i]
                entry["pi_j"] = pi[j]
            else:
                entry["pi_i"] = None
                entry["pi_j"] = None
            rows.append(entry)

    # --- Build LaTeX ---
    has_pi = pi is not None
    has_p_jump = p_jump is not None

    # Column spec
    ncols = 2  # Reaction + k
    if has_p_jump:
        ncols += 1
    if has_pi:
        ncols += 2

    unit_str = f" ({rate_unit})" if rate_unit else ""

    header_cols = [rf"Reaction", rf"$k^{{\mathrm{{direct}}}}_{{ij}}${unit_str}"]
    if has_p_jump:
        header_cols.append(r"$P(i\rightarrow j)$")
    if has_pi:
        header_cols.append(r"$\pi_i$")
        header_cols.append(r"$\pi_j$")

    col_spec = "c" * ncols

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\resizebox{0.9\columnwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\hline",
        " & ".join(header_cols) + r" \\",
        r"\hline",
    ]

    for row in rows:
        i, j = row["i"], row["j"]
        parts = [rf"${i} \rightarrow {j}$"]

        # k_ij
        parts.append(fmt_sci(row["k_ij"], precision))

        # P(i -> j)
        if has_p_jump:
            p_val = row["p_ij"]
            parts.append(fmt_sci(p_val, precision) if p_val is not None else "--")

        # pi_i, pi_j
        if has_pi:
            parts.append(fmt_sci(row["pi_i"], precision))
            parts.append(fmt_sci(row["pi_j"], precision))

        lines.append(" & ".join(parts) + r" \\")

    lines += [
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


def latex_pairwise_rate_table(
    rate_dir: Path,
    reference_dir: Path | None = None,
    output_file: str = "transition_pair_table.tex",
    caption: str = "Pairwise MFPT-derived rates from committor vector and reference counting.",
    label: str = "tab:pairwise_transition_rates",
    precision: int = 3,
) -> None:
    """Build a compact LaTeX table of pairwise k_mfpt comparisons."""

    text = render_pairwise_mfpt_reference_table(
        rate_dir=rate_dir,
        reference_dir=reference_dir,
        output_file=output_file,
        caption=caption,
        label=label,
        precision=precision,
    )
    Path(output_file).write_text(text)
    print(f"Wrote LaTeX table to: {output_file}")


def render_pairwise_mfpt_reference_table(
    rate_dir: Path,
    reference_dir: Path | None = None,
    output_file: str = "transition_pair_table.tex",
    caption: str = "Pairwise MFPT-derived rates from committor vector and reference counting.",
    label: str = "tab:pairwise_transition_rates",
    precision: int = 3,
) -> str:
    """Render one row per i<j with forward/backward k_mfpt values."""

    data = read_rate_dir(rate_dir)
    n = data["n_states"]
    k_mfpt = data["k_mfpt"]
    k_mfpt_std = data["k_mfpt_std"]
    rate_unit = data["rate_unit"] or ""

    ref_k = None
    ref_std = None
    ref_unit = None
    if reference_dir is not None:
        ref_dir = Path(reference_dir)
        if ref_dir.is_dir():
            ref_data = read_rate_dir(ref_dir)
            ref_k = ref_data["k_mfpt"]
            ref_std = ref_data["k_mfpt_std"]
            ref_unit = ref_data.get("rate_unit") or ""
        else:
            ref_k = read_matrix(ref_dir)

    has_ref = ref_k is not None

    ncols = 2  # Pair + k_ij/k_ji
    if has_ref:
        ncols += 1

    unit_str = f" ({rate_unit})" if rate_unit else ""
    ref_unit_str = f" ({ref_unit})" if ref_unit else ""

    header_cols = [r"Pair", rf"$k^{{\mathrm{{MFPT}}}}_{{ij}}$ / $k^{{\mathrm{{MFPT}}}}_{{ji}}${unit_str}"]
    if has_ref:
        header_cols.append(rf"Ref $k^{{\mathrm{{MFPT}}}}_{{ij}}$ / $k^{{\mathrm{{MFPT}}}}_{{ji}}${ref_unit_str}")

    col_spec = "c" * ncols

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\resizebox{0.9\columnwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\hline",
        " & ".join(header_cols) + r" \\",
        r"\hline",
    ]

    for i in range(n):
        for j in range(i + 1, n):
            parts = [rf"${i} \leftrightarrow {j}$"]

            # k_ij / k_ji
            parts.append(
                fmt_rate(k_mfpt[i, j], None if k_mfpt_std is None else k_mfpt_std[i, j], precision)
                + r" / "
                + fmt_rate(k_mfpt[j, i], None if k_mfpt_std is None else k_mfpt_std[j, i], precision)
            )

            # Reference
            if has_ref:
                parts.append(
                    fmt_rate(ref_k[i, j], None if ref_std is None else ref_std[i, j], precision)
                    + r" / "
                    + fmt_rate(ref_k[j, i], None if ref_std is None else ref_std[j, i], precision)
                )

            lines.append(" & ".join(parts) + r" \\")

    lines += [
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a LaTeX table from TensorQ rate_out directory."
    )
    parser.add_argument(
        "rate_dir",
        nargs="?",
        default="rate_out",
        help="Path to the rate_out directory (or a single k_mfpt .csv/.npy file).",
    )
    parser.add_argument(
        "reference",
        nargs="?",
        default=None,
        help="Optional reference rate_out directory (or single k_mfpt .csv/.npy file).",
    )
    parser.add_argument(
        "-o", "--output", default="transition_pair_table.tex", help="Output LaTeX file."
    )
    parser.add_argument(
        "--caption",
        default="Transition rates from committor vector.",
        help="Table caption.",
    )
    parser.add_argument("--label", default="tab:transition_rates", help="LaTeX label.")
    parser.add_argument(
        "--precision", type=int, default=3, help="Number of significant figures."
    )
    parser.add_argument(
        "--mode",
        choices=("both", "directed", "pairwise"),
        default="both",
        help=(
            "both: write direct-kij and compact k_mfpt/reference tables; "
            "directed: one row per i->j in index order; pairwise: one row per i<->j pair."
        ),
    )
    args = parser.parse_args()

    rate_path = Path(args.rate_dir)
    if not rate_path.is_dir():
        # Backward compat: single file
        ref_path = Path(args.reference) if args.reference else None
        # Use old function signature
        k_mfpt = read_matrix(rate_path)
        ref_k = read_matrix(ref_path) if ref_path else None
        # ... fall through to simple pairwise
        print("Single-file mode: use pairwise output.")
        if args.mode in {"both", "directed"}:
            args.mode = "pairwise"

    ref_path = Path(args.reference) if args.reference else None

    if args.mode == "both":
        direct_text = render_directed_direct_rate_table(
            rate_dir=rate_path if rate_path.is_dir() else rate_path.parent,
            output_file=args.output,
            caption=args.caption + " Direct rates.",
            label=args.label + "_direct",
            precision=args.precision,
        )
        mfpt_text = render_pairwise_mfpt_reference_table(
            rate_dir=rate_path if rate_path.is_dir() else rate_path.parent,
            reference_dir=ref_path,
            output_file=args.output,
            caption=args.caption + " MFPT-derived rates.",
            label=args.label + "_mfpt",
            precision=args.precision,
        )
        Path(args.output).write_text(direct_text + "\n" + mfpt_text)
        print(f"Wrote LaTeX tables to: {args.output}")
    elif args.mode == "directed":
        latex_directed_rate_table(
            rate_dir=rate_path if rate_path.is_dir() else rate_path.parent,
            reference_dir=ref_path,
            output_file=args.output,
            caption=args.caption,
            label=args.label,
            precision=args.precision,
        )
    else:
        latex_pairwise_rate_table(
            rate_dir=rate_path if rate_path.is_dir() else rate_path.parent,
            reference_dir=ref_path,
            output_file=args.output,
            caption=args.caption,
            label=args.label,
            precision=args.precision,
        )


if __name__ == "__main__":
    main()
