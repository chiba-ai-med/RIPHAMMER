"""TAD boundary detection and comparison."""

import numpy as np
import pandas as pd
import cooler
import cooltools


def call_tads(cool_path, window=500000, ignore_diags=2, clr_weight_name=None):
    """Detect TAD boundaries using cooltools insulation score.

    Parameters
    ----------
    cool_path : str
        Path to .cool file.
    window : int
        Window size in bp for insulation score.
    ignore_diags : int
        Number of diagonals to ignore.
    clr_weight_name : str or None
        Weight column name. None for NMF-derived .cool files (no balancing).

    Returns
    -------
    insulation : DataFrame
        cooltools insulation output with boundary calls.
    boundaries : set of (chrom, start) tuples
    """
    clr = cooler.Cooler(cool_path)
    ins = cooltools.insulation(
        clr, [window], ignore_diags=ignore_diags,
        clr_weight_name=clr_weight_name,
    )

    bcol = f"is_boundary_{window}"
    boundaries = set()
    if bcol in ins.columns:
        brows = ins[ins[bcol] == True]
        for _, row in brows.iterrows():
            boundaries.add((row["chrom"], int(row["start"])))

    return ins, boundaries


def compare_boundaries(boundaries_a, boundaries_b, resolution=100000,
                       tolerance=1):
    """Compare two sets of TAD boundaries.

    Parameters
    ----------
    boundaries_a : set of (chrom, start)
        Reference boundaries.
    boundaries_b : set of (chrom, start)
        Query boundaries.
    resolution : int
        Bin size in bp (used with tolerance).
    tolerance : int
        Number of bins tolerance for fuzzy matching.

    Returns
    -------
    dict with keys: n_ref, n_query, exact_overlap, jaccard,
                    recall, precision
    """
    exact = len(boundaries_a & boundaries_b)

    recall_hits = 0
    for chrom, pos in boundaries_a:
        for delta in range(-tolerance, tolerance + 1):
            if (chrom, pos + delta * resolution) in boundaries_b:
                recall_hits += 1
                break

    precision_hits = 0
    for chrom, pos in boundaries_b:
        for delta in range(-tolerance, tolerance + 1):
            if (chrom, pos + delta * resolution) in boundaries_a:
                precision_hits += 1
                break

    n_ref = max(len(boundaries_a), 1)
    n_query = max(len(boundaries_b), 1)
    union = max(len(boundaries_a | boundaries_b), 1)

    return {
        "n_ref": len(boundaries_a),
        "n_query": len(boundaries_b),
        "exact_overlap": exact,
        "jaccard": exact / union,
        "recall": recall_hits / n_ref,
        "precision": precision_hits / n_query,
    }


def insulation_correlation(ins_a, ins_b, window=500000, chroms=None):
    """Compute per-chromosome Pearson correlation of insulation profiles.

    Parameters
    ----------
    ins_a, ins_b : DataFrame
        cooltools insulation output.
    window : int
    chroms : list of str or None
        Chromosomes to compare. None = all shared.

    Returns
    -------
    DataFrame with columns [chrom, r, n_bins]
    """
    ins_col = f"log2_insulation_score_{window}"
    if chroms is None:
        chroms = sorted(
            set(ins_a["chrom"].unique()) & set(ins_b["chrom"].unique()),
            key=lambda c: (
                100 if c.replace("chr", "") == "X"
                else 101 if c.replace("chr", "") == "Y"
                else int(c.replace("chr", ""))
            ),
        )

    results = []
    for chrom in chroms:
        va = ins_a.loc[ins_a["chrom"] == chrom, ins_col].values
        vb = ins_b.loc[ins_b["chrom"] == chrom, ins_col].values
        n = min(len(va), len(vb))
        if n < 10:
            continue
        valid = np.isfinite(va[:n]) & np.isfinite(vb[:n])
        if valid.sum() < 10:
            continue
        r = np.corrcoef(va[:n][valid], vb[:n][valid])[0, 1]
        results.append({"chrom": chrom, "r": r, "n_bins": int(valid.sum())})

    return pd.DataFrame(results)
