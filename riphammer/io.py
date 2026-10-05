"""I/O utilities: mcool → feature matrix, W → cool reconstruction."""

import numpy as np
import pandas as pd
import cooler


CHROM_SIZES_MM10 = {
    "chr1": 195471971, "chr2": 182113224, "chr3": 160039680,
    "chr4": 156508116, "chr5": 151834684, "chr6": 149736546,
    "chr7": 145441459, "chr8": 129401213, "chr9": 124595110,
    "chr10": 130694993, "chr11": 122082543, "chr12": 120129022,
    "chr13": 120421639, "chr14": 124902244, "chr15": 104043685,
    "chr16": 98207768, "chr17": 94987271, "chr18": 90702639,
    "chr19": 61431566, "chrX": 171031299, "chrY": 91744698,
}


def _chrom_sort_key(c):
    n = c.replace("chr", "")
    if n == "X":
        return 100
    if n == "Y":
        return 101
    return int(n)


def _get_chroms(chrom_sizes, autosomes_only=True):
    chroms = sorted(chrom_sizes.keys(), key=_chrom_sort_key)
    if autosomes_only:
        chroms = [c for c in chroms if c not in ("chrX", "chrY")]
    return chroms


def _build_feature_index(chroms, chrom_sizes, resolution, max_dist_bins):
    """Build feature index: (chrom, bin_i, bin_j) for all bin pairs within max_dist."""
    chrom_nbins = {c: (chrom_sizes[c] + resolution - 1) // resolution
                   for c in chroms}

    features = []
    distances = []
    chrom_feat_offset = {}
    chrom_row_offsets = {}
    total = 0

    for c in chroms:
        chrom_feat_offset[c] = total
        nb = chrom_nbins[c]
        offsets = [0] * (nb + 1)
        for i in range(nb):
            offsets[i + 1] = offsets[i] + min(max_dist_bins, nb - 1 - i) + 1
        chrom_row_offsets[c] = offsets
        for i in range(nb):
            j_max = min(i + max_dist_bins, nb - 1)
            for j in range(i, j_max + 1):
                features.append((c, i, j))
                distances.append(j - i)
        total += offsets[nb]

    feat_df = pd.DataFrame(features, columns=["chrom", "bin_i", "bin_j"])
    dist_arr = np.array(distances, dtype=np.int32)

    return feat_df, dist_arr, chrom_feat_offset, chrom_row_offsets, chrom_nbins


def mcool_to_features(cool_uris, resolution=100000, max_dist_bp=5000000,
                      chrom_sizes=None, autosomes_only=True, labels=None):
    """Convert mcool files to feature matrix for Multi-band NMF.

    Parameters
    ----------
    cool_uris : list of str
        Cooler URIs (e.g. "sample.mcool::resolutions/100000").
    resolution : int
        Bin resolution in bp.
    max_dist_bp : int
        Maximum genomic distance in bp.
    chrom_sizes : dict or None
        Chromosome sizes. Defaults to mm10.
    autosomes_only : bool
        Exclude chrX, chrY.
    labels : list of str or None
        Sample labels. If None, derived from filenames.

    Returns
    -------
    X : ndarray, shape (n_features, n_samples)
    features : DataFrame with columns [chrom, bin_i, bin_j]
    distances : ndarray, shape (n_features,)
    sample_names : list of str
    """
    if chrom_sizes is None:
        chrom_sizes = CHROM_SIZES_MM10

    max_dist_bins = max_dist_bp // resolution
    chroms = _get_chroms(chrom_sizes, autosomes_only)
    chrom_nbins = {c: (chrom_sizes[c] + resolution - 1) // resolution
                   for c in chroms}

    feat_df, dist_arr, chrom_feat_offset, chrom_row_offsets, _ = \
        _build_feature_index(chroms, chrom_sizes, resolution, max_dist_bins)

    n_features = len(feat_df)
    n_samples = len(cool_uris)
    X = np.zeros((n_features, n_samples), dtype=np.float64)

    sample_names = []
    for si, uri in enumerate(cool_uris):
        if labels and si < len(labels):
            name = labels[si]
        else:
            import os
            name = os.path.basename(uri.split("::")[0]).replace(".mcool", "")
        sample_names.append(name)

        clr = cooler.Cooler(uri)
        has_balance = "weight" in clr.bins().columns

        for chrom in chroms:
            nb = chrom_nbins[chrom]
            mat = clr.matrix(balance=has_balance).fetch(chrom)
            mat = np.nan_to_num(mat, nan=0.0)
            mat = np.maximum(mat, 0.0)

            off = chrom_feat_offset[chrom]
            row_off = chrom_row_offsets[chrom]
            for i in range(min(nb, mat.shape[0])):
                j_max = min(i + max_dist_bins, nb - 1, mat.shape[1] - 1)
                for j in range(i, j_max + 1):
                    feat_idx = off + row_off[i] + (j - i)
                    X[feat_idx, si] = mat[i, j]

        print(f"  [{si+1}/{n_samples}] {name}: "
              f"nnz={np.sum(X[:, si] > 0)}", flush=True)

    return X, feat_df, dist_arr, sample_names


def w_to_cool(w_col, features, cool_path, chrom_sizes=None,
              resolution=100000, scale=1e6):
    """Reconstruct symmetric contact matrix from W column and save as .cool.

    Parameters
    ----------
    w_col : ndarray, shape (n_features,)
        Single column of W (one NMF component).
    features : DataFrame
        Feature index with columns [chrom, bin_i, bin_j].
    cool_path : str
        Output .cool file path.
    chrom_sizes : dict or None
        Defaults to mm10.
    resolution : int
    scale : float
        Multiply W values by this factor before storing (cooler rounds
        small floats to zero).

    Returns
    -------
    cool_path : str
    """
    if chrom_sizes is None:
        chrom_sizes = CHROM_SIZES_MM10

    chroms = _get_chroms(chrom_sizes, autosomes_only=True)

    bins_list = []
    for c in chroms:
        n_bins = (chrom_sizes[c] + resolution - 1) // resolution
        for b in range(n_bins):
            start = b * resolution
            end = min(start + resolution, chrom_sizes[c])
            bins_list.append((c, start, end))
    bins_df = pd.DataFrame(bins_list, columns=["chrom", "start", "end"])

    chrom_bin_offset = {}
    offset = 0
    for c in chroms:
        chrom_bin_offset[c] = offset
        offset += (chrom_sizes[c] + resolution - 1) // resolution

    bin1_ids, bin2_ids, counts = [], [], []
    for chrom in chroms:
        mask = features["chrom"] == chrom
        if not mask.any():
            continue
        feat_chr = features[mask]
        w_chr = w_col[mask.values]
        pos = w_chr > 0
        if not pos.any():
            continue
        off = chrom_bin_offset[chrom]
        bi = (feat_chr.loc[pos, "bin_i"].values + off).astype(int)
        bj = (feat_chr.loc[pos, "bin_j"].values + off).astype(int)
        vals = w_chr[pos] * scale
        bin1_ids.extend(bi)
        bin2_ids.extend(bj)
        counts.extend(vals)

    pixels = pd.DataFrame({
        "bin1_id": bin1_ids, "bin2_id": bin2_ids, "count": counts,
    })
    cooler.create_cooler(cool_path, bins_df, pixels, ordered=True,
                         columns=["count"])
    return cool_path
