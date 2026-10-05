"""Command-line interface for RIPHAMMER."""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def cmd_fit(args):
    """Run Multi-band NMF."""
    from riphammer.core import MultibandNMF

    print(f"[riphammer fit] loading {args.input}...", flush=True)
    X = np.load(os.path.join(args.input, "X_bulk.npy"))
    feat = pd.read_csv(
        os.path.join(args.input, "features.tsv"), sep="\t",
        header=None, names=["chrom", "bin_i", "bin_j"],
    )
    distances = (feat["bin_j"] - feat["bin_i"]).values

    if args.autosomes_only:
        mask = ~feat["chrom"].isin(["chrX", "chrY"])
        print(f"  filtering to autosomes: {mask.sum()}/{len(feat)}", flush=True)
        X = X[mask.values, :]
        distances = distances[mask.values]
        feat = feat[mask].reset_index(drop=True)

    model = MultibandNMF(
        n_components=args.k,
        max_iter=args.n_iter,
        random_state=args.seed,
        lambda_w=args.lambda_w,
    )
    print(f"  X: {X.shape}, K={args.k}, max_iter={args.n_iter}", flush=True)
    model.fit(X, distances, verbose=True)

    os.makedirs(args.output, exist_ok=True)
    W = model.get_W_full()
    np.savetxt(os.path.join(args.output, f"W_nmf_k{args.k}.tsv"),
               W, delimiter="\t")
    np.savetxt(os.path.join(args.output, f"H_nmf_k{args.k}.tsv"),
               model.H_, delimiter="\t")
    feat.to_csv(os.path.join(args.output, "features.tsv"),
                sep="\t", header=False, index=False)

    import shutil
    cells_src = os.path.join(args.input, "cells.tsv")
    if os.path.exists(cells_src):
        shutil.copy2(cells_src, os.path.join(args.output, "cells.tsv"))

    print(f"\n[saved] {args.output}/", flush=True)

    H_frac = model.get_H_normalized()
    cells_path = os.path.join(args.output, "cells.tsv")
    if os.path.exists(cells_path):
        cells = pd.read_csv(cells_path, sep="\t", header=None,
                            names=["sample", "label"])
        cols = cells["sample"].tolist()
    else:
        cols = [f"s{i}" for i in range(H_frac.shape[1])]
    df = pd.DataFrame(
        H_frac,
        index=[f"comp{k+1}" for k in range(args.k)],
        columns=cols,
    )
    print(f"\nH matrix (normalized):\n{df.round(3).to_string()}", flush=True)


def cmd_tad(args):
    """Detect TADs from NMF components."""
    from riphammer.io import w_to_cool
    from riphammer.tad import call_tads

    print(f"[riphammer tad] loading {args.input}...", flush=True)
    W = np.loadtxt(os.path.join(args.input, f"W_nmf_k{args.k}.tsv"),
                   delimiter="\t")
    feat = pd.read_csv(
        os.path.join(args.input, "features.tsv"), sep="\t",
        header=None, names=["chrom", "bin_i", "bin_j"],
    )

    os.makedirs(args.output, exist_ok=True)

    for k in range(args.k):
        cool_path = os.path.join(args.output, f"comp{k+1}.cool")
        print(f"  comp{k+1}: ", end="", flush=True)
        w_to_cool(W[:, k], feat, cool_path, resolution=args.resolution)
        ins, boundaries = call_tads(
            cool_path, window=args.window, clr_weight_name=None,
        )
        ins.to_csv(
            os.path.join(args.output, f"comp{k+1}_insulation.tsv"),
            sep="\t", index=False,
        )
        print(f"{len(boundaries)} boundaries", flush=True)

    print(f"\n[saved] {args.output}/", flush=True)


def cmd_vectorize(args):
    """Convert mcool files to feature matrix."""
    from riphammer.io import mcool_to_features

    print(f"[riphammer vectorize] {len(args.mcool)} files...", flush=True)
    X, feat_df, dist_arr, sample_names = mcool_to_features(
        args.mcool,
        resolution=args.resolution,
        max_dist_bp=args.max_dist,
        autosomes_only=args.autosomes_only,
        labels=args.labels,
    )

    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, "X_bulk.npy"), X)
    feat_df.to_csv(os.path.join(args.output, "features.tsv"),
                   sep="\t", header=False, index=False)

    with open(os.path.join(args.output, "cells.tsv"), "w") as f:
        for name in sample_names:
            f.write(f"{name}\t{name}\n")

    print(f"\n[saved] {args.output}/ "
          f"(X: {X.shape}, features: {len(feat_df)})", flush=True)


def main():
    parser = argparse.ArgumentParser(
        prog="riphammer",
        description="RIPHAMMER: Multi-band NMF for Hi-C TAD deconvolution",
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command")

    # fit
    p_fit = sub.add_parser("fit", help="Run Multi-band NMF")
    p_fit.add_argument("--input", "-i", required=True,
                       help="Input directory (X_bulk.npy, features.tsv)")
    p_fit.add_argument("--output", "-o", required=True,
                       help="Output directory")
    p_fit.add_argument("--k", type=int, default=8,
                       help="Number of components (default: 8)")
    p_fit.add_argument("--n-iter", type=int, default=500,
                       help="Max iterations (default: 500)")
    p_fit.add_argument("--seed", type=int, default=42)
    p_fit.add_argument("--lambda-w", type=float, default=0.0,
                       help="L2 regularization on W (default: 0)")
    p_fit.add_argument("--autosomes-only", action="store_true", default=True)
    p_fit.add_argument("--no-autosomes-only", dest="autosomes_only",
                       action="store_false")

    # tad
    p_tad = sub.add_parser("tad", help="Detect TADs from NMF components")
    p_tad.add_argument("--input", "-i", required=True,
                       help="NMF output directory")
    p_tad.add_argument("--output", "-o", required=True)
    p_tad.add_argument("--k", type=int, default=8)
    p_tad.add_argument("--resolution", type=int, default=100000)
    p_tad.add_argument("--window", type=int, default=500000,
                       help="Insulation window size (default: 500000)")

    # vectorize
    p_vec = sub.add_parser("vectorize",
                           help="Convert mcool files to feature matrix")
    p_vec.add_argument("mcool", nargs="+", help="mcool URIs")
    p_vec.add_argument("--output", "-o", required=True)
    p_vec.add_argument("--resolution", type=int, default=100000)
    p_vec.add_argument("--max-dist", type=int, default=5000000)
    p_vec.add_argument("--autosomes-only", action="store_true", default=True)
    p_vec.add_argument("--no-autosomes-only", dest="autosomes_only",
                       action="store_false")
    p_vec.add_argument("--labels", nargs="+", default=None,
                       help="Sample labels (same order as mcool files)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    {"fit": cmd_fit, "tad": cmd_tad, "vectorize": cmd_vectorize}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
