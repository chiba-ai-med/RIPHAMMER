#!/usr/bin/env python3
"""2×2 comparison benchmark and performance tests.

Run:  python tests/test_benchmark.py
Outputs results to stdout (no files modified).
"""

import time
import tracemalloc
import numpy as np
from scipy.optimize import linear_sum_assignment
from riphammer.core import MultibandNMF


def _make_benchmark_data(n_features=2000, n_samples=10, n_components=4,
                         max_dist=20, seed=0):
    rng = np.random.RandomState(seed)
    distances = rng.randint(0, max_dist + 1, size=n_features)
    W_true = rng.rand(n_features, n_components) + 0.1
    H_true = rng.rand(n_components, n_samples) + 0.1
    X = W_true @ H_true
    X += rng.rand(*X.shape) * 0.01
    return X, distances


def _align_components(H_ref, H_target):
    """Align target components to reference via Hungarian algorithm."""
    K = H_ref.shape[0]
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = np.sum((H_ref[i, :] - H_target[j, :]) ** 2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind


def run_2x2_comparison():
    """Run 4 conditions and compare."""
    X, distances = _make_benchmark_data()
    n_iter = 200

    conditions = {
        'A (multiband, L1-norm)':  dict(mode='multiband', normalize_bands=True),
        'B (single, L1-norm)':     dict(mode='single',    normalize_bands=True),
        'C (multiband, raw)':      dict(mode='multiband', normalize_bands=False),
        'D (single, raw)':         dict(mode='single',    normalize_bands=False),
    }

    results = {}

    for name, opts in conditions.items():
        for nh_label, nh in [('H-norm', True), ('H-raw', False)]:
            key = f"{name} / {nh_label}"

            tracemalloc.start()
            t0 = time.perf_counter()

            model = MultibandNMF(
                n_components=4, max_iter=n_iter, random_state=42,
                normalize_h=nh, **opts)
            model.fit(X, distances, verbose=False)

            elapsed = time.perf_counter() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            results[key] = {
                'model': model,
                'time': elapsed,
                'peak_mb': peak_mem / 1024 / 1024,
                'error': model.reconstruction_error_,
            }

    # Print comparison table
    print("=" * 90)
    print("2×2×2 Comparison (N=2000, M=10, K=4, D≤20, 200 iter)")
    print("=" * 90)
    print(f"{'Condition':<35} {'Error':>10} {'Time (s)':>10} {'Peak MB':>10}")
    print("-" * 90)
    for key, r in results.items():
        print(f"{key:<35} {r['error']:10.4f} {r['time']:10.2f} {r['peak_mb']:10.1f}")

    # Compare H between conditions (with component alignment)
    print("\n" + "=" * 70)
    print("H similarity (Frobenius distance after alignment)")
    print("=" * 70)

    keys = list(results.keys())
    ref_key = keys[0]
    H_ref = results[ref_key]['model'].H_

    for key in keys[1:]:
        H_other = results[key]['model'].H_
        perm = _align_components(H_ref, H_other)
        H_aligned = H_other[perm, :]
        dist = np.linalg.norm(H_ref - H_aligned)
        print(f"  {ref_key} vs {key}: {dist:.6f}")

    # A vs B should be ~0 (mathematical equivalence)
    for nh_label in ['H-norm', 'H-raw']:
        key_a = f"A (multiband, L1-norm) / {nh_label}"
        key_b = f"B (single, L1-norm) / {nh_label}"
        H_a = results[key_a]['model'].H_
        H_b = results[key_b]['model'].H_
        diff = np.max(np.abs(H_a - H_b))
        print(f"\n  A vs B ({nh_label}) max|H_a - H_b|: {diff:.2e} "
              f"(should be ~1e-10)")

        key_c = f"C (multiband, raw) / {nh_label}"
        key_d = f"D (single, raw) / {nh_label}"
        H_c = results[key_c]['model'].H_
        H_d = results[key_d]['model'].H_
        diff2 = np.max(np.abs(H_c - H_d))
        print(f"  C vs D ({nh_label}) max|H_c - H_d|: {diff2:.2e} "
              f"(should be ~1e-10)")

    return results


def run_scalability():
    """Vary N, M, K, D and compare multiband vs single."""
    print("\n" + "=" * 80)
    print("Scalability: multiband vs single (normalize_h=True, normalize_bands=True)")
    print("=" * 80)

    configs = [
        # (N, M, K, max_dist, n_iter)
        (500,   5, 4, 10, 100),
        (2000,  5, 4, 10, 100),
        (5000,  5, 4, 10, 100),
        (10000, 5, 4, 10, 100),
        (2000, 10, 4, 10, 100),
        (2000, 20, 4, 10, 100),
        (2000, 10, 2, 10, 100),
        (2000, 10, 8, 10, 100),
        (2000, 10, 4,  5, 100),
        (2000, 10, 4, 30, 100),
    ]

    print(f"{'N':>7} {'M':>4} {'K':>3} {'D':>3} "
          f"{'MB time':>10} {'SG time':>10} {'MB peak':>10} {'SG peak':>10} "
          f"{'err diff':>10}")
    print("-" * 80)

    for N, M, K, max_dist, n_iter in configs:
        rng = np.random.RandomState(0)
        distances = rng.randint(0, max_dist + 1, size=N)
        X = rng.rand(N, M).astype(np.float64) + 0.01

        results = {}
        for mode in ['multiband', 'single']:
            tracemalloc.start()
            t0 = time.perf_counter()
            model = MultibandNMF(
                n_components=K, max_iter=n_iter, random_state=42,
                mode=mode)
            model.fit(X, distances, verbose=False)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results[mode] = (elapsed, peak / 1024 / 1024, model.reconstruction_error_)

        t_mb, p_mb, e_mb = results['multiband']
        t_sg, p_sg, e_sg = results['single']
        print(f"{N:7d} {M:4d} {K:3d} {max_dist:3d} "
              f"{t_mb:10.2f} {t_sg:10.2f} {p_mb:10.1f} {p_sg:10.1f} "
              f"{abs(e_mb - e_sg):10.2e}")


def run_h_norm_effect():
    """Show how H normalization affects reconstruction error trajectory."""
    X, distances = _make_benchmark_data()

    print("\n" + "=" * 70)
    print("H normalization effect on reconstruction error")
    print("=" * 70)

    for nh, label in [(True, 'H-norm=True'), (False, 'H-norm=False')]:
        model = MultibandNMF(
            n_components=4, max_iter=500, random_state=42,
            normalize_h=nh)
        model.fit(X, distances, verbose=False)
        errs = model.errors_
        print(f"\n  {label}:")
        for it, e in errs:
            print(f"    iter {it:4d}: {e:.6f}")

        monotonic = all(errs[i][1] >= errs[i+1][1]
                        for i in range(len(errs)-1))
        print(f"    Monotonically decreasing: {monotonic}")


if __name__ == '__main__':
    run_2x2_comparison()
    run_scalability()
    run_h_norm_effect()
