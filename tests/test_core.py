"""Tests for Multi-band NMF core."""

import numpy as np
import pytest

from riphammer.core import MultibandNMF


def _make_synthetic(n_features=200, n_samples=5, n_components=3, max_dist=10):
    """Create synthetic data with known structure."""
    rng = np.random.RandomState(0)
    distances = rng.randint(0, max_dist + 1, size=n_features)
    W_true = rng.rand(n_features, n_components) + 0.1
    H_true = rng.rand(n_components, n_samples) + 0.1
    X = W_true @ H_true
    X += rng.rand(*X.shape) * 0.01
    return X, distances, W_true, H_true


# ── Existing tests (backward compatibility) ──────────────────────────

def test_fit_basic():
    X, distances, _, _ = _make_synthetic()
    model = MultibandNMF(n_components=3, max_iter=50, random_state=42)
    model.fit(X, distances, verbose=False)
    assert model.H_.shape == (3, 5)
    assert model.get_W_full().shape == (200, 3)
    assert model.reconstruction_error_ is not None


def test_fit_reduces_error():
    X, distances, _, _ = _make_synthetic()
    model = MultibandNMF(n_components=3, max_iter=100, random_state=42)
    model.fit(X, distances, verbose=True)
    errors = [e for _, e in model.errors_]
    assert errors[-1] < errors[0]


def test_h_normalized_sums_to_one():
    X, distances, _, _ = _make_synthetic()
    model = MultibandNMF(n_components=3, max_iter=50, random_state=42)
    model.fit(X, distances, verbose=False)
    H_norm = model.get_H_normalized()
    col_sums = H_norm.sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)


def test_fold_in():
    X, distances, _, _ = _make_synthetic(n_samples=5)
    model = MultibandNMF(n_components=3, max_iter=100, random_state=42)
    model.fit(X, distances, verbose=False)

    X_new = X[:, :2] + np.random.RandomState(1).rand(200, 2) * 0.01
    H_new = model.fold_in(X_new, distances, max_iter=50)
    assert H_new.shape == (3, 2)
    assert np.all(H_new >= 0)


def test_lambda_w():
    X, distances, _, _ = _make_synthetic()
    model = MultibandNMF(n_components=3, max_iter=50, random_state=42,
                         lambda_w=0.1)
    model.fit(X, distances, verbose=False)
    assert model.H_.shape == (3, 5)


def test_not_fitted_raises():
    model = MultibandNMF()
    with pytest.raises(RuntimeError):
        model.get_W_full()
    with pytest.raises(RuntimeError):
        model.get_H_normalized()
    with pytest.raises(RuntimeError):
        model.fold_in(np.zeros((10, 2)), np.zeros(10, dtype=int))


# ── New: mode and normalization options ──────────────────────────────

def test_single_mode_basic():
    X, distances, _, _ = _make_synthetic()
    model = MultibandNMF(n_components=3, max_iter=50, random_state=42,
                         mode='single')
    model.fit(X, distances, verbose=False)
    assert model.H_.shape == (3, 5)
    assert model.get_W_full().shape == (200, 3)


def test_single_mode_reduces_error():
    X, distances, _, _ = _make_synthetic()
    model = MultibandNMF(n_components=3, max_iter=100, random_state=42,
                         mode='single')
    model.fit(X, distances, verbose=False)
    errors = [e for _, e in model.errors_]
    assert errors[-1] < errors[0]


def test_no_normalize_h():
    X, distances, _, _ = _make_synthetic()
    model = MultibandNMF(n_components=3, max_iter=50, random_state=42,
                         normalize_h=False)
    model.fit(X, distances, verbose=False)
    col_sums = model.H_.sum(axis=0)
    assert not np.allclose(col_sums, 1.0, atol=0.01)


def test_normalize_h_columns_sum_to_one():
    X, distances, _, _ = _make_synthetic()
    model = MultibandNMF(n_components=3, max_iter=50, random_state=42,
                         normalize_h=True)
    model.fit(X, distances, verbose=False)
    col_sums = model.H_.sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)


def test_no_normalize_bands():
    X, distances, _, _ = _make_synthetic()
    model = MultibandNMF(n_components=3, max_iter=50, random_state=42,
                         normalize_bands=False)
    model.fit(X, distances, verbose=False)
    assert model.H_.shape == (3, 5)


@pytest.mark.parametrize("mode", ["multiband", "single"])
@pytest.mark.parametrize("normalize_h", [True, False])
@pytest.mark.parametrize("normalize_bands", [True, False])
def test_all_combinations(mode, normalize_h, normalize_bands):
    """All 2x2x2 combinations run without error."""
    X, distances, _, _ = _make_synthetic()
    model = MultibandNMF(
        n_components=3, max_iter=20, random_state=42,
        mode=mode, normalize_h=normalize_h,
        normalize_bands=normalize_bands,
    )
    model.fit(X, distances, verbose=False)
    assert model.H_.shape == (3, 5)
    assert model.get_W_full().shape == (200, 3)


def test_invalid_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        model = MultibandNMF(mode='invalid')
        X, distances, _, _ = _make_synthetic()
        model.fit(X, distances, verbose=False)


# ── Mathematical equivalence: multiband == single ────────────────────

def test_equivalence_per_iteration():
    """Multi-band and single-matrix modes produce identical results
    when given the same initialization and normalization settings."""
    X, distances, _, _ = _make_synthetic(
        n_features=100, n_samples=4, n_components=2, max_dist=5)

    n_iter = 10
    for nh in [True, False]:
        for nb in [True, False]:
            m_multi = MultibandNMF(
                n_components=2, max_iter=n_iter, random_state=42,
                mode='multiband', normalize_h=nh, normalize_bands=nb)
            m_multi.fit(X, distances, verbose=False, trace=True)

            m_single = MultibandNMF(
                n_components=2, max_iter=n_iter, random_state=42,
                mode='single', normalize_h=nh, normalize_bands=nb)
            m_single.fit(X, distances, verbose=False, trace=True)

            for step in range(n_iter):
                tr_m = m_multi.trace_[step]
                tr_s = m_single.trace_[step]

                np.testing.assert_allclose(
                    tr_m['H'], tr_s['H'], atol=1e-10,
                    err_msg=f"H mismatch at iter {step+1} "
                            f"(nh={nh}, nb={nb})")

                for d in tr_m['W_bands']:
                    np.testing.assert_allclose(
                        tr_m['W_bands'][d], tr_s['W_bands'][d],
                        atol=1e-10,
                        err_msg=f"W_d={d} mismatch at iter {step+1} "
                                f"(nh={nh}, nb={nb})")


def test_equivalence_final_error():
    """Final reconstruction errors match between modes."""
    X, distances, _, _ = _make_synthetic()
    m1 = MultibandNMF(n_components=3, max_iter=100, random_state=42,
                      mode='multiband')
    m1.fit(X, distances, verbose=False)

    m2 = MultibandNMF(n_components=3, max_iter=100, random_state=42,
                      mode='single')
    m2.fit(X, distances, verbose=False)

    np.testing.assert_allclose(
        m1.reconstruction_error_, m2.reconstruction_error_, rtol=1e-10)
    np.testing.assert_allclose(m1.H_, m2.H_, atol=1e-10)


# ── fold_in with normalize_h ────────────────────────────────────────

def test_fold_in_normalize_h():
    X, distances, _, _ = _make_synthetic(n_samples=5)
    model = MultibandNMF(n_components=3, max_iter=50, random_state=42,
                         normalize_h=True)
    model.fit(X, distances, verbose=False)

    X_new = X[:, :2] + np.random.RandomState(1).rand(200, 2) * 0.01
    H_new = model.fold_in(X_new, distances, max_iter=50)
    col_sums = H_new.sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)


def test_fold_in_no_normalize_h():
    X, distances, _, _ = _make_synthetic(n_samples=5)
    model = MultibandNMF(n_components=3, max_iter=50, random_state=42,
                         normalize_h=False)
    model.fit(X, distances, verbose=False)

    X_new = X[:, :2] + np.random.RandomState(1).rand(200, 2) * 0.01
    H_new = model.fold_in(X_new, distances, max_iter=50)
    assert H_new.shape == (3, 2)
    assert np.all(H_new >= 0)
