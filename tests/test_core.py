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
