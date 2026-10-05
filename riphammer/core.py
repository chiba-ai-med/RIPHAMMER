"""Multi-band NMF: distance-band coupled NMF with shared H."""

import numpy as np


class MultibandNMF:
    """Multi-band NMF for Hi-C deconvolution with TAD preservation.

    Decomposes feature matrix X into band-specific bases W_d and a shared
    mixing coefficient matrix H:  X_d ≈ W_d @ H  for each distance band d.

    Parameters
    ----------
    n_components : int
        Number of NMF components (K).
    max_iter : int
        Maximum number of MU update iterations.
    random_state : int or None
        Random seed for initialization.
    lambda_w : float
        L2 regularization strength on W.
    tol : float
        Convergence tolerance (not currently used for early stopping).
    """

    def __init__(self, n_components=8, max_iter=500, random_state=42,
                 lambda_w=0.0, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.lambda_w = lambda_w
        self.tol = tol
        self.W_bands_ = None
        self.H_ = None
        self.band_indices_ = None
        self.errors_ = []

    def fit(self, X, distances, verbose=True):
        """Fit Multi-band NMF.

        Parameters
        ----------
        X : ndarray, shape (n_features, n_samples)
            Feature matrix (contact values). Features are bin-pairs,
            samples are Hi-C experiments.
        distances : ndarray, shape (n_features,)
            Genomic distance in bins for each feature (d = j - i).
        verbose : bool
            Print progress every 50 iterations.

        Returns
        -------
        self
        """
        N, M = X.shape
        K = self.n_components
        eps = 1e-10

        band_indices = _group_by_distance(distances)
        X_bands = _normalize_bands(X, band_indices)

        rng = np.random.RandomState(self.random_state)
        H = rng.rand(K, M).astype(np.float64) + 1e-6
        W_bands = {}
        for d, idx in band_indices.items():
            W_bands[d] = rng.rand(len(idx), K).astype(np.float64) + 1e-6

        HHt = H @ H.T
        self.errors_ = []

        for it in range(self.max_iter):
            for d in band_indices:
                Xd = X_bands[d]
                Wd = W_bands[d]
                Wd *= (Xd @ H.T) / (Wd @ HHt + self.lambda_w * Wd + eps)
                W_bands[d] = Wd

            num = np.zeros((K, M), dtype=np.float64)
            den = np.zeros((K, K), dtype=np.float64)
            for d in band_indices:
                Wd = W_bands[d]
                num += Wd.T @ X_bands[d]
                den += Wd.T @ Wd
            H *= num / (den @ H + eps)
            HHt = H @ H.T

            if (it + 1) % 50 == 0 or it == 0 or it == self.max_iter - 1:
                err = _reconstruction_error(X_bands, W_bands, H, band_indices)
                self.errors_.append((it + 1, err))
                if verbose:
                    print(f"  iter {it+1}: ||X - WH||_F = {err:.6f}",
                          flush=True)

        self.W_bands_ = W_bands
        self.H_ = H
        self.band_indices_ = band_indices
        self.n_samples_ = M
        self.n_features_ = N
        return self

    def fold_in(self, X, distances, max_iter=100, verbose=False):
        """Estimate H for new samples with W fixed.

        Parameters
        ----------
        X : ndarray, shape (n_features, n_new_samples)
        distances : ndarray, shape (n_features,)
        max_iter : int
        verbose : bool

        Returns
        -------
        H_new : ndarray, shape (n_components, n_new_samples)
        """
        if self.W_bands_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        K = self.n_components
        M_new = X.shape[1]
        eps = 1e-10

        band_indices = _group_by_distance(distances)
        X_bands = _normalize_bands(X, band_indices)

        rng = np.random.RandomState(self.random_state)
        H_new = rng.rand(K, M_new).astype(np.float64) + 1e-6

        for it in range(max_iter):
            num = np.zeros((K, M_new), dtype=np.float64)
            den = np.zeros((K, K), dtype=np.float64)
            for d in band_indices:
                if d not in self.W_bands_:
                    continue
                Wd = self.W_bands_[d]
                num += Wd.T @ X_bands[d]
                den += Wd.T @ Wd
            H_new *= num / (den @ H_new + eps)

            if verbose and ((it + 1) % 20 == 0 or it == 0):
                err = _reconstruction_error(
                    X_bands, self.W_bands_, H_new, band_indices)
                print(f"  fold-in iter {it+1}: ||X - WH||_F = {err:.6f}",
                      flush=True)

        return H_new

    def get_W_full(self):
        """Return concatenated W matrix (n_features x n_components).

        Band-specific W_d are placed back in their original feature positions.
        """
        if self.W_bands_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        W = np.zeros((self.n_features_, self.n_components), dtype=np.float64)
        for d, idx in self.band_indices_.items():
            W[idx, :] = self.W_bands_[d]
        return W

    def get_H_normalized(self):
        """Return H normalized so each column sums to 1 (mixing fractions)."""
        if self.H_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        col_sums = self.H_.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        return self.H_ / col_sums

    @property
    def reconstruction_error_(self):
        """Final reconstruction error (Frobenius norm)."""
        if not self.errors_:
            return None
        return self.errors_[-1][1]


def _group_by_distance(distances):
    band_indices = {}
    for d in range(distances.max() + 1):
        idx = np.where(distances == d)[0]
        if len(idx) > 0:
            band_indices[d] = idx
    return band_indices


def _normalize_bands(X, band_indices):
    X_bands = {}
    for d, idx in band_indices.items():
        Xd = X[idx, :].copy()
        col_sums = Xd.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        X_bands[d] = Xd / col_sums[np.newaxis, :]
    return X_bands


def _reconstruction_error(X_bands, W_bands, H, band_indices):
    total = 0.0
    for d in band_indices:
        if d in W_bands and d in X_bands:
            diff = X_bands[d] - W_bands[d] @ H
            total += np.sum(diff ** 2)
    return np.sqrt(total)
