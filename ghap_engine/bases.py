"""Basis functions: natural cubic splines, lag polynomials, DLNM cross-basis, Fourier terms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NaturalSpline:
    """Natural cubic spline basis without intercept (equivalent span to R splines::ns).

    Linear beyond the boundary knots, so evaluation outside the fitted range
    (forecasts, extreme exposures) extrapolates linearly rather than exploding.
    """

    internal: np.ndarray
    boundary: tuple[float, float]

    @classmethod
    def from_data(cls, x: np.ndarray, df: int, knots_at: np.ndarray | None = None) -> "NaturalSpline":
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        lo, hi = float(np.min(x)), float(np.max(x))
        if df <= 1 or hi <= lo:
            return cls(np.array([]), (lo, hi if hi > lo else lo + 1.0))
        probs = knots_at if knots_at is not None else np.linspace(0, 1, df + 1)[1:-1]
        internal = np.unique(np.quantile(x, probs))
        internal = internal[(internal > lo) & (internal < hi)]
        return cls(internal, (lo, hi))

    @property
    def df(self) -> int:
        return len(self.internal) + 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        lo, hi = self.boundary
        scale = hi - lo
        z = (x - lo) / scale  # rescale to [0, 1] for numerical conditioning
        if len(self.internal) == 0:
            return z[..., None]
        knots = np.concatenate([[0.0], (self.internal - lo) / scale, [1.0]])
        k_last = knots[-1]

        def d(k):
            return (np.clip(z - knots[k], 0, None) ** 3 - np.clip(z - k_last, 0, None) ** 3) / (k_last - knots[k])

        d_pen = d(len(knots) - 2)
        cols = [z] + [d(k) - d_pen for k in range(len(knots) - 2)]
        return np.stack(cols, axis=-1)


def lag_basis(lag_max: int, degree: int | None = None) -> np.ndarray:
    """Polynomial lag basis C (rows = lags 0..L). Degree 1 for L<=2 (framework), else 2."""
    if lag_max == 0:
        return np.ones((1, 1))
    if degree is None:
        degree = 1 if lag_max <= 2 else 2
    lags = np.arange(lag_max + 1) / lag_max
    return np.vander(lags, degree + 1, increasing=True)


def lag_matrix(series: np.ndarray, lag_max: int) -> np.ndarray:
    """Columns are x_t, x_{t-1}, ..., x_{t-L} for a single ordered monthly series (NaN where unavailable)."""
    n = len(series)
    out = np.full((n, lag_max + 1), np.nan)
    for lag in range(lag_max + 1):
        out[lag:, lag] = series[: n - lag]
    return out


def crossbasis(lagged: np.ndarray, spline: NaturalSpline, cmat: np.ndarray) -> np.ndarray:
    """DLNM cross-basis: cb[n, j*K + k] = sum_l b_j(x_{n,l}) c_k(l)."""
    b = spline(lagged)  # n x (L+1) x J
    return np.einsum("nlj,lk->njk", b, cmat).reshape(lagged.shape[0], -1)


def reference_crossbasis(ref: float, spline: NaturalSpline, cmat: np.ndarray) -> np.ndarray:
    """Cross-basis row for a constant exposure history at the reference value."""
    return np.kron(spline(np.array([ref]))[0], cmat.sum(axis=0))


def cumulative_gradient(x: np.ndarray, ref: float, spline: NaturalSpline, cmat: np.ndarray) -> np.ndarray:
    """Rows g(x) such that log cumulative RR(x vs ref) = g(x) @ beta_cb (constant exposure over all lags)."""
    db = spline(np.atleast_1d(x)) - spline(np.array([ref]))  # m x J
    csum = cmat.sum(axis=0)
    return np.einsum("mj,k->mjk", db, csum).reshape(len(db), -1)


def lag_gradient(x: float, ref: float, lag: int, spline: NaturalSpline, cmat: np.ndarray) -> np.ndarray:
    """Row g such that lag-specific log RR at `lag` for exposure x vs ref = g @ beta_cb."""
    db = spline(np.array([x]))[0] - spline(np.array([ref]))[0]
    return np.kron(db, cmat[lag])


def fourier(month_of_year: np.ndarray, harmonics: int) -> np.ndarray:
    """Seasonal harmonics of the calendar month (1..12)."""
    m = np.asarray(month_of_year, float)
    cols = []
    for k in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * k * m / 12), np.cos(2 * np.pi * k * m / 12)]
    return np.column_stack(cols) if cols else np.empty((len(m), 0))
