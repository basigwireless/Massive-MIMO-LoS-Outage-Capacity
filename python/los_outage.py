"""
Python port of the SPAWC 2017 LoS massive-MIMO outage-capacity simulation.

Original MATLAB code: Prabhu Chandhar, Chandhar Research Labs, Chennai, India
(Calc_SPAWC2017.m, outageCalculation.m, SPAWC2017_MRC.m).

System model
------------
* A base station with an M-element uniform linear array (ULA), element spacing
  ``del`` (half a wavelength by default), placed along the x-axis.
* K single-antenna users (e.g. drones) dropped uniformly at random inside a
  sphere of radius ``R`` (500 m) centred on the array.
* Pure line-of-sight (LoS) channels with a near-field (Fresnel) phase model:

      g_k[m] = exp(-j 2*pi/lambda * (d_k + m^2 del^2 / (2 d_k) - del (m-1) cos(phi_k)))

  where d_k is the user distance and cos(phi_k) = sin(el_k) cos(az_k) is the
  direction cosine with respect to the array axis.
* With maximum-ratio combining (MRC) for user k, the normalised interference is

      U = sum_{j != k} |g_j^H g_k|^2 / M^2

  and the SINR is 1 / (U + 1/(M rho)).  The q-outage capacity is
  C_q = log2(1 + 1/(eta_q + 1/(M rho))) where P(U > eta_q) = q.
"""

from __future__ import annotations

import numpy as np

C_LIGHT = 3e8  # m/s


def drop_users(K: int, n_trials: int, radius: float = 500.0, rng=None):
    """Drop K users uniformly in a sphere for ``n_trials`` independent trials.

    Returns
    -------
    d : (K, n_trials) distances [m]
    cos_phi : (K, n_trials) direction cosine with respect to the array axis
    """
    rng = np.random.default_rng(rng)
    # Elevation with cos(el) uniform in [-1, 1] -> uniform direction on the sphere
    el = np.arcsin(2 * rng.random((K, n_trials)) - 1) + np.pi / 2
    az = 2 * np.pi * rng.random((K, n_trials))
    # Radius with pdf ~ r^2 -> uniform in volume
    d = radius * rng.random((K, n_trials)) ** (1 / 3)
    cos_phi = np.sin(el) * np.cos(az)
    return d, cos_phi


def interference(M: int, K: int, del_: float, lam: float, n_trials: int = 100_000,
                 radius: float = 500.0, rng=None, chunk: int = 2000) -> np.ndarray:
    """Per-interferer terms |g_j^H g_k|^2 for user k = 0 (equivalent to Calc_SPAWC2017.m).

    Returns
    -------
    y : (K-1, n_trials) array; ``y.sum(0) / M**2`` is the normalised interference U.
    """
    d, cos_phi = drop_users(K, n_trials, radius, rng)
    m = np.arange(1, M + 1, dtype=float)
    k0 = 2 * np.pi / lam
    y = np.empty((K - 1, n_trials))

    def phase(dd, cc):  # dd, cc: (..., n) -> (..., n, M)
        dd = dd[..., None]
        cc = cc[..., None]
        return k0 * (dd + (m ** 2 * del_ ** 2) / (2 * dd) - del_ * (m - 1) * cc)

    for s in range(0, n_trials, chunk):
        sl = slice(s, s + chunk)
        pk = phase(d[0, sl], cos_phi[0, sl])            # (n, M)
        pj = phase(d[1:, sl], cos_phi[1:, sl])          # (K-1, n, M)
        inner = np.exp(-1j * (pj - pk[None])).sum(-1)    # g_j^H g_k (up to conj)
        y[:, sl] = np.abs(inner) ** 2
    return y


def outage_threshold(U: np.ndarray, q: float) -> float:
    """eta_q such that P(U > eta_q) = q (empirical (1-q)-quantile, as outageCalculation 'TPT')."""
    return float(np.quantile(U, 1 - q, method="inverted_cdf"))


def outage_capacity(eta, M, rho: float = 1.0):
    """C_q = log2(1 + 1/(eta + 1/(M rho)))."""
    M = np.asarray(M, dtype=float)
    return np.log2(1 + 1 / (np.asarray(eta) + 1 / (M * rho)))


def markov_bound(M, K: int, q: float, rho: float = 1.0):
    """Markov lower bound on the q-outage capacity (Eq. (35) of the paper)."""
    M = np.asarray(M, dtype=float)
    return np.log2(1 + 1 / ((K - 1) / (M * q) + 1 / (M * rho)))


def chebyshev_bound(M, K: int, q: float, rho: float = 1.0):
    """Chebyshev lower bound on the q-outage capacity (Eq. (35) of the paper).

    The expression is transcribed exactly from SPAWC2017_MRC.m.
    """
    M = np.asarray(M, dtype=float)
    x = np.sqrt((K - 1) * (M - 1) * (2 * M - 1)) / 3 / M / q
    return np.log2(1 + M / ((K - 1) + x + 1 / rho))
