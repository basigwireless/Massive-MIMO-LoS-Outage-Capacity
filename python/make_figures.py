"""
Reproduce the figures of SPAWC2017_MRC.m in Python.

Usage
-----
    python make_figures.py                 # full run (1e5 trials, as in MATLAB)
    python make_figures.py --trials 20000  # quicker run
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from los_outage import (C_LIGHT, chebyshev_bound, interference, markov_bound,
                        outage_capacity, outage_threshold)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=100_000, help="Monte-Carlo trials per M")
    ap.add_argument("--K", type=int, default=20, help="number of users")
    ap.add_argument("--rho", type=float, default=1.0, help="SNR (linear)")
    ap.add_argument("--fc", type=float, default=2.5e9, help="carrier frequency [Hz]")
    ap.add_argument("--seed", type=int, default=2017)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "figures"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lam = C_LIGHT / args.fc
    del_ = 0.5 * lam
    K, rho = args.K, args.rho
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------ simulation
    M_sim = np.r_[np.arange(10, 101, 10), np.arange(200, 1001, 100)]
    M_cdf = [20, 50, 100, 200, 500, 1000]
    U_store, C1, C10 = {}, [], []
    for M in M_sim:
        t0 = time.time()
        U = interference(int(M), K, del_, lam, args.trials, rng=rng).sum(0) / M ** 2
        if M in M_cdf:
            U_store[int(M)] = U
        C1.append(outage_capacity(outage_threshold(U, 0.01), M, rho))
        C10.append(outage_capacity(outage_threshold(U, 0.10), M, rho))
        print(f"M={M:5d}  C_1%={C1[-1]:.3f}  C_10%={C10[-1]:.3f}  ({time.time() - t0:.1f}s)")
    for M in M_cdf:  # M=20/50 are not in M_sim
        if M not in U_store:
            U_store[M] = interference(M, K, del_, lam, args.trials, rng=rng).sum(0) / M ** 2

    np.savez(os.path.join(args.out, "results.npz"), M=M_sim, C_out_1=C1, C_out_10=C10)

    M = np.arange(10, 1001)

    # ------------------------------------------------------------- Figures 1 & 2
    for fig_no, q, Csim in [(1, 0.01, C1), (2, 0.10, C10)]:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.semilogy(M, markov_bound(M, K, q, rho), label="Markov bound (Eq. (35))")
        ax.semilogy(M, chebyshev_bound(M, K, q, rho), "--", label="Chebyshev bound (Eq. (35))")
        ax.semilogy(M_sim, Csim, "o-", ms=4, label="Simulation")
        ax.set_xlabel("Number of BS antennas $M$")
        ax.set_ylabel(f"{q * 100:.0f}% outage capacity (bits/s/Hz)")
        ax.set_title(f"{q * 100:.0f}% outage capacity, K = {K}, ρ = {rho:g}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"fig{fig_no}_outage_capacity_{int(q * 100)}pct.png"), dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ Figure 3
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 7))
    Ma = np.r_[np.arange(20, 101, 20), np.arange(200, 2001, 100)]
    Mb = np.arange(1e5, 1e7 + 1, 1e5)
    for q in [0.001, 0.01, 0.1]:
        lab = f"q = {q * 100:g}%"
        a1.plot(Ma, np.log2(1 + Ma * rho / (rho * (K - 1) / q + 1)), label=lab)
        a2.semilogx(Mb, np.log2(1 + Mb * rho / (rho * (K - 1) / q + 1)), label=lab)
    for a in (a1, a2):
        a.set_xlabel("$M$")
        a.set_ylabel(r"$R_q^{\mathrm{Mkv,lb}}$ (bits/s/Hz)")
        a.grid(True, which="both", alpha=0.3)
        a.legend()
        a.autoscale(tight=True)
    a1.set_title("Markov lower bound on the q-outage rate")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig3_markov_lower_bound.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ Figure 4
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for Mc in M_cdf:
        U = np.sort(U_store[Mc])
        ax.step(U, np.arange(1, U.size + 1) / U.size, where="post", label=f"M = {Mc}")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Normalised interference $U$")
    ax.set_ylabel("CDF")
    ax.set_title(f"Empirical CDF of $U$, K = {K}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig4_interference_cdf.png"), dpi=150)
    plt.close(fig)

    print(f"Figures written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
