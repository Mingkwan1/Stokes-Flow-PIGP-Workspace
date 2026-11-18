"""
Start-up of pressure-driven flow in a 2D plane channel.

Waters & King (1970), "Unsteady flow of an elastico-viscous liquid",
Rheol. Acta 9, 345-355, Section 3(a):
    Eq. [64]  viscoelastic (Oldroyd liquid B; single-mode Maxwell when S2 = 0)
    Eq. [65]  Newtonian

    u/u0 = -4 y1 (y1 - 1) - 32 * sum_{n=n_start}^{n_end} sin(N y1)/N^3 * A_N(t1),
    N = (2n - 1) pi

    Newtonian    : A_N = exp(-N^2 t1)
    Viscoelastic : A_N = exp(-alpha_N t1 / (2 S1)) * G_N(t1)                 [64]
        G_N   = cosh(beta_N t1/(2S1)) + (1 + N^2 (S2 - 2 S1))/beta_N * sinh(beta_N t1/(2S1))   [42]
        alpha_N = 1 + S2 N^2,  beta_N = sqrt((1 + S2 N^2)^2 - 4 S1 N^2)                       [28]

Dimensionless groups [24]:
    y1 = y/h, t1 = eta0 t/(rho h^2), u0 = -k h^2/(8 eta0)  (k = dp/dx),
    S1 = eta0 lam1/(rho h^2), S2 = eta0 lam2/(rho h^2),  S1 > S2 >= 0.

The flow is 2D (x along the channel, y across it): u = u(y, t), v = 0.

Example
-------
    python channel_startup.py --model viscoelastic --n_end 100
    python channel_startup.py --model newtonian    --n_end inf
    python channel_startup.py --model both --S1 5 --S2 0 --n_start 1 --n_end 500
"""
import argparse
import math
import warnings
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

# ----------------------------------------------------------------------------
# Default fluid: 100 mM CPyCl / 60 mM NaSal wormlike-micelle solution, 25 C.
# Linear viscoelasticity well described by a single-mode Maxwell model:
#   lambda = 1.4 s, eta0 = 49 Pa s, G0 = 29 Pa  (arXiv:2504.02951).
# Density is not reported there -> water-like value assumed.
# ----------------------------------------------------------------------------
DEFAULT_FLUID = dict(eta0=49.0, lam1=1.4, lam2=0.0, rho=1000.0, h=0.01)


# ============================================================================
# Modal time functions A_N(t1)
# ============================================================================
def mode_newtonian(N, t):
    """A_N(t) = exp(-N^2 t). N: (K,), t: (Nt,) -> (K, Nt)."""
    return jnp.exp(-(N[:, None] ** 2) * t[None, :])


def mode_viscoelastic(N, t, S1, S2):
    """A_N(t) = exp(-alpha t/(2S1)) G_N(t), written without overflow.

    e^{-ax} cosh(bx)    = (e^{(b-a)x} + e^{-(b+a)x}) / 2
    e^{-ax} sinh(bx)/b  = (e^{(b-a)x} - e^{-(b+a)x}) / (2b)
    with b - a = -4 S1 N^2 / (a + b)   (avoids cancellation at large N).
    beta may be imaginary (oscillating modes); the result is real either way.
    """
    N2 = N[:, None] ** 2                                   # (K,1)
    alpha = 1.0 + S2 * N2
    beta = jnp.sqrt((alpha**2 - 4.0 * S1 * N2).astype(jnp.complex128))
    c = 1.0 + N2 * (S2 - 2.0 * S1)
    x = t[None, :] / (2.0 * S1)                            # (1,Nt)

    ep = jnp.exp((-4.0 * S1 * N2 / (alpha + beta)) * x)    # e^{(b-a)x}
    em = jnp.exp(-(beta + alpha) * x)                      # e^{-(b+a)x}
    cosh_part = 0.5 * (ep + em)

    bx = beta * x
    small = jnp.abs(bx) < 1e-4                             # beta -> 0 limit
    beta_safe = jnp.where(jnp.abs(beta) < 1e-300, 1.0, beta)
    sinh_over_beta = jnp.where(
        small,
        x * jnp.exp(-alpha * x) * (1.0 + bx**2 / 6.0),
        (ep - em) / (2.0 * beta_safe),
    )
    return jnp.real(cosh_part + c * sinh_over_beta)


@partial(jax.jit, static_argnames="model")
def block_sum(y, t, n_blk, mask, S1, S2, model):
    """Sum of series terms for one block of n values.

    Returns
    -------
    contrib : (Ny, Nt)  sum_k 32 sin(N_k y)/N_k^3 A_k(t)
    bound   : scalar    sum_k 32/N_k^3 max_t |A_k|  (upper bound of |contrib|)
    """
    N = (2.0 * n_blk - 1.0) * jnp.pi
    if model == "newtonian":
        A = mode_newtonian(N, t)
    else:
        A = mode_viscoelastic(N, t, S1, S2)
    w = mask * 32.0 / N**3                                 # (K,)
    S = jnp.sin(y[:, None] * N[None, :]) * w[None, :]      # (Ny,K)
    contrib = S @ A                                        # (Ny,Nt)
    bound = jnp.sum(w * jnp.max(jnp.abs(A), axis=1))
    return contrib, bound


def velocity(y, t, model, S1=0.0, S2=0.0, n_start=1, n_end=100,
             tol=1e-6, block=256, n_max=1_000_000):
    """u/u0 on the grid (y, t). Eq. [64] (viscoelastic) or [65] (newtonian).

    n_end : int or math.inf. With inf, blocks of terms are added until the
            block's bound < tol (or n_max is reached).
    Returns (u (Ny, Nt) numpy array, last n used).
    """
    if model == "viscoelastic" and not (S1 > 0 and S1 >= S2 >= 0):
        raise ValueError(f"Need S1 >= S2 >= 0 and S1 > 0, got S1={S1}, S2={S2}")
    y = jnp.asarray(y, jnp.float64)
    t = jnp.asarray(t, jnp.float64)
    infinite = math.isinf(n_end)
    if not infinite:
        block = int(min(block, n_end - n_start + 1))

    total = jnp.zeros((y.size, t.size))
    n0 = n_start
    while True:
        n_blk = jnp.arange(n0, n0 + block, dtype=jnp.float64)
        mask = jnp.ones(block) if infinite else (n_blk <= n_end).astype(jnp.float64)
        contrib, bound = block_sum(y, t, n_blk, mask, float(S1), float(S2), model)
        total = total + contrib
        n_last = n0 + block - 1
        if not infinite and n_last >= n_end:
            n_last = int(n_end)
            break
        if infinite and float(bound) < tol:
            break
        if n_last >= n_max:
            warnings.warn(f"n_max={n_max} reached before tol={tol} (last block bound "
                          f"{float(bound):.2e}). Increase --n_max or --tol.")
            break
        n0 += block

    steady = -4.0 * y * (y - 1.0)
    return np.asarray(steady[:, None] - total), n_last


# ============================================================================
# Plotting
# ============================================================================
# Default validated palette (dataviz skill): series-1 blue, series-2 orange,
# blue ramp for sequential magnitude.
COL = {"viscoelastic": "#2a78d6", "newtonian": "#eb6834"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
DIV_CMAP = LinearSegmentedColormap.from_list("div", ["#d95926", "#f0efec", "#184f95"])
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})


def plot_channel_fields(results, y, L=4.0, fname="channel_field.png"):
    """2D channel (x, y) colored by u/u0, with arrows, at each snapshot time.
    One row per model, each with its own colour scale."""
    models = list(results)
    ns = max(len(results[m]["t_snap"]) for m in models)
    x = np.linspace(0, L, 60)
    xq, yq_idx = np.linspace(0.25, L - 0.25, 8), np.linspace(0, y.size - 1, 11).astype(int)
    fig, axes = plt.subplots(len(models), ns, squeeze=False,
                             figsize=(2.3 * ns + 1.0, 2.1 * len(models) + 0.4),
                             sharex=True, sharey=True, constrained_layout=True)
    for i, m in enumerate(models):
        U, ts_all = results[m]["u_snap"], results[m]["t_snap"]     # (Ny, Ns)
        umin, umax = U.min(), U.max()
        amax = max(abs(umin), abs(umax))
        if umin < -1e-3 * amax:
            norm, cmap = TwoSlopeNorm(vmin=-amax, vcenter=0.0, vmax=amax), DIV_CMAP
        else:
            norm, cmap = plt.Normalize(0.0, umax), SEQ_CMAP
        for j, ts in enumerate(ts_all):
            ax = axes[i, j]
            field = np.repeat(U[:, j][:, None], x.size, axis=1)
            pc = ax.pcolormesh(x, y, field, cmap=cmap, norm=norm, shading="gouraud")
            XQ, YQ = np.meshgrid(xq, y[yq_idx])
            ax.quiver(XQ, YQ, np.repeat(U[yq_idx, j][:, None], xq.size, 1), 0 * XQ,
                      scale=amax * 10, width=0.006, color=INK, alpha=0.6)
            ax.grid(False)
            ax.set_title(f"{m}, t1 = {ts:.3g}", fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel("y1 = y/h")
            if i == len(models) - 1:
                ax.set_xlabel("x / h")
        fig.colorbar(pc, ax=axes[i, :], label=f"u / u0 ({m})", shrink=0.9)
    fig.savefig(fname, dpi=150)
    return fname


def plot_profiles_and_centerline(results, y, fname="profiles.png"):
    """Per model: velocity profiles at snapshot times + centreline history."""
    models = list(results)
    fig, axes = plt.subplots(len(models), 2, squeeze=False, figsize=(9.5, 3.4 * len(models)),
                             constrained_layout=True)
    for i, m in enumerate(models):
        r = results[m]
        U, ts_all = r["u_snap"], r["t_snap"]
        shades = SEQ_CMAP(np.linspace(0.25, 1.0, len(ts_all)))
        ax = axes[i, 0]
        for j, ts in enumerate(ts_all):
            ax.plot(U[:, j], y, color=shades[j], lw=2, label=f"t1={ts:.3g}")
        ax.plot(-4 * y * (y - 1), y, "--", color=MUTED, lw=1.5, label="steady")
        ax.set_title(f"{m}: profiles (n = {r['n_range']})", fontsize=9)
        ax.set_xlabel("u / u0")
        ax.set_ylabel("y1")
        ax.legend(fontsize=7, frameon=False)
        ax = axes[i, 1]
        ax.plot(r["t_line"], r["u_center"], color=COL[m], lw=1.5)
        ax.axhline(1.0, color=MUTED, lw=1, ls="--")
        ax.set_title(f"{m}: centreline u(0.5, t1) / u0", fontsize=9)
        ax.set_xlabel("t1 = eta0 t / (rho h^2)")
        ax.set_ylabel("u / u0")
    fig.savefig(fname, dpi=150)
    return fname


# ============================================================================
# CLI
# ============================================================================
def parse_n_end(s):
    return math.inf if s.lower() in ("inf", "infinity") else int(s)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["viscoelastic", "newtonian", "both"], default="both")
    p.add_argument("--n_start", type=int, default=1, help="first n of the sum (paper: 1)")
    p.add_argument("--n_end", type=parse_n_end, default=100, help="last n, integer or 'inf'")
    p.add_argument("--tol", type=float, default=1e-6, help="stopping tol when n_end=inf")
    p.add_argument("--block", type=int, default=256, help="terms evaluated per block")
    p.add_argument("--n_max", type=int, default=1_000_000, help="hard cap when n_end=inf")
    # fluid: either S1/S2 directly, or physical properties
    p.add_argument("--S1", type=float, default=None, help="override eta0*lam1/(rho h^2)")
    p.add_argument("--S2", type=float, default=None, help="override eta0*lam2/(rho h^2)")
    p.add_argument("--eta0", type=float, default=DEFAULT_FLUID["eta0"], help="Pa s")
    p.add_argument("--lam1", type=float, default=DEFAULT_FLUID["lam1"], help="relaxation time, s")
    p.add_argument("--lam2", type=float, default=DEFAULT_FLUID["lam2"],
                   help="retardation time, s (0 = Maxwell)")
    p.add_argument("--rho", type=float, default=DEFAULT_FLUID["rho"], help="kg/m^3")
    p.add_argument("--h", type=float, default=DEFAULT_FLUID["h"], help="channel gap, m")
    # grids
    p.add_argument("--ny", type=int, default=101)
    p.add_argument("--nt", type=int, default=2000, help="points for the centreline curve")
    p.add_argument("--t_end", type=float, default=None,
                   help="dimensionless end time (default: 6*S1 viscoelastic, 1 newtonian)")
    p.add_argument("--times", type=float, nargs="+", default=None, help="snapshot times t1")
    p.add_argument("--out_dir", type=str, default=".",
                   help="directory for output figures (created if missing)")
    p.add_argument("--no_show", action="store_true")
    a = p.parse_args()

    if a.n_start < 1:
        p.error("--n_start must be >= 1")
    if not math.isinf(a.n_end) and a.n_end < a.n_start:
        p.error("--n_end must be >= --n_start")
    if a.n_start > 1:
        print(f"Note: n_start={a.n_start} drops modes 1..{a.n_start - 1}; the result is a "
              "partial sum, not the physical solution (initial condition u=0 is lost).")

    tscale = a.rho * a.h**2 / a.eta0
    S1 = a.S1 if a.S1 is not None else a.eta0 * a.lam1 / (a.rho * a.h**2)
    S2 = a.S2 if a.S2 is not None else a.eta0 * a.lam2 / (a.rho * a.h**2)
    models = ["viscoelastic", "newtonian"] if a.model == "both" else [a.model]
    print(f"S1 = {S1:.4g}, S2 = {S2:.4g}  ({'Maxwell' if S2 == 0 else 'Oldroyd-B'})")
    print(f"time scale rho h^2/eta0 = {tscale:.3g} s  (t = t1 * {tscale:.3g} s)")

    y = np.linspace(0.0, 1.0, a.ny)
    results = {}
    for m in models:
        # Auto time window: Newtonian steady by t1 ~ 1; Maxwell envelope ~ exp(-t1/(2 S1)).
        t_end = a.t_end or (max(1.0, 6.0 * S1) if m == "viscoelastic" else 1.0)
        t_snap = (np.array(a.times) if a.times
                  else t_end * np.array([0.005, 0.02, 0.05, 0.15, 0.4, 1.0]))
        t_line = np.linspace(0.0, t_end, a.nt)
        u_snap, n1 = velocity(y, t_snap, m, S1, S2, a.n_start, a.n_end, a.tol, a.block, a.n_max)
        u_c, n2 = velocity(np.array([0.5]), t_line, m, S1, S2, a.n_start, a.n_end,
                           a.tol, a.block, a.n_max)
        results[m] = dict(u_snap=u_snap, u_center=u_c[0], t_snap=t_snap, t_line=t_line,
                          n_range=f"{a.n_start}..{max(n1, n2)}")
        print(f"{m:12s}: summed n = {results[m]['n_range']}, t1 in [0, {t_end:.4g}] "
              f"(= {t_end * tscale:.3g} s), centreline max = {u_c.max():.3f}")

    out = Path(a.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    f1 = plot_channel_fields(results, y, fname=out / "channel_field_oldroyd.png")
    f2 = plot_profiles_and_centerline(results, y, fname=out / "profiles_oldroyd.png")
    print(f"saved {f1}, {f2}")
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()