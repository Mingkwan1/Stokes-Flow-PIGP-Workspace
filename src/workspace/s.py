"""
Unsteady Stokes PIGP -- one backward-Euler step from rest, frozen hyperparameters.

    u^0 = 0  ->  u^1 at t = DT

theta is fitted ONCE against y^1 and then cached to theta_frozen.npy.
K and its Cholesky are built once and reused; every later time step is
two triangular solves.

    python step1_frozen.py            # use cache if present
    python step1_frozen.py --fit      # force refit
    python step1_frozen.py --fit --nm-iter 300
"""
import argparse, time
from functools import partial
from pathlib import Path

import numpy as onp
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import jaxopt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unsteady import (
    L, a, avg_width, eta, FBODY, DT, EPS_JITTER, ntrain, width,
    blocks, build_K, build_Q, make_y, neg_log_posterior, theta_init,
    R_G1, R_G2, KSTAR,
)

ap = argparse.ArgumentParser()
ap.add_argument("--fit", action="store_true", help="force re-optimization")
ap.add_argument("--nm-iter", type=int, default=300)
ap.add_argument("--tol", type=float, default=1e-4)
ap.add_argument("--nx", type=int, default=120)
ap.add_argument("--ny", type=int, default=41)
args = ap.parse_args()

OUT = Path("outputs_unsteady"); OUT.mkdir(exist_ok=True)
CACHE = OUT / "theta_frozen.npy"
PARAM_BLOCKS = ["u1u1", "u1u2", "u2u2", "u1p", "u2p", "pp"]


# ----------------------------------------------------------------- reporting
def report_theta(th):
    delta = float(onp.sqrt(eta * DT))          # Stokes layer thickness at t=DT
    dy = float(avg_width / 17)                 # G-grid transverse spacing
    print(f"  {'block':<6}{'gamma':>9}{'lx':>9}{'ly':>9}   note")
    for i, nm in enumerate(PARAM_BLOCKS):
        g = float(th[3 * i])
        lx = float(jnp.exp(th[3 * i + 1]))
        ly = float(jnp.exp(th[3 * i + 2]))
        note = ""
        if ly > 2 * delta:
            note = f"<- ly > 2*sqrt(eta*dt)={2*delta:.3f}, BL underresolved"
        elif min(lx, ly) < dy:
            note = f"<- below G-grid spacing {dy:.3f}"
        print(f"  {nm:<6}{g:>9.3f}{lx:>9.3f}{ly:>9.3f}   {note}")


# ------------------------------------------------------------------ optimizer
def run_nelder_mead(init, fun, maxiter, tol):
    it = [0]
    def cb(xk):
        it[0] += 1
        if it[0] % 25 == 0:
            print(f"  NM {it[0]:<5} | {float(fun(xk)):.6e}", flush=True)
    res = jaxopt.ScipyMinimize(
        fun=fun, method="Nelder-Mead", tol=tol, maxiter=maxiter,
        callback=cb, options={"adaptive": True},
    ).run(init)
    return res.params, it[0]


# ------------------------------------------- frozen-theta factorization + solve
@jax.jit
def factorize(th):
    """One Cholesky, reused for every subsequent time step."""
    K = build_K(th) + (EPS_JITTER ** 2) * jnp.eye(ntrain)
    return jnp.linalg.cholesky(K)


@jax.jit
def alpha_of(Lc, y):
    """K^-1 y via the frozen factor. This is all a later time step needs."""
    return solve_triangular(Lc.T, solve_triangular(Lc, y, lower=True), lower=False)


@partial(jax.jit, static_argnames=("star",))
def posterior(Lc, th, y, R_star, star):
    """Mean and std at R_star using the frozen factor."""
    Q = build_Q(th, R_star, star)
    mean = Q @ alpha_of(Lc, y)
    V = solve_triangular(Lc, Q.T, lower=True)
    pv = jax.vmap(lambda r: KSTAR[star](r, r, th))(R_star)
    std = jnp.sqrt(jnp.clip(pv - jnp.sum(V ** 2, axis=0), 0.0))
    return mean, std


# ------------------------------------------------------------- t = 0 and t = DT
u1_0 = jnp.zeros(R_G1.shape[0])          # initial condition: fluid at rest
u2_0 = jnp.zeros(R_G2.shape[0])
y1 = make_y(u1_0, u2_0)                  # G-rows carry dt*F, everything else 0

print(f"[cfg] dt={DT}  ntrain={ntrain}  dt*F_x={DT*float(FBODY[0]):.4f}  "
      f"sqrt(eta*dt)={onp.sqrt(eta*DT):.3f}")

fun = jax.jit(partial(neg_log_posterior, y=y1))

theta = None
if not args.fit and CACHE.exists():
    theta = jnp.asarray(onp.load(CACHE))
    print(f"[cache] loaded theta from {CACHE}")

if theta is None:
    print(f"Initial value: {float(fun(theta_init)):.6e}", flush=True)
    t0 = time.time()
    theta, nit = run_nelder_mead(theta_init, fun, args.nm_iter, args.tol)
    print(f"Final value:   {float(fun(theta)):.6e}   "
          f"[{time.time()-t0:.0f}s, {nit} iters]")
    onp.save(CACHE, onp.asarray(theta))
    print(f"[cache] wrote {CACHE}")
else:
    print(f"Value at cached theta: {float(fun(theta)):.6e}")

print("\n[theta] FROZEN for all subsequent time steps")
report_theta(theta)

# ---- factor once ----------------------------------------------------------
t0 = time.time()
Lc = factorize(theta)
Lc.block_until_ready()
print(f"\n[factor] Cholesky built once in {time.time()-t0:.1f}s "
      f"(reused for every later step)")

# ---- test grid ------------------------------------------------------------
NX, NY = args.nx, args.ny
x_line = jnp.linspace(0.0, L, NX)
e_line = jnp.linspace(-0.999, 0.999, NY) * 0.5
XX, EE = jnp.meshgrid(x_line, e_line, indexing="ij")
YY = EE * width(XX)
R_test = jnp.stack([XX.ravel(), YY.ravel()], -1)

u1_1, u1_1s = posterior(Lc, theta, y1, R_test, "u1")
u2_1, u2_1s = posterior(Lc, theta, y1, R_test, "u2")
p_1,  p_1s  = posterior(Lc, theta, y1, R_test, "p")

# feedback values at the G points -- this is what step 2 would consume
u1_next, _ = posterior(Lc, theta, y1, R_G1, "u1")
u2_next, _ = posterior(Lc, theta, y1, R_G2, "u2")

# --------------------------------------------------------------- diagnostics
U1 = onp.asarray(u1_1).reshape(NX, NY)
U2 = onp.asarray(u2_1).reshape(NX, NY)
P  = onp.asarray(p_1).reshape(NX, NY)
bound = DT * float(FBODY[0])

print(f"\n{'field':<6}{'min':>12}{'max':>12}{'mean':>12}")
for nm, A in [("u_x", U1), ("u_y", U2), ("p", P)]:
    print(f"{nm:<6}{A.min():>12.5f}{A.max():>12.5f}{A.mean():>12.5f}")

print(f"\n  dt*F_x (inviscid core bound)  = {bound:.5f}")
print(f"  u_x centreline mean / max     = {U1[:, NY//2].mean():.5f} / "
      f"{U1[:, NY//2].max():.5f}")
print(f"  u_x nearest-wall row mean     = {U1[:, 0].mean():+.5f}   (no-slip check)")
print(f"  overshoot past bound          = {max(0.0, U1.max()-bound):.5f}")
print(f"  max posterior std u_x         = {float(u1_1s.max()):.3e}")

onp.savez(OUT / "step1.npz", theta=onp.asarray(theta),
          u1=U1, u2=U2, p=P, u1_std=onp.asarray(u1_1s).reshape(NX, NY),
          XX=onp.asarray(XX), YY=onp.asarray(YY), NX=NX, NY=NY, DT=DT,
          u1_next=onp.asarray(u1_next), u2_next=onp.asarray(u2_next))


# --------------------------------------------------------------------- plots
def plot_compare():
    Xn, Yn = onp.asarray(XX), onp.asarray(YY)
    xw = onp.linspace(0, float(L), 400)
    w = float(avg_width) + 2 * float(a) * onp.sin(2 * onp.pi * xw / float(L))

    U1_0 = onp.zeros_like(U1)                 # t = 0, at rest
    dU1 = U1 - U1_0                           # difference == u^1 here
    vmax = float(onp.abs(dU1).max())

    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    for k, (A, ttl, cmap, lim) in enumerate([
        (U1_0, r"$u_x$ at $t=0$  (initial condition, at rest)", "viridis", (0, vmax)),
        (U1,   rf"$u_x$ at $t=\Delta t={DT}$", "viridis", (0, vmax)),
        (dU1,  r"difference  $u_x(\Delta t)-u_x(0)$", "RdBu_r", (-vmax, vmax)),
    ]):
        c = ax[k].pcolormesh(Xn, Yn, A, cmap=cmap, shading="gouraud",
                             vmin=lim[0], vmax=lim[1])
        ax[k].plot(xw,  w / 2, "k", lw=1.4)
        ax[k].plot(xw, -w / 2, "k", lw=1.4)
        fig.colorbar(c, ax=ax[k], pad=0.02)
        ax[k].set_title(ttl, fontsize=11)
        ax[k].set_ylabel("$y$")
        ax[k].set_aspect("equal", adjustable="box")
    ax[-1].set_xlabel("$x$")
    fig.tight_layout()
    fig.savefig(OUT / "step1_fields.png", dpi=140, bbox_inches="tight")

    # transverse profiles: t=0 vs t=dt at four stations
    fig2, ax2 = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    stations = [(0, "$x=0$"), (NX // 4, "$x=L/4$"),
                (NX // 2, "$x=L/2$"), (3 * NX // 4, "$x=3L/4$")]
    for k, (i, lab) in enumerate(stations):
        ax2[k].plot(U1_0[i], Yn[i], "k:", lw=2, label="$t=0$")
        ax2[k].plot(U1[i],   Yn[i], "C0-", lw=2, label=rf"$t={DT}$")
        ax2[k].axvline(bound, ls="--", c="C3", lw=1.2,
                       label=rf"$\Delta t\,F_x={bound:.2f}$")
        ax2[k].set_title(lab)
        ax2[k].set_xlabel("$u_x$")
        ax2[k].grid(alpha=0.25)
    ax2[0].set_ylabel("$y$")
    ax2[0].legend(fontsize=8, loc="best")
    fig2.suptitle(r"transverse $u_x$ profiles, $t=0$ vs $t=\Delta t$", y=1.02)
    fig2.tight_layout()
    fig2.savefig(OUT / "step1_profiles.png", dpi=140, bbox_inches="tight")
    print(f"\n[plot] {OUT/'step1_fields.png'}")
    print(f"[plot] {OUT/'step1_profiles.png'}")


plot_compare()