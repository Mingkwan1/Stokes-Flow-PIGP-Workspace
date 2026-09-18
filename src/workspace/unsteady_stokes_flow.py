from functools import wraps
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jrd

from jax.scipy.linalg import solve_triangular

import jaxopt

import numpy as onp

import argparse
import hashlib
import json
import time
from pathlib import Path

import plotting
import fem_stokes
import evolution_plot

from functools import partial

import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description="2D Stokes flow PIGP (18 hyperparameters)")
parser.add_argument("--fem", action="store_true",
                    help="solve the same problem with dolfinx and compare")
parser.add_argument("--fem-refit", action="store_true",
                    help="force re-solving the FEM even if a valid cache exists")
parser.add_argument("--fem-nx", type=int, default=200)
parser.add_argument("--fem-ny", type=int, default=80)
parser.add_argument("--fit", action="store_true", help="force re-optimization")
parser.add_argument("--no-cache", action="store_true", help="do not read or write cache")
parser.add_argument("--nm-iter", type=int, default=500)
parser.add_argument("--tol", type=float, default=1e-4)
parser.add_argument("--zero-up", action="store_true",
                    help="force k_up = k_pu = 0 (no u-p cross-covariance); "
                         "skips ~half the derivative-kernel work")

parser.add_argument("--geometry", choices=["sinusoidal", "plates"],
                    default="sinusoidal",
                    help="channel shape: 'sinusoidal' wavy walls (a as set "
                         "below), or 'plates' for flat parallel plates (a=0, "
                         "recovering the analytical Poiseuille problem)")
parser.add_argument("--n-candidate", type=int, default=680,
                    help="size of the random candidate pool that artificial "
                         "points are subsampled from each step; independent "
                         "of N_f/N_s")
parser.add_argument("--n-loop", type=int, default=10,
                    help="number of backward-Euler steps to march")
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--n-artificial", type=int, default=270,
                    help="artificial data points carried between steps "
                         "(subsampled from the N_f collocation grid)")
parser.add_argument("--artificial-seed", type=int, default=42)
parser.add_argument("--fixed-points", action="store_true",
                    help="keep the same artificial-data locations every step; "
                         "lets K be factorized ONCE instead of per step")
parser.add_argument("--fresh-points", action="store_true",
                    help="draw new uniform locations each step instead of "
                         "subsampling the collocation grid (closer to the paper)")
parser.add_argument("--refit-every-step", action="store_true",
                    help="re-optimize theta at every step (very slow)")
parser.add_argument("--plot-every", type=int, default=1,
                    help="write field plots every k steps (0 = only the last)")
parser.add_argument("--test-margin", type=float, default=0.999,
                    help="test grid extends to this fraction of the local "
                         "half-width. --fem caps it at 0.98: the FEM mesh "
                         "stops at the wall and the interpolator returns NaN "
                         "outside it, which poisons the L2 error metric.")

parser.add_argument("--evolution", action="store_true",
                        help="write the u_x/u_y evolution figure at the end "
                             "of the march")
parser.add_argument("--n-snap", type=int, default=5,
                        help="number of time snapshots in that figure")
parser.add_argument("--evolution-cmap", default="RdBu_r",
                        help="colormap, used for BOTH u_x and u_y")
parser.add_argument("--evolution-share-rows", action="store_true",
                        help="put u_x and u_y on one common colour scale")

args = parser.parse_args()

SPECIMEN = f"usf_dt0.05_10loops_{args.n_artificial}artificial_{args.geometry}_200loop"

OUTDIR = Path(__file__).resolve().parent / "outputs"/ SPECIMEN
(OUTDIR / "plots" ).mkdir(parents=True, exist_ok=True)
PLOT_PATH = CACHE_PATH = OUTDIR / "plots" 

CACHE_OUTDIR = Path(__file__).resolve().parent / "cache_outputs"/ SPECIMEN
(CACHE_OUTDIR).mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_OUTDIR / "final_params_unsteady.npz"

def Product_Squared_Exponential_Kernel(r, rp, theta):
    gamma, log_lx ,log_ly = theta[0],theta[1],theta[2]
    return jnp.exp(gamma - 0.5*((r[0]-rp[0])/jnp.exp(log_lx))**2 - 0.5*((r[1]-rp[1])/jnp.exp(log_ly))**2)

@jax.jit
def Cholesky(K, ϵ):
    try:
        L = jnp.linalg.cholesky(K + jnp.diag(ϵ))
        return L
    except:
        print("Unexpected Error")
        raise

def width(x):
    """Width of the pipe at position x."""
    return avg_width + 2 * a * jnp.sin (2 * jnp.pi * x / L)

η = 1.0 # Viscosity of the fluid
ρ = 1.0
Q = 1.0 # Flow rate
L = 2.5 # Characteristic Length
avg_width = 1 # Average width
a = 0.2 if args.geometry == "sinusoidal" else 0.0   # wall amplitude
if args.geometry == "plates":
    print("[config] flat parallel plates: a=0, "f"constant half-width {avg_width/2:.3f}")
ΔP = -30 # Pressure Diff
FBODY = -ΔP/L * jnp.array([1.0, 0.0]) # Body force
MARGIN = 0.999

Δt = args.dt

theta_init = jnp.array([
    1.2, -1.2, -1.2,   # u1-u1
   1.2, -1.2, -1.2,   # u1-u2  
    1.2, -1.2, -1.2,   # u2-u2
   0.0, -1.2, -1.2,   # u1-p
   0.0, -1.2, -1.2,   # u2-p
    1.2, -1.2, -1.2,   # p-p
])

EPS_JITTER = 1e-3

## 1.) velocity at bounday u_b (62 x 2 points)

x_u1_b = jnp.linspace(0.0, L, 62)
y_u1_b = width(x_u1_b) / 2
R_u1 = jnp.stack([x_u1_b, y_u1_b], axis=-1)

x_u2_b = jnp.linspace(0.0, L, 62)
y_u2_b = -width(x_u2_b) / 2
R_u2 = jnp.stack([x_u2_b, y_u2_b], axis=-1)

u1_b = jnp.zeros_like(x_u1_b)
u2_b = jnp.zeros_like(x_u2_b)

R_wall = jnp.concatenate([R_u1, R_u2], axis=0)      # (124, 2)
u_wall = jnp.zeros(R_wall.shape[0])


R_u_train, u1_train, u2_train = R_wall, u_wall, u_wall

## 2.) Slip periodic velocity (15 x 2 points)

y_s_u_1 = jnp.linspace(-avg_width/2, avg_width/2, 15)
x_s_u_1 = jnp.zeros_like(y_s_u_1)
R_dSu1 = jnp.stack([x_s_u_1,y_s_u_1], axis=-1)

y_s_u_2 = jnp.linspace(-avg_width/2, avg_width/2, 15)
x_s_u_2 = jnp.zeros_like(y_s_u_2)
R_dSu2 = jnp.stack([x_s_u_2,y_s_u_2], axis=-1)

s_u_1 = jnp.zeros_like(y_s_u_1)
s_u_2 = jnp.zeros_like(y_s_u_2)

## 3.) Slip periodic pressure (15 points)

y_s_p = jnp.linspace(-avg_width/2, avg_width/2, 15)
x_s_p = jnp.zeros_like(y_s_p)
R_sp = jnp.stack([x_s_p,y_s_p], axis=-1)

s_p = jnp.zeros_like(y_s_p)

## 4.) Governing equations f (337 x 2 points)

N_f = N_s = 340

def points_grid(N=N_f, n_cols=20):
    n_rows = int(jnp.ceil(N / n_cols))                 # 17 -> 340, trim 3
    x_c = (jnp.arange(n_cols) + 0.5) / n_cols * L      # x in [0,L), no duplicate at L
    eta = ((jnp.arange(n_rows) + 0.5) / n_rows - 0.5) * (1 - 2*MARGIN)
    X, E = jnp.meshgrid(x_c, eta, indexing="ij")
    x = jnp.asarray(X.ravel()[:N])
    return x, E.ravel()[:N] * width(x)

x_f, y_f = points_grid(N=N_f)
R_f = jnp.stack([x_f,y_f], axis=-1)

x_f_1 = x_f
y_f_1 = y_f
R_f1 = jnp.stack([x_f_1,y_f_1], axis=-1)

f_1 = jnp.full(N_f, -ΔP / L)
x_f_2 = x_f
y_f_2 = y_f
R_f2 = jnp.stack([x_f_1,y_f_1], axis=-1)
f_2 = jnp.zeros(N_f)

R_G1, R_G2 = R_f1, R_f2

## 5.) Governing equations s (337 points)

x_s, y_s = x_f, y_f 
R_s = jnp.stack([x_s,y_s], axis=-1)
s = jnp.zeros(N_s)

###


def sample_artificial(n, key, fresh=False, margin=0.98):
    """Draw n artificial-data locations from the candidate pool R_candidate."""
    if fresh:
        return sample_uniform(n, key, margin)
    if n >= R_candidate.shape[0]:
        return R_candidate
    idx = jrd.choice(key, R_candidate.shape[0], (n,), replace=False)
    return R_candidate[idx]

def plot_training_points(save_path=None):
    fig, ax = plt.subplots(figsize=(11, 5))

    # --- sinusoidal walls (reference curves, not training points) ----------
    x_wall = onp.linspace(0.0, float(L), 400)
    w = onp.asarray(width(jnp.asarray(x_wall)))
    ax.plot(x_wall,  w/2, color="gray", lw=1.2, zorder=1, label="pipe wall")
    ax.plot(x_wall, -w/2, color="gray", lw=1.2, zorder=1)

    # --- boundary / point sets, each a distinct marker+color ---------------
    groups = [
        (R_u1,   "tab:red",    "o", f"$u_1$ wall no-slip ({R_u1.shape[0]})"),
        (R_u2,   "tab:orange", "o", f"$u_2$ wall no-slip ({R_u2.shape[0]})"),
        (R_dSu1, "tab:purple", "^", f"periodic $u_1$ ({R_dSu1.shape[0]})"),
        (R_dSu2, "tab:pink",   "v", f"periodic $u_2$ ({R_dSu2.shape[0]})"),
        (R_sp,   "tab:brown",  "s", f"periodic $p$ ({R_sp.shape[0]})"),
        (R_f1,   "tab:blue",   ".", f"$f_1$ momentum-x ({R_f1.shape[0]})"),
        (R_f2,   "tab:cyan",   ".", f"$f_2$ momentum-y ({R_f2.shape[0]})"),
        (R_s,    "tab:green",  "x", f"$s$ continuity ({R_s.shape[0]})"),
    ]

    for R, color, marker, label in groups:
        Rn = onp.asarray(R)
        ax.scatter(Rn[:, 0], Rn[:, 1], s=28, c=color, marker=marker,
                   alpha=0.75, edgecolors="none", zorder=3, label=label)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Training point locations, sinusoidal channel")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=4, fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] saved {save_path}")
    plt.show()
    return fig, ax

# plot_training_points(save_path=OUTDIR / "training_points.png")

###

theta = lambda t, i: jax.lax.dynamic_slice(t, (3*i,), (3,))

# ---- operator combinators (unprimed = slot 0, primed = slot 1) --------------
D    = lambda k, a: None if k is None else (lambda r, rp, t: jax.grad(k, 0)(r, rp, t)[a])
Dp   = lambda k, a: None if k is None else (lambda r, rp, t: jax.grad(k, 1)(r, rp, t)[a])
Lap  = lambda k:    None if k is None else (lambda r, rp, t: jnp.trace(jax.hessian(k, 0)(r, rp, t)))
Lapp = lambda k:    None if k is None else (lambda r, rp, t: jnp.trace(jax.hessian(k, 1)(r, rp, t)))
mul  = lambda c, k: None if k is None else (lambda r, rp, t: c * k(r, rp, t))

def add(*ks):
    ks = [k for k in ks if k is not None]
    if not ks:
        return None
    if len(ks) == 1:
        return ks[0]
    return lambda r, rp, t: sum(k(r, rp, t) for k in ks)

_eL  = jnp.array([L, 0.0])
S    = lambda k: None if k is None else (lambda r, rp, t: k(r, rp, t) - k(r + _eL, rp, t))
Sp   = lambda k: None if k is None else (lambda r, rp, t: k(r, rp, t) - k(r, rp + _eL, t))

### Base kernel (Total of 6 kernels and 18 parameters)

k_uu = [[lambda r, rp, t: Product_Squared_Exponential_Kernel(r,  rp, theta(t, 0)),      
         lambda r, rp, t: Product_Squared_Exponential_Kernel(r,  rp, theta(t, 1))],    
        [lambda r, rp, t: Product_Squared_Exponential_Kernel(rp, r,  theta(t, 1)),     
         lambda r, rp, t: Product_Squared_Exponential_Kernel(r,  rp, theta(t, 2))]]     

k_up = [lambda r, rp, t: Product_Squared_Exponential_Kernel(r,  rp, theta(t, 3)),
        lambda r, rp, t: Product_Squared_Exponential_Kernel(r,  rp, theta(t, 4))]
k_pu = k_up

if args.zero_up:
    k_up = [None, None]
    k_pu = [None, None]
    
k_pp = lambda r, rp, t: Product_Squared_Exponential_Kernel(r, rp, theta(t, 5))

### Derived kernels 

k_u_dsu = lambda a,b: Sp(k_uu[a][b])
k_u_dsp = lambda a: Sp(k_up[a])
k_uf = lambda a, b: add(Dp(k_up[a], b), mul(-η, Lapp(k_uu[a][b])))      
k_us = lambda a:    add(*[Dp(k_uu[a][b], b) for b in range(2)])     


k_pf = lambda a: add(Dp(k_pp, a), mul(-η, Lapp(k_pu[a])))
k_ps =              add(*[Dp(k_pu[b], b) for b in range(2)])        

k_dsu_dsu = lambda a,b: Sp(S(k_uu[a][b]))
k_dsu_dsp = lambda a: Sp(S(k_up[a]))
k_dsu_f = lambda a,b: S(k_uf(a,b))
k_dsu_s = lambda a: S(k_us(a))

k_dsp_dsp = Sp(S(k_pp))
k_dsp_f = lambda b: S(k_pf(b))
k_dsp_std = S(k_ps)

k_ff = lambda a, b: add(D(k_pf(b), a),  mul(-η, Lap(k_uf(a, b))))       
k_fs = lambda a:    add(D(k_ps, a),     mul(-η, Lap(k_us(a))))       

k_ss =              add(*[D(k_us(a), a) for a in range(2)])  

# Time dependant kernels (G)

k_fu = lambda a, b: add(D(k_pu[b], a), mul(-η, Lap(k_uu[a][b])))

k_uG  = lambda a, b: add(k_uu[a][b], mul(Δt, k_uf(a, b)))
k_pG  = lambda b:    add(k_pu[b],    mul(Δt, k_pf(b)))

k_GG  = lambda a, b: add(k_uu[a][b],
                         mul(Δt,    k_uf(a, b)),
                         mul(Δt,    k_fu(a, b)),
                         mul(Δt**2, k_ff(a, b)))

k_Gs  = lambda a:    add(k_us(a), mul(Δt, k_fs(a)))

k_dsu_G = lambda a, b: S(k_uG(a, b))
k_dsp_G = lambda b:    S(k_pG(b))

## Buiding K

BLOCK_NAMES = ["u1", "u2", "Su1", "Su2", "Sp", "G1", "G2", "s"]

def kmat(k):
    if k is None:
        return lambda R1, R2, t: jnp.zeros((R1.shape[0], R2.shape[0]))
    return jax.jit(jax.vmap(jax.vmap(k, (None, 0, None)), (0, None, None)))

def make_blocks(R_G):
    """Block layout. Only the G blocks move between time steps."""
    return [("u1",  R_u_train),      # no-slip, u_x
            ("u2",  R_u_train),      # no-slip, u_y
            ("Su1", R_dSu1),         # periodicity, u_x
            ("Su2", R_dSu2),         # periodicity, u_y
            ("Sp",  R_sp),           # periodicity, p
            ("G1",  R_G),            # backward-Euler momentum, x
            ("G2",  R_G),            # backward-Euler momentum, y
            ("s",   R_s)]            # continuity

# blocks = [
#     ("u1",  R_u_train),                 # no-slip, u_x
#     ("u2",  R_u_train),                 # no-slip, u_y
#     ("Su1", R_dSu1),                    # periodicity, u_x
#     ("Su2", R_dSu2),                    # periodicity, u_y
#     ("Sp",  R_sp),                      # periodicity, p
#     ("G1",  R_G1),                      # backward-Euler momentum, x
#     ("G2",  R_G2),                      # backward-Euler momentum, y
#     ("s",   R_s),                       # continuity
# ]

kfn = {
    ("u1","u1"): k_uu[0][0],           ("u1","u2"): k_uu[0][1],
    ("u2","u2"): k_uu[1][1],

    ("u1","Su1"): k_u_dsu(0,0),        ("u1","Su2"): k_u_dsu(0,1),
    ("u2","Su1"): k_u_dsu(1,0),        ("u2","Su2"): k_u_dsu(1,1),
    ("u1","Sp"):  k_u_dsp(0),          ("u2","Sp"):  k_u_dsp(1),

    ("Su1","Su1"): k_dsu_dsu(0,0),     ("Su1","Su2"): k_dsu_dsu(0,1),
    ("Su2","Su1"): k_dsu_dsu(1,0),     ("Su2","Su2"): k_dsu_dsu(1,1),
    ("Su1","Sp"):  k_dsu_dsp(0),       ("Su2","Sp"):  k_dsu_dsp(1),
    ("Sp","Sp"):   k_dsp_dsp,

    ("u1","G1"): k_uG(0,0),            ("u1","G2"): k_uG(0,1),
    ("u2","G1"): k_uG(1,0),            ("u2","G2"): k_uG(1,1),
    ("u1","s"):  k_us(0),              ("u2","s"):  k_us(1),

    ("Su1","G1"): k_dsu_G(0,0),        ("Su1","G2"): k_dsu_G(0,1),
    ("Su2","G1"): k_dsu_G(1,0),        ("Su2","G2"): k_dsu_G(1,1),
    ("Su1","s"):  k_dsu_s(0),          ("Su2","s"):  k_dsu_s(1),

    ("Sp","G1"): k_dsp_G(0),           ("Sp","G2"): k_dsp_G(1),
    ("Sp","s"):  k_dsp_std,

    ("G1","G1"): k_GG(0,0),            ("G1","G2"): k_GG(0,1),
    ("G2","G2"): k_GG(1,1),
    ("G1","s"):  k_Gs(0),              ("G2","s"):  k_Gs(1),

    ("s","s"):   k_ss,
}

def build_K(theta,R):
    blocks = make_blocks(R)
    n = len(blocks)
    rows = [[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            name_i, R_i = blocks[i]
            name_j, R_j = blocks[j]
            k = kfn[(name_i, name_j)]
            rows[i][j] = kmat(k)(R_i, R_j, theta)
            rows[j][i] = rows[i][j].T       
    return jnp.block(rows)

theta_init = jnp.array([
    1.0, jnp.log(0.50), jnp.log(0.30),   # u1-u1
   0.8, jnp.log(0.60), jnp.log(0.35),   # u1-u2  
    1.1, jnp.log(0.55), jnp.log(0.32),   # u2-u2
   0.0, jnp.log(0.70), jnp.log(0.40),   # u1-p
   0.0, jnp.log(0.65), jnp.log(0.38),   # u2-p
    0.7, jnp.log(0.80), jnp.log(0.45),   # p-p
])

theta_init = jnp.array([
    1.2, jnp.log(0.30), jnp.log(0.30),   # u1-u1
   1.2, jnp.log(0.30), jnp.log(0.3),   # u1-u2  
    1.2, jnp.log(0.3), jnp.log(0.3),   # u2-u2
   0.0, jnp.log(0.30), jnp.log(0.30),   # u1-p
   0.0, jnp.log(0.3), jnp.log(0.30),   # u2-p
    1.2, jnp.log(0.30), jnp.log(0.3),   # p-p
])

def make_y(u1_prev, u2_prev):
    """u{1,2}_prev: u^{n-1} evaluated at R_G1 / R_G2."""
    return jnp.concatenate([
        jnp.zeros(R_u_train.shape[0]),        # no-slip u_x
        jnp.zeros(R_u_train.shape[0]),        # no-slip u_y
        s_u_1, s_u_2, s_p,                    # periodicity (all zero)
        u1_prev + Δt * FBODY[0],              # G1
        u2_prev + Δt * FBODY[1],              # G2
        jnp.zeros(N_s),                       # continuity
    ])

# t = 0: fluid at rest
y0 = make_y(jnp.zeros(R_G1.shape[0]), jnp.zeros(R_G2.shape[0]))

def n_train(R_G):
    return (2*R_u_train.shape[0] + R_dSu1.shape[0] + R_dSu2.shape[0]
            + R_sp.shape[0] + 2*R_G.shape[0] + R_s.shape[0])

# y = jnp.concatenate([
#     u1_train,                      # (62,)  no-slip, top wall
#     u2_train,                      # (62,)  no-slip, bottom wall
#     s_u_1,                     # (15,)  periodicity, u^x
#     s_u_2,                     # (15,)  periodicity, u^y
#     s_p,                       # (15,)  periodicity, p
#     f_1,                       # (340,) = -ΔP/L
#     f_2,                       # (340,) = 0
#     s,                         # (340,) = 0
# ])
# print(y)

def inspect_K_blocks(theta,R):
    blocks = make_blocks(R)
    n = len(blocks)
    print(f"{'block':<12}{'shape':<12}{'min':>12}{'max':>12}{'absmax':>12}{'has_nan':>9}{'has_inf':>9}")
    for i in range(n):
        for j in range(i, n):
            name_i, R_i = blocks[i]
            name_j, R_j = blocks[j]
            k = kfn[(name_i, name_j)]
            Kij = kmat(k)(R_i, R_j, theta)
            label = f"{name_i}-{name_j}"
            print(f"{label:<12}{str(Kij.shape):<12}"
                  f"{float(Kij.min()):>12.3e}{float(Kij.max()):>12.3e}"
                  f"{float(jnp.abs(Kij).max()):>12.3e}"
                  f"{bool(jnp.any(jnp.isnan(Kij))):>9}"
                  f"{bool(jnp.any(jnp.isinf(Kij))):>9}")

# inspect_K_blocks(theta_init)

def neg_log_posterior(theta, y, R_G):
    ntrain=y.shape[0]
    K  = build_K(theta,R_G) + (EPS_JITTER**2) * jnp.eye(ntrain)
    Lc = jnp.linalg.cholesky(K)
    v  = solve_triangular(Lc, y, lower=True)
    return (0.5*jnp.dot(v, v) + jnp.sum(jnp.log(jnp.diag(Lc)))
            + 0.5*ntrain*jnp.log(2.0*jnp.pi)
            + 0.5*jnp.sum(theta[0::3]**2)/(2.0**2))

@jax.jit
def factorize(theta, R_G):
    """One Cholesky. Reusable across steps only if R_G is held fixed."""
    n = n_train(R_G)
    K = build_K(theta, R_G) + (EPS_JITTER**2) * jnp.eye(n)
    return jnp.linalg.cholesky(K)

fun = neg_log_posterior

def run_nelder_mead(init_params, fun, maxiter=500, tol=1e-3):
    # Simple list to keep track of iterations across callback calls
    iter_count = [0]
    
    def callback(xk):
        iter_count[0] += 1
        # Calculate current loss/posterior value
        val = fun(xk)
        print(f"Nelder-Mead Iteration {iter_count[0]:<4} | Value: {val:.4e}")

    solver = jaxopt.ScipyMinimize(
        fun=fun,
        method="Nelder-Mead",
        tol=tol,
        maxiter=maxiter,
        callback=callback, # Triggers on each iteration
        options={"disp": True, "adaptive": True},
    )
    
    res = solver.run(init_params)
    return res.params, res.state

PARAM_BLOCKS = ["u1u1", "u1u2", "u2u2", "u1p", "u2p", "pp"]

def _arr_hash(*arrays):
    """Content hash of point sets - catches a moved collocation grid that
    a shape-only check would miss."""
    h = hashlib.sha256()
    for A in arrays:
        h.update(onp.ascontiguousarray(onp.asarray(A), dtype=onp.float64).tobytes())
    return h.hexdigest()[:16]


def config_fingerprint(R_G0):
    """Hash everything that changes the optimum. A cache keyed only on a
    filename would silently return theta fitted for different physics."""
    cfg = {
        "dim": 2, "scheme": "backward-euler",
        "eta": float(η), "rho": float(ρ), "L": float(L), "dP": float(ΔP),
        "a": float(a), "avg_width": float(avg_width),
        "eps": float(EPS_JITTER), "margin": float(MARGIN), "dt": float(Δt),
        "N_f": int(N_f), "N_s": int(N_s),
        "n_artificial": int(args.n_artificial),
        "fresh_points": bool(args.fresh_points),
        "artificial_seed": int(args.artificial_seed),
        "n_wall": int(R_wall.shape[0]),
        "n_dSu": int(R_dSu1.shape[0]), "n_Sp": int(R_sp.shape[0]),
        "blocks": BLOCK_NAMES,
        "ntrain": int(n_train(R_G0)),
        "optimizer": "nelder-mead",
        "nm_iter": int(args.nm_iter), "tol": float(args.tol),
        "theta_init": [float(v) for v in theta_init],
        "points_hash": _arr_hash(R_u_train, R_dSu1, R_dSu2, R_sp, R_G0, R_s),
        "n_artificial": int(args.n_artificial),
        "n_candidate": int(args.n_candidate),
        "zero_u-p": bool(args.zero_up),
    }
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    return h, cfg


def load_cache(fingerprint):
    if not CACHE_PATH.exists():
        return None
    d = onp.load(CACHE_PATH, allow_pickle=False)
    if str(d["fingerprint"]) != fingerprint:
        print("[cache] fingerprint mismatch (config changed) -> refitting")
        return None
    th = jnp.asarray(d["theta"])
    if th.shape != theta_init.shape:
        print(f"[cache] parameter count changed ({th.shape} vs {theta_init.shape}) -> refitting")
        return None
    return th


def save_cache(fingerprint, theta, value):
    onp.savez(CACHE_PATH,
              theta=onp.asarray(theta),
              fingerprint=onp.asarray(fingerprint),
              value=onp.asarray(float(value)))
    print(f"[cache] wrote {CACHE_PATH}")


def report_theta(theta):
    print(f"  {'block':<6}{'gamma':>9}{'lx':>9}{'ly':>9}")
    for i, name in enumerate(PARAM_BLOCKS):
        g  = float(theta[3*i])
        lx = float(jnp.exp(theta[3*i + 1]))
        ly = float(jnp.exp(theta[3*i + 2]))
        note = ""
        if min(lx, ly) < 0.125:  note = "  <- below f-grid spacing"
        if max(lx, ly) > 1.0:    note = "  <- exceeds channel height"
        print(f"  {name:<6}{g:>9.3f}{lx:>9.3f}{ly:>9.3f}{note}")

# ---- test points: a grid inside the channel -------------------------------

def sample_uniform(n, key, margin=0.98):
    kx, ke = jrd.split(key)
    x = jrd.uniform(kx, (n,), minval=0.0, maxval=L)
    e = jrd.uniform(ke, (n,), minval=-0.5, maxval=0.5) * margin
    return jnp.stack([x, e * width(x)], axis=-1)

key = jrd.PRNGKey(args.artificial_seed)
key, k_cand = jrd.split(key)
R_candidate = sample_uniform(args.n_candidate, k_cand)

NX, NY = 120, 41
TEST_MARGIN = min(args.test_margin, 0.98) if args.fem else args.test_margin
if args.fem and args.test_margin > 0.98:
    print(f"[config] --fem: test margin capped {args.test_margin} -> 0.98")
x_line = jnp.linspace(0.0, L, NX)
eta_line = jnp.linspace(-TEST_MARGIN, TEST_MARGIN, NY) * 0.5
XX, EE = jnp.meshgrid(x_line, eta_line, indexing="ij")
YY = EE * width(XX)
R_test = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)

key = jrd.PRNGKey(args.artificial_seed)
key, k0 = jrd.split(key)
R_G = sample_artificial(args.n_artificial, k0, fresh=args.fresh_points)
 
fp, cfg = config_fingerprint(R_G)
print(f"[config] fingerprint {fp}")
print(f"[config] dt={Δt}  n_loop={args.n_loop}  n_artificial={args.n_artificial}"
      f"  ntrain={n_train(R_G)}  dt*F_x={float(Δt/ρ*FBODY[0]):.4f}"
      f"  sqrt(eta*dt/rho)={onp.sqrt(η*Δt/ρ):.3f}")
if args.fixed_points:
    print("[config] fixed artificial-data locations -> K factorized once")
elif args.fresh_points:
    print("[config] fresh uniform artificial-data locations each step")
else:
    print("[config] artificial data resampled from the collocation grid each step")


row_u1 = [k_uu[0][0], k_uu[0][1], Sp(k_uu[0][0]), Sp(k_uu[0][1]),
          Sp(k_up[0]),  k_uG(0, 0), k_uG(0, 1), k_us(0)]

row_u2 = [k_uu[1][0], k_uu[1][1], Sp(k_uu[1][0]), Sp(k_uu[1][1]),
          Sp(k_up[1]),  k_uG(1, 0), k_uG(1, 1), k_us(1)]

row_p  = [k_pu[0],     k_pu[1],    Sp(k_pu[0]),   Sp(k_pu[1]),
          Sp(k_pp),     k_pG(0),    k_pG(1),      k_ps]

ROWS = {"u1": row_u1, "u2": row_u2, "p": row_p}
KSTAR = {"u1": k_uu[0][0], "u2": k_uu[1][1], "p": k_pp}

def build_K_star_obs(theta, R_star, star, R):
    """Q, shape (len(R_star), ntrain). Columns follow `blocks` order."""
    cols = [kmat(k)(R_star, R, theta)
            for k, (_, R) in zip(ROWS[star], make_blocks(R))]
    return jnp.concatenate(cols, axis=1)

def prior_var_diag(theta, R_star, star):
    """diag(K**) only - the full n_star x n_star matrix is never needed."""
    k = KSTAR[star]
    return jax.vmap(lambda r: k(r, r, theta))(R_star)

@partial(jax.jit, static_argnames=("star",))
def predict(Lc,theta, y,R_star,R_G, star="u1"):
    Q  = build_K_star_obs(theta, R_star, star,R_G)

    alpha = solve_triangular(Lc.T, solve_triangular(Lc, y, lower=True), lower=False)
    mean  = Q @ alpha                                    # (n_star,)

    V    = solve_triangular(Lc, Q.T, lower=True)         # (ntrain, n_star)
    var  = prior_var_diag(theta, R_star, star) - jnp.sum(V**2, axis=0)
    std  = jnp.sqrt(jnp.clip(var, 0.0))
    return mean, std

###

# ==================================================================== time loop
 
key = jrd.PRNGKey(args.artificial_seed)
key, k0 = jrd.split(key)
R_G = sample_artificial(args.n_artificial, k0, fresh=args.fresh_points)
 
fp, cfg = config_fingerprint(R_G)
print(f"[config] fingerprint {fp}")
print(f"[config] dt={Δt}  n_loop={args.n_loop}  n_artificial={args.n_artificial}"
      f"  ntrain={n_train(R_G)}  dt*F_x={float(Δt/ρ*FBODY[0]):.4f}"
      f"  sqrt(eta*dt/rho)={onp.sqrt(η*Δt/ρ):.3f}")
if args.fixed_points:
    print("[config] fixed artificial-data locations -> K factorized once")
elif args.fresh_points:
    print("[config] fresh uniform artificial-data locations each step")
else:
    print("[config] artificial data resampled from the collocation grid each step")
 
# ---- initial condition: fluid at rest -------------------------------------
u1_prev = jnp.zeros(R_G.shape[0])
u2_prev = jnp.zeros(R_G.shape[0])
 
# ---- fit theta once against the first step --------------------------------
y = make_y(u1_prev, u2_prev)
obj = jax.jit(partial(neg_log_posterior, y=y, R_G=R_G))
 
final_params = None
if not args.fit and not args.no_cache:
    final_params = load_cache(fp)
    if final_params is not None:
        print(f"[cache] loaded theta (skipping optimization)")
 
if final_params is None:
    print(f"Initial value: {float(obj(theta_init)):.6e}", flush=True)
    t0 = time.time()
    final_params, nit = run_nelder_mead(theta_init, obj,
                                        maxiter=args.nm_iter, tol=args.tol)
    print(f"Final value:   {float(obj(final_params)):.6e}  "
          f"[{time.time()-t0:.0f}s, {nit} iters]")
    if not args.no_cache:
        save_cache(fp, final_params, obj(final_params))
 
print("\n[theta] FROZEN for the whole march" if not args.refit_every_step
      else "\n[theta] refit at every step")
report_theta(final_params)
 
# ---- march ----------------------------------------------------------------
history = []          # per-step diagnostics
Lc_cached = None

# N evenly spaced snapshots for the evolution plot, N = args.n_snap
SNAP_STEPS = sorted({max(1, int(round(k * args.n_loop / args.n_snap)))
                     for k in range(1, args.n_snap + 1)})
snapshots = []        # (t, U1, U2) captured at those steps
print(f"[config] evolution snapshots at steps {SNAP_STEPS} "
      f"(t = {', '.join(f'{s*Δt:.3f}' for s in SNAP_STEPS)})")

print(f"\n{'step':>5}{'t':>8}{'ux_max':>11}{'ux_ctr':>11}{'ux_wall':>11}"
      f"{'uy_absmax':>11}{'|du|':>11}{'std_max':>10}{'sec':>7}")

u1_test_prev = onp.zeros(NX*NY)

F1_test = None
if args.fem:
    f1, f2, fpres = fem_stokes.get(
        onp.asarray(R_test), OUTDIR / "fem_reference.npz", refit=args.fem_refit,
        L=float(L), a=float(a), avg_width=float(avg_width),
        eta=float(η), body_force_x=float(FBODY[0]),
        nx=args.fem_nx, ny=args.fem_ny)
    F1_test = onp.asarray(f1).reshape(NX, NY)
    fem_good = onp.isfinite(F1_test)
    print(f"[fem] steady reference: u_x max {F1_test[fem_good].max():.5f}, "
          f"centreline {F1_test[:, NY//2][onp.isfinite(F1_test[:, NY//2])].mean():.5f}")
    
for n in range(1, args.n_loop + 1):
    t_wall = time.time()

    # 1) observation vector for this step
    y = make_y(u1_prev, u2_prev)

    # 2) optionally refit theta (paper warm-starts from the previous optimum)
    if args.refit_every_step and n > 1:
        obj_n = jax.jit(partial(neg_log_posterior, y=y, R_G=R_G))
        final_params, _ = run_nelder_mead(final_params, obj_n,
                                          maxiter=args.nm_iter, tol=args.tol)
        print(onp.asarray(final_params))
    # 3) factorize. Reused only when the artificial-data locations are frozen.
    if args.fixed_points and Lc_cached is not None:
        Lc = Lc_cached
    else:
        Lc = factorize(final_params, R_G)
        if args.fixed_points:
            Lc_cached = Lc

    # 4) posterior on the test grid (for plots / diagnostics)
    u1_mean, u1_std = predict(Lc, final_params, y, R_test, R_G, "u1")
    u2_mean, u2_std = predict(Lc, final_params, y, R_test, R_G, "u2")
    p_mean,  p_std  = predict(Lc, final_params, y, R_test,  R_G, "p")

    # 5) draw the NEXT artificial-data locations, then evaluate the posterior
    key, kn = jrd.split(key)
    R_G_next = (R_G if args.fixed_points
                else sample_artificial(args.n_artificial, kn, fresh=args.fresh_points))

    u1_next, _ = predict(Lc, final_params, y, R_G_next, R_G, "u1")
    u2_next, _ = predict(Lc, final_params, y, R_G_next, R_G, "u2")

    # ---- diagnostics
    U1 = onp.asarray(u1_mean).reshape(NX, NY)
    U2 = onp.asarray(u2_mean).reshape(NX, NY)
    du = float(onp.abs(onp.asarray(u1_mean) - u1_test_prev).max())
    u1_test_prev = onp.asarray(u1_mean)
    t_now = n * Δt

    print(f"{n:>5}{t_now:>8.3f}{U1.max():>11.5f}{U1[:, NY//2].mean():>11.5f}"
          f"{U1[:, 0].mean():>11.2e}{onp.abs(U2).max():>11.5f}{du:>11.2e}"
          f"{float(u1_std.max()):>10.2e}{time.time()-t_wall:>7.1f}", flush=True)

    history.append(dict(step=n, t=t_now,
                        ux_max=U1.max(), ux_ctr=float(U1[:, NY//2].mean()),
                        ux_wall=float(U1[:, 0].mean()),
                        uy_absmax=float(onp.abs(U2).max()), dmax=du,
                        std_max=float(u1_std.max())))

    # capture snapshot for the evolution figure -- THIS is the line that was missing
    if n in SNAP_STEPS:
        snapshots.append((n, t_now, U1.copy(), U2.copy()))
        if F1_test is not None:
            good = fem_good & onp.isfinite(U1)
            rel = onp.linalg.norm(U1[good] - F1_test[good]) / onp.linalg.norm(F1_test[good])
            print(f"      [fem] t={t_now:.3f}  rel L2 vs steady = {rel:.4f}")
            history[-1]["fem_rel_l2"] = rel
    last = (n == args.n_loop)
    if last or (args.plot_every and n % args.plot_every == 0):
        ctx = plotting.PlotCtx(
            XX=onp.asarray(XX), YY=onp.asarray(YY), NX=NX, NY=NY,
            L=float(L), a=float(a), avg_width=float(avg_width),
            R_wall=onp.asarray(R_wall), outdir=OUTDIR,
            tag=f"{fp}_t{n:04d}", subdir="base",
            sparse_pts=onp.asarray(R_G))
        fig_n = plotting.plot_all(ctx, {"u_x": (u1_mean, u1_std),
                                        "u_y": (u2_mean, u2_std),
                                        "p":   (p_mean,  p_std)}, fem=None)
        plt.close(fig_n)      # plot_all returns the combined fig still open;
                              # over a long march these accumulate.

    onp.savez(OUTDIR / f"state_t{n:04d}.npz",
              t=t_now, theta=onp.asarray(final_params),
              u1=U1, u2=U2, p=onp.asarray(p_mean).reshape(NX, NY),
              u1_std=onp.asarray(u1_std).reshape(NX, NY),   # fixed typo: was u1_stdtd
              R_G=onp.asarray(R_G),
              u1_artificial=onp.asarray(u1_next),
              u2_artificial=onp.asarray(u2_next))

    # 6) hand off to the next step
    u1_prev, u2_prev, R_G = u1_next, u2_next, R_G_next


# ==================================================================== summary

# if args.evolution:
#     evolution_plot.plot_evolution(
#         snapshots, OUTDIR / f"evolution_{fp}.png",
#         XX=onp.asarray(XX), YY=onp.asarray(YY),
#         L=float(L), a=float(a), avg_width=float(avg_width),
#         cmap=args.evolution_cmap,
#         share_rows=args.evolution_share_rows,
#         title=(rf"velocity evolution, $\Delta t={Δt}$, backward Euler, "
#                rf"{args.n_artificial} artificial points/step"))
#     plt.close("all")
# else:
#     print("[evo] skipped (pass --evolution to write it, or run "
#           "evolution_plot.py against the saved states)")

t_hist  = onp.array([h["t"] for h in history])
ctr     = onp.array([h["ux_ctr"] for h in history])
mx      = onp.array([h["ux_max"] for h in history])
dmax    = onp.array([h["dmax"] for h in history])
wall    = onp.array([h["ux_wall"] for h in history])

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(t_hist, ctr, "o-", label="centreline mean")
ax[0].plot(t_hist, mx,  "s-", ms=3, label="domain max")
ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$u_x$")
ax[0].set_title("spin-up from rest"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

ax[1].semilogy(t_hist, onp.maximum(dmax, 1e-16), "o-")
ax[1].set_xlabel("$t$"); ax[1].set_ylabel(r"$\max|u_x^n-u_x^{n-1}|$")
ax[1].set_title("step-to-step change (steady when flat)"); ax[1].grid(alpha=.3)

ax[2].semilogy(t_hist, onp.abs(wall) + 1e-16, "o-")
ax[2].set_xlabel("$t$"); ax[2].set_ylabel(r"$|u_x|$ at wall row")
ax[2].set_title("no-slip residual"); ax[2].grid(alpha=.3)

fig.tight_layout()
fig.savefig(OUTDIR / f"march_summary_{fp}.png", dpi=140, bbox_inches="tight")
print(f"\n[plot] {OUTDIR / f'march_summary_{fp}.png'}")

onp.savez(OUTDIR / f"march_history_{fp}.npz",
          **{k: onp.array([h[k] for h in history]) for k in history[0]})

fem_results = None
if args.fem:
    fem_results = fem_stokes.get_unsteady(
        onp.asarray(R_test), SNAP_STEPS,
        OUTDIR / "fem_unsteady_reference.npz", refit=args.fem_refit,
        L=float(L), a=float(a), avg_width=float(avg_width),
        eta=float(η), rho=float(ρ), body_force_x=float(FBODY[0]),
        dt=float(Δt), nx=args.fem_nx, ny=args.fem_ny)

evolution_plot.plot_evolution_vs_fem(
    snapshots, OUTDIR, XX=onp.asarray(XX), YY=onp.asarray(YY),
    L=float(L), a=float(a), avg_width=float(avg_width),
    fem=fem_results, cmap=args.evolution_cmap, tag=fp, dt=float(Δt),
    n_artificial=args.n_artificial)

plt.show()