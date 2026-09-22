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

plt.rcParams.update({
    "font.size":         14,   # was 26
    "axes.labelsize":    16,   # was 28
    "axes.titlesize":    15,   # was 26
    "xtick.labelsize":   13,   # was 22
    "ytick.labelsize":   13,   # was 22
    "legend.fontsize":   13,   # was 22
    "axes.linewidth":    1.2,  # was 1.8
    "lines.linewidth":   1.8,  # was 2.6
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size":  5,   "ytick.major.size":  5,
    "savefig.pad_inches": 0.02,
})

parser = argparse.ArgumentParser(description="2D Stokes flow PIGP (18 hyperparameters)")

parser.add_argument("--profiles", action="store_true",
                    help="write wall-normal y-vs-velocity profile plots")
parser.add_argument("--profile-x-ux", type=float, nargs=2,
                    default=[0.625, 1.875],
                    help="x locations for the u_x profiles (default L/4, 3L/4)")
parser.add_argument("--profile-x-uy", type=float, nargs=2,
                    default=[0.9375, 1.5625],
                    help="x locations for the u_y profiles (default 3L/8, 5L/8)")
parser.add_argument("--steady-tol", type=float, default=1e-3,
                    help="max|du_x|/max|u_x| below which flow is called steady")
parser.add_argument("--steady-consecutive", type=int, default=2,
                    help="consecutive steps that must satisfy --steady-tol")
parser.add_argument("--profile-n-times", type=int, default=3,
                    help="how many of the --n-snap snapshots to draw in the "
                         "profile figure (first/middle/last)")
# parser.add_argument("--fatol", type=float, default=0.1,
#                     help="absolute objective tolerance (scipy fatol). "
#                          "Objective is O(1e3) here, so ~0.1 is ~1e-4 relative.")
# parser.add_argument("--xatol", type=float, default=0.05,
#                     help="absolute parameter tolerance (scipy xatol). "
#                          "Kept looser than fatol on purpose: hyperparameters "
#                          "commonly have flat/degenerate directions (e.g. "
#                          "length-scale vs. amplitude trade-off) where x keeps "
#                          "drifting slightly after the objective has converged.")

parser.add_argument("--fem", action="store_true",
                    help="solve the same problem with dolfinx and compare")
parser.add_argument("--fem-refit", action="store_true",
                    help="force re-solving the FEM even if a valid cache exists")
parser.add_argument("--fem-nx", type=int, default=200)
parser.add_argument("--fem-ny", type=int, default=80)
parser.add_argument("--fit", action="store_true", help="force re-optimization")
parser.add_argument("--no-cache", action="store_true", help="do not read or write cache")
parser.add_argument("--nm-iter", type=int, default=500)
parser.add_argument("--tol", type=float, default=1e-2)
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

SPECIMEN = (
    f"patch_usf_{args.dt}_{args.n_loop}_{args.n_artificial}artificial_"
    f"{args.geometry}_nm_{args.nm_iter}_tol_{args.tol}_"
    f"fresh_points_{args.fresh_points}_"
    f"refit_everystep_{args.refit_every_step}"
)
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
ρ = 1.0 # Density rho
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

# --- [PATCH 1] block offsets, needed to index the G1/G2 rows of K^-1 Q^T ---
def block_slices(R_G):
    """Start/end offsets of each named block in the K / y ordering."""
    offsets = {}
    start = 0
    for name, R in make_blocks(R_G):
        offsets[name] = (start, start + R.shape[0])
        start += R.shape[0]
    return offsets
# --- [END PATCH 1] ----------------------------------------------------------

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

def run_nelder_mead(init_params, fun, maxiter=500, tol=1e-1):
    iter_count = [0]
    
    def callback(xk):
        iter_count[0] += 1
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

def find_steady_state(history, rel_tol=1e-2, n_consecutive=3,
                      plateau_window=6, plateau_ratio=1.4):
    """Steady when EITHER
         (a) rel change < rel_tol for n_consecutive steps, or
         (b) rel change stops decaying: the median over the last
             `plateau_window` steps is no better than `plateau_ratio`x
             the median over the preceding window of the same length.
    (b) exists because resampling the artificial points each step puts a
    noise floor under the rel change, so (a) alone can never fire if
    rel_tol sits below that floor."""
    rel = []
    for h in history:
        scale = abs(h["ux_max"])
        r = onp.nan if scale < 1e-14 else h["dmax"] / scale
        h["rel_change"] = r
        rel.append(r)
    rel = onp.asarray(rel, dtype=float)

    hits = 0
    for i, r in enumerate(rel):
        if onp.isfinite(r) and r < rel_tol:
            hits += 1
            if hits >= n_consecutive:
                return history[i]["t"], history[i]["step"], "tolerance"
        else:
            hits = 0

    w = plateau_window
    for i in range(2*w, len(rel)):
        recent = onp.nanmedian(rel[i-w+1:i+1])
        older  = onp.nanmedian(rel[i-2*w+1:i-w+1])
        if onp.isfinite(recent) and onp.isfinite(older) and older < plateau_ratio*recent:
            return history[i]["t"], history[i]["step"], "plateau"

    return None, None, None

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
        "uncertainty_propagation": True,   # [PATCH] fingerprint bump: Sigma^{t-1,t-1} now marginalized
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

# --- [PATCH 2] joint (u1,u2) prior-covariance builder, needed to seed the
# next step's Sigma^{t-1,t-1} block (full matrix, not just the diagonal) ---
def build_Kss_joint(theta, R_star):
    """Prior covariance of the joint (u1,u2) field at R_star -> (2n, 2n)."""
    K11 = kmat(k_uu[0][0])(R_star, R_star, theta)
    K12 = kmat(k_uu[0][1])(R_star, R_star, theta)
    K21 = kmat(k_uu[1][0])(R_star, R_star, theta)
    K22 = kmat(k_uu[1][1])(R_star, R_star, theta)
    return jnp.block([[K11, K12], [K21, K22]])
# --- [END PATCH 2] ----------------------------------------------------------

def build_K_star_obs(theta, R_star, star, R):
    """Q, shape (len(R_star), ntrain). Columns follow `blocks` order."""
    cols = [kmat(k)(R_star, R, theta)
            for k, (_, R) in zip(ROWS[star], make_blocks(R))]
    return jnp.concatenate(cols, axis=1)

def prior_var_diag(theta, R_star, star):
    """diag(K**) only - the full n_star x n_star matrix is never needed."""
    k = KSTAR[star]
    return jax.vmap(lambda r: k(r, r, theta))(R_star)

# --- [PATCH 3] predict -> predict_diag / predict_joint ---------------------
# Implements Sigma^{t,t} = K** - Q K^-1 Q^T + Q K^-1 M K^-1 Q^T
# (eq. in sec. "Uncertainty propagation"), where M is zero except in the
# (G1,G2)x(G1,G2) block, where it equals Sigma^{t-1,t-1}.

@partial(jax.jit, static_argnames=("star",))
def predict_diag(Lc, theta, y, R_star, R_G, star, M_block, idx_G):
    """Marginal mean/std at R_star, including the propagated-uncertainty
    correction  diag(Q K^-1 M K^-1 Q^T)."""
    Q  = build_K_star_obs(theta, R_star, star, R_G)
    alpha = solve_triangular(Lc.T, solve_triangular(Lc, y, lower=True), lower=False)
    mean  = Q @ alpha                                    # (n_star,)

    V   = solve_triangular(Lc, Q.T, lower=True)           # K^-1/2 Q^T, (ntrain, n_star)
    var = prior_var_diag(theta, R_star, star) - jnp.sum(V**2, axis=0)

    W = solve_triangular(Lc.T, V, lower=False)             # K^-1 Q^T, (ntrain, n_star)
    C = W[idx_G, :]                                        # (2n_G, n_star)
    var = var + jnp.sum(C * (M_block @ C), axis=0)

    std = jnp.sqrt(jnp.clip(var, 0.0))
    return mean, std


@jax.jit
def predict_joint(Lc, theta, y, R_star, R_G, M_block, idx_G):
    """Joint (u1,u2) mean and FULL covariance at R_star. Used to seed the
    NEXT step's Sigma^{t-1,t-1} (fed back in as M_block)."""
    n = R_star.shape[0]
    Q1 = build_K_star_obs(theta, R_star, "u1", R_G)
    Q2 = build_K_star_obs(theta, R_star, "u2", R_G)
    Q  = jnp.concatenate([Q1, Q2], axis=0)                 # (2n, ntrain)

    alpha = solve_triangular(Lc.T, solve_triangular(Lc, y, lower=True), lower=False)
    mean  = Q @ alpha

    Kss = build_Kss_joint(theta, R_star)
    V   = solve_triangular(Lc, Q.T, lower=True)
    cov = Kss - V.T @ V

    W = solve_triangular(Lc.T, V, lower=False)
    C = W[idx_G, :]                                        # (2n_G, 2n)
    cov = cov + C.T @ (M_block @ C)

    cov = 0.5 * (cov + cov.T)                              # symmetrize
    return mean[:n], mean[n:], cov
# --- [END PATCH 3] -----------------------------------------------------------

#### ==================================================================== time loop
 
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

# --- [PATCH 4] Sigma^{0,0} = 0 (rest state is exactly known) ---------------
_offs = block_slices(R_G)
idx_G = jnp.concatenate([jnp.arange(*_offs["G1"]), jnp.arange(*_offs["G2"])])
Sigma_prev = jnp.zeros((idx_G.shape[0], idx_G.shape[0]))
# --- [END PATCH 4] -----------------------------------------------------------
 
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
                                        maxiter=args.nm_iter,  
                                        tol = args.tol,
                                        )
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
      f"{'uy_absmax':>11}{'|du|':>11}"
      f"{'2sig_ux':>10}{'cov_ux':>9}{'2sig_uy':>10}{'cov_uy':>9}{'sec':>7}")

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
# fem_steps = sorted(set(SNAP_STEPS) | set(range(5, args.n_loop + 1, 5)))

fem_march = None
if args.fem:
    fem_steps_all = list(range(1, args.n_loop + 1))
    fem_march = fem_stokes.get_unsteady(
        onp.asarray(R_test), fem_steps_all,
        OUTDIR / "fem_unsteady_reference.npz", refit=args.fem_refit,
        L=float(L), a=float(a), avg_width=float(avg_width),
        eta=float(η), rho=float(ρ), body_force_x=float(FBODY[0]),
        dt=float(Δt), nx=args.fem_nx, ny=args.fem_ny)

    if args.n_loop >= 10:
        print("\n" + "=" * 66)
        print("FEM STEADY-STATE CHECK (every 5th step)")
        print("=" * 66)
        steps5 = sorted(s for s in fem_march if s % 5 == 0)
        fem_t_steady = None
        for t1s, t2s in zip(steps5, steps5[1:]):
            steady, rel_ux, rel_uy = fem_stokes.fem_steady_check(
                fem_march, t1s, t2s, Δt, rel_tol=args.steady_tol)
            if steady and fem_t_steady is None:
                fem_t_steady = t2s * Δt
        if fem_t_steady is not None:
            print(f"--> FEM reaches steady state by t = {fem_t_steady:.4f}")
        else:
            print(f"--> FEM not steady within t <= {steps5[-1]*Δt:.4f}")
        print("=" * 66)
    else:
        print(f"[fem-steady] skipped: n_loop={args.n_loop} < 10 "
             f"(need at least two 5-step-spaced samples)")
    
for n in range(1, args.n_loop + 1):
    t_wall = time.time()

    # 1) observation vector for this step
    y = make_y(u1_prev, u2_prev)

    # 2) optionally refit theta (paper warm-starts from the previous optimum)
    if args.refit_every_step and n > 1:
        obj_n = jax.jit(partial(neg_log_posterior, y=y, R_G=R_G))
        final_params, _ = run_nelder_mead(final_params, obj_n,
                                          maxiter=args.nm_iter,  
                                          tol = args.tol,
                                          )
        print(onp.asarray(final_params))
    # 3) factorize. Reused only when the artificial-data locations are frozen.
    if args.fixed_points and Lc_cached is not None:
        Lc = Lc_cached
    else:
        Lc = factorize(final_params, R_G)
        if args.fixed_points:
            Lc_cached = Lc

    # 4) posterior on the test grid (for plots / diagnostics), including
    #    the propagated-uncertainty correction from Sigma_prev / M
    u1_mean, u1_std = predict_diag(Lc, final_params, y, R_test, R_G, "u1", Sigma_prev, idx_G)
    u2_mean, u2_std = predict_diag(Lc, final_params, y, R_test, R_G, "u2", Sigma_prev, idx_G)
    p_mean,  p_std  = predict_diag(Lc, final_params, y, R_test, R_G, "p",  Sigma_prev, idx_G)

    # 5) draw the NEXT artificial-data locations, then evaluate the JOINT
    #    posterior there (mean + full covariance) -> becomes next Sigma_prev
    key, kn = jrd.split(key)
    R_G_next = (R_G if args.fixed_points
                else sample_artificial(args.n_artificial, kn, fresh=args.fresh_points))

    u1_next, u2_next, Sigma_next = predict_joint(
        Lc, final_params, y, R_G_next, R_G, Sigma_prev, idx_G)

    # ---- diagnostics
    U1 = onp.asarray(u1_mean).reshape(NX, NY)
    U2 = onp.asarray(u2_mean).reshape(NX, NY)
    S1 = onp.asarray(u1_std).reshape(NX, NY)
    S2 = onp.asarray(u2_std).reshape(NX, NY)
    du = float(onp.abs(onp.asarray(u1_mean) - u1_test_prev).max())
    u1_test_prev = onp.asarray(u1_mean)
    t_now = n * Δt

    def _uncertainty_stats(S, comp_name):
        i_max = int(onp.argmax(S))
        x_max, y_max = (float(v) for v in onp.asarray(R_test)[i_max])
        return {
            f"twosig_max_{comp_name}":  float(2.0 * S.ravel()[i_max]),
            f"twosig_mean_{comp_name}": float(2.0 * S.mean()),
            f"twosig_x_{comp_name}":    x_max,
            f"twosig_y_{comp_name}":    y_max,
        }

    diag = dict(step=n, t=t_now,
               ux_max=U1.max(), ux_ctr=float(U1[:, NY//2].mean()),
               ux_wall=float(U1[:, 0].mean()),
               uy_absmax=float(onp.abs(U2).max()), dmax=du,
               std_max=float(S1.max()))
    diag.update(_uncertainty_stats(S1, "ux"))
    diag.update(_uncertainty_stats(S2, "uy"))

    # ---- coverage: fraction of points where the TRUE (FEM) error at this
    # same step falls within the model's own 95% band -- separately for
    # u_x and u_y, since they can be calibrated very differently (u_y is
    # the small, secondary component, so its relative error -- and
    # plausibly its calibration -- behaves differently from u_x).
    cov_ux_str = cov_uy_str = "     n/a"
    if fem_march is not None and n in fem_march:
        f1n, f2n, _ = fem_march[n]
        F1n = onp.asarray(f1n).reshape(NX, NY)
        F2n = onp.asarray(f2n).reshape(NX, NY)

        good_x = fem_good & onp.isfinite(U1) & onp.isfinite(F1n)
        abs_err_x = onp.abs(U1 - F1n)
        cov_x = float((abs_err_x[good_x] <= (2.0*S1)[good_x]).mean())
        diag["coverage_95_ux"]     = cov_x
        diag["mean_abs_err_ux"]    = float(abs_err_x[good_x].mean())
        cov_ux_str = f"{100*cov_x:7.1f}%"

        good_y = fem_good & onp.isfinite(U2) & onp.isfinite(F2n)
        abs_err_y = onp.abs(U2 - F2n)
        cov_y = float((abs_err_y[good_y] <= (2.0*S2)[good_y]).mean())
        diag["coverage_95_uy"]     = cov_y
        diag["mean_abs_err_uy"]    = float(abs_err_y[good_y].mean())
        cov_uy_str = f"{100*cov_y:7.1f}%"

    print(f"{n:>5}{t_now:>8.3f}{U1.max():>11.5f}{U1[:, NY//2].mean():>11.5f}"
          f"{U1[:, 0].mean():>11.2e}{onp.abs(U2).max():>11.5f}{du:>11.2e}"
          f"{diag['twosig_max_ux']:>10.2e}{cov_ux_str:>9}"
          f"{diag['twosig_max_uy']:>10.2e}{cov_uy_str:>9}"
          f"{time.time()-t_wall:>7.1f}", flush=True)

    history.append(diag)

    # capture snapshot for the evolution figure -- THIS is the line that was missing
    if n in SNAP_STEPS:
        snapshots.append((n, t_now, U1.copy(), U2.copy(), S1.copy(), S2.copy()))
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
              u2_artificial=onp.asarray(u2_next),
              Sigma_diag_artificial=onp.asarray(jnp.diag(Sigma_next)))

    # 6) hand off to the next step (Sigma_next carries the propagated
    #    uncertainty forward, replacing the old "exact-value" assumption)
    u1_prev, u2_prev, R_G, Sigma_prev = u1_next, u2_next, R_G_next, Sigma_next


# ==================================================================== summary

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

# ==================== (a) COVERAGE / CALIBRATION, ux & uy ====================
def _plot_calibration(comp_name, symbol):
    if not any(f"coverage_95_{comp_name}" in h for h in history):
        print(f"[unc] no FEM per-step data for {comp_name} -- skipped (run with --fem)")
        return
    t_cov = onp.array([h["t"] for h in history if f"coverage_95_{comp_name}" in h])
    cov   = onp.array([h[f"coverage_95_{comp_name}"] for h in history
                       if f"coverage_95_{comp_name}" in h])
    twosig_max_h  = onp.array([h[f"twosig_max_{comp_name}"]  for h in history])
    twosig_mean_h = onp.array([h[f"twosig_mean_{comp_name}"] for h in history])

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

    ax[0].semilogy(t_hist, twosig_max_h,  "o-", label=rf"max $2\sigma$ ({symbol})")
    ax[0].semilogy(t_hist, twosig_mean_h, "s-", ms=3, label=rf"mean $2\sigma$ ({symbol})")
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel(rf"$2\sigma({symbol[1:-1]})$")
    ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(t_cov, 100*cov, "o-", color="tab:green")
    ax[1].axhline(95.0, color="r", ls="--", lw=1.5, label="nominal 95%")
    ax[1].set_ylim(0, 102)
    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel(f"points within $2\\sigma$ of FEM  [%]  ({symbol})")
    ax[1].legend(); ax[1].grid(alpha=.3)

    fig.subplots_adjust(wspace=0.30)
    out = OUTDIR / f"uncertainty_calibration_{comp_name}_{fp}.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] {out}")

_plot_calibration("ux", r"$u_x$")
_plot_calibration("uy", r"$u_y$")
# ===============================================================================

onp.savez(OUTDIR / f"march_history_{fp}.npz",
          **{k: onp.array([h[k] for h in history]) for k in history[0]})

evolution_plot.plot_evolution_vs_fem(
    snapshots, OUTDIR, XX=onp.asarray(XX), YY=onp.asarray(YY),
    L=float(L), a=float(a), avg_width=float(avg_width),
    fem=fem_march, cmap=args.evolution_cmap, tag=fp, dt=float(Δt),
    n_artificial=args.n_artificial)

evolution_plot.plot_uncertainty_evolution(
    snapshots, OUTDIR, XX=onp.asarray(XX), YY=onp.asarray(YY),
    L=float(L), a=float(a), avg_width=float(avg_width),
    fem=fem_march, tag=f"ux_{fp}", comp=0, symbol=r"$u_x$")

evolution_plot.plot_uncertainty_evolution(
    snapshots, OUTDIR, XX=onp.asarray(XX), YY=onp.asarray(YY),
    L=float(L), a=float(a), avg_width=float(avg_width),
    fem=fem_march, tag=f"uy_{fp}", comp=1, symbol=r"$u_y$")

if args.profiles:
    evolution_plot.plot_profiles(
        snapshots, OUTDIR,
        XX=onp.asarray(XX), YY=onp.asarray(YY),
        L=float(L), a=float(a), avg_width=float(avg_width),
        x_ux=tuple(args.profile_x_ux), x_uy=tuple(args.profile_x_uy),
        fem=fem_march, tag=fp,
        n_times=args.profile_n_times)  

t_steady, n_steady, reason = find_steady_state(history,
                                       rel_tol=args.steady_tol,
                                       n_consecutive=args.steady_consecutive)

print("\n" + "=" * 66)
print("STEADY-STATE ASSESSMENT")
print("=" * 66)
print(f"  criterion   : max|u_x^n - u_x^(n-1)| / max|u_x^n| < {args.steady_tol:g}")
print(f"                sustained for {args.steady_consecutive} consecutive steps")
print(f"  dt = {Δt:g}   steps = {args.n_loop}   t_final = {args.n_loop*Δt:.4f}")
print("-" * 66)
print(f"  {'step':>5}{'t':>9}{'ux_max':>12}{'rel change':>14}")
for h in history:
    rc = h.get("rel_change", onp.nan)
    mark = ""
    if n_steady is not None and h["step"] == n_steady:
        mark = "   <-- steady"
    print(f"  {h['step']:>5}{h['t']:>9.4f}{h['ux_max']:>12.6f}"
          f"{rc:>14.3e}{mark}")
print("-" * 66)
if t_steady is not None:
    hs = history[n_steady - 1]
    label = ("relative change fell below tolerance" if reason == "tolerance"
             else "relative change stopped decaying (hit the noise floor)")
    print(f"  --> STEADY STATE REACHED at t = {t_steady:.4f}  (step {n_steady})")
    print(f"      criterion fired    : {label}")
    print(f"      u_x max            : {hs['ux_max']:.6f}")
    print(f"      u_x centreline mean: {hs['ux_ctr']:.6f}")
    print(f"      |u_x| at wall row  : {abs(hs['ux_wall']):.3e}")
else:
    hf = history[-1]
    rel_f = hf["dmax"] / max(abs(hf["ux_max"]), 1e-14)
    print(f"  --> NOT STEADY within t <= {hf['t']:.4f}")
    print(f"      final relative change: {rel_f:.3e}  (tol {args.steady_tol:g})")
    print(f"      increase --n-loop, or relax --steady-tol")
print("=" * 66)

onp.savez(OUTDIR / f"steady_state_{fp}.npz",
          t_steady=onp.asarray(onp.nan if t_steady is None else t_steady),
          step_steady=onp.asarray(-1 if n_steady is None else n_steady),
          rel_tol=onp.asarray(args.steady_tol),
          reason=onp.asarray("none" if reason is None else reason),
          t=onp.array([h["t"] for h in history]),
          rel_change=onp.array([h.get("rel_change", onp.nan) for h in history]))
# ==================================================================== report
def write_report(history, fp, args, t_steady, n_steady, reason, outdir):
    """Single human-readable .txt + machine-readable .csv summarizing the
    whole march: per-step errors/uncertainty, calibration, steady-state."""
    import csv

    txt_path = outdir / f"report_{fp}.txt"
    csv_path = outdir / f"report_{fp}.csv"

    # --- CSV: one row per step, union of all keys that ever appeared -------
    all_keys = []
    seen = set()
    for h in history:
        for k in h:
            if k not in seen:
                seen.add(k); all_keys.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, restval="", extrasaction="ignore")
        w.writeheader()
        for h in history:
            w.writerow(h)
    print(f"[report] wrote {csv_path}")

    # --- TXT: readable report ------------------------------------------------
    lines = []
    lines.append("=" * 78)
    lines.append("PIGP UNSTEADY STOKES FLOW -- RUN REPORT")
    lines.append("=" * 78)
    lines.append(f"fingerprint       : {fp}")
    lines.append(f"geometry          : {args.geometry}")
    lines.append(f"dt / n_loop       : {args.dt} / {args.n_loop}  (t_final = {args.dt*args.n_loop:.4f})")
    lines.append(f"n_artificial      : {args.n_artificial}  fresh_points={args.fresh_points}  fixed_points={args.fixed_points}")
    lines.append(f"nm_iter / tol     : {args.nm_iter} / {args.tol}")
    lines.append(f"refit_every_step  : {args.refit_every_step}")
    lines.append(f"fem comparison    : {args.fem}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("PER-STEP DIAGNOSTICS")
    lines.append("-" * 78)
    has_fem = any("coverage_95_ux" in h for h in history)
    if has_fem:
        header = (f"{'step':>5}{'t':>9}{'ux_max':>11}{'|du|':>11}{'rel_chg':>11}"
                   f"{'2sig_ux':>11}{'cov_ux':>9}{'mae_ux':>11}"
                   f"{'2sig_uy':>11}{'cov_uy':>9}{'mae_uy':>11}{'fem_L2':>9}")
    else:
        header = (f"{'step':>5}{'t':>9}{'ux_max':>11}{'|du|':>11}{'rel_chg':>11}"
                   f"{'2sig_ux':>11}{'2sig_uy':>11}")
    lines.append(header)
    for h in history:
        rc = h.get("rel_change", float("nan"))
        row = f"{h['step']:>5}{h['t']:>9.4f}{h['ux_max']:>11.5f}{h['dmax']:>11.2e}{rc:>11.2e}"
        row += f"{h['twosig_max_ux']:>11.2e}"
        if has_fem:
            cov_x = h.get("coverage_95_ux")
            mae_x = h.get("mean_abs_err_ux")
            row += f"{(100*cov_x if cov_x is not None else float('nan')):>8.1f}%"
            row += f"{(mae_x if mae_x is not None else float('nan')):>11.2e}"
        row += f"{h['twosig_max_uy']:>11.2e}"
        if has_fem:
            cov_y = h.get("coverage_95_uy")
            mae_y = h.get("mean_abs_err_uy")
            row += f"{(100*cov_y if cov_y is not None else float('nan')):>8.1f}%"
            row += f"{(mae_y if mae_y is not None else float('nan')):>11.2e}"
            fl2 = h.get("fem_rel_l2")
            row += f"{(fl2 if fl2 is not None else float('nan')):>9.4f}" if fl2 is not None else f"{'':>9}"
        lines.append(row)
    lines.append("")

    if has_fem:
        lines.append("-" * 78)
        lines.append("CALIBRATION SUMMARY (final step)")
        lines.append("-" * 78)
        h_last_fem = [h for h in history if "coverage_95_ux" in h]
        if h_last_fem:
            hl = h_last_fem[-1]
            lines.append(f"  t = {hl['t']:.4f}")
            lines.append(f"  u_x : coverage_95 = {100*hl['coverage_95_ux']:.1f}%   "
                          f"mean|err| = {hl['mean_abs_err_ux']:.3e}   "
                          f"max 2sigma = {hl['twosig_max_ux']:.3e}")
            lines.append(f"  u_y : coverage_95 = {100*hl['coverage_95_uy']:.1f}%   "
                          f"mean|err| = {hl['mean_abs_err_uy']:.3e}   "
                          f"max 2sigma = {hl['twosig_max_uy']:.3e}")
        lines.append("")

    lines.append("-" * 78)
    lines.append("STEADY-STATE ASSESSMENT")
    lines.append("-" * 78)
    lines.append(f"  criterion : max|du_x|/max|u_x| < {args.steady_tol:g}, "
                  f"{args.steady_consecutive} consecutive steps")
    if t_steady is not None:
        hs = history[n_steady - 1]
        label = ("tolerance" if reason == "tolerance" else "plateau (noise floor)")
        lines.append(f"  RESULT    : reached at t = {t_steady:.4f} (step {n_steady})  [{label}]")
        lines.append(f"  ux_max    : {hs['ux_max']:.6f}")
        lines.append(f"  ux_ctr    : {hs['ux_ctr']:.6f}")
        lines.append(f"  |ux_wall| : {abs(hs['ux_wall']):.3e}")
    else:
        hf = history[-1]
        rel_f = hf["dmax"] / max(abs(hf["ux_max"]), 1e-14)
        lines.append(f"  RESULT    : NOT steady within t <= {hf['t']:.4f}")
        lines.append(f"  final rel change: {rel_f:.3e}  (tol {args.steady_tol:g})")
    lines.append("=" * 78)

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {txt_path}")

write_report(history, fp, args, t_steady, n_steady, reason, OUTDIR)
plt.show()