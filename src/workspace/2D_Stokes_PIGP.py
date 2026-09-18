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
from pathlib import Path

import plotting
import fem_stokes

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

parser.add_argument("--sparse", action="store_true",
                    help="replace the no-slip wall points with N random interior "
                         "velocity points sampled from the FEM solution (paper 3.3)")
parser.add_argument("--n-sparse", type=int, default=40)
parser.add_argument("--sparse-seed", type=int, default=42)
parser.add_argument("--sparse-margin", type=float, default=0.95,
                    help="keep samples inside this fraction of the local half-width")

args = parser.parse_args()

OUTDIR = Path(__file__).resolve().parent / "outputs_2D"
for _sub in ("base", "sparse"):
    (OUTDIR / "plots" / _sub).mkdir(parents=True, exist_ok=True)
CACHE_PATH = OUTDIR / ("final_params_sparse.npz" if args.sparse else "final_params.npz")

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
Q = 1.0 # Flow rate
L = 2.5 # Characteristic Length
a = 0.2 # Dimensionless value
avg_width = 1 # Average width
ΔP = -30 # Pressure Diff
FBODY = -ΔP/L * jnp.array([1.0, 0.0]) # Body force
MARGIN = 0.999

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


def sample_sparse_points(n, key, margin):
    """Uniform in x over [0,L), uniform in the width-scaled transverse
    coordinate. `margin` < 1 keeps points off the wall so the FEM
    interpolator does not fall outside the mesh."""
    kx, ke = jrd.split(key)
    x = jrd.uniform(kx, (n,), minval=0.0, maxval=L)
    e = jrd.uniform(ke, (n,), minval=-0.5, maxval=0.5) * margin
    return jnp.stack([x, e * width(x)], axis=-1)

if args.sparse:
    R_sparse = sample_sparse_points(args.n_sparse,
                                    jrd.PRNGKey(args.sparse_seed),
                                    args.sparse_margin)
    _cache = OUTDIR / f"fem_sparse_n{args.n_sparse}_s{args.sparse_seed}.npz"
    _u1, _u2, _ = fem_stokes.get(
        onp.asarray(R_sparse), _cache, refit=args.fem_refit,
        L=float(L), a=float(a), avg_width=float(avg_width),
        eta=float(η), body_force_x=float(FBODY[0]),
        nx=args.fem_nx, ny=args.fem_ny)
    u1_train, u2_train = jnp.asarray(_u1), jnp.asarray(_u2)
    if not bool(jnp.all(jnp.isfinite(u1_train) & jnp.isfinite(u2_train))):
        raise RuntimeError("FEM gave non-finite velocities at the sparse points; "
                           "lower --sparse-margin or refine --fem-nx/--fem-ny")
    R_u_train = R_sparse
    print(f"[sparse] {args.n_sparse} FEM velocity points (seed {args.sparse_seed}), "
          f"wall no-slip points dropped; "
          f"u_x in [{float(u1_train.min()):.3f}, {float(u1_train.max()):.3f}]")
else:
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

x_f_1 = x_f
y_f_1 = y_f
R_f1 = jnp.stack([x_f_1,y_f_1], axis=-1)

f_1 = jnp.full(N_f, -ΔP / L)
x_f_2 = x_f
y_f_2 = y_f
R_f2 = jnp.stack([x_f_1,y_f_1], axis=-1)
f_2 = jnp.zeros(N_f)

## 5.) Governing equations s (337 points)

x_s, y_s = x_f, y_f 
R_s = jnp.stack([x_s,y_s], axis=-1)
s = jnp.zeros(N_s)

###

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
D    = lambda k, a: lambda r, rp, t: jax.grad(k, 0)(r, rp, t)[a]
Dp   = lambda k, a: lambda r, rp, t: jax.grad(k, 1)(r, rp, t)[a]
Lap  = lambda k:    lambda r, rp, t: jnp.trace(jax.hessian(k, 0)(r, rp, t))
Lapp = lambda k:    lambda r, rp, t: jnp.trace(jax.hessian(k, 1)(r, rp, t))
add  = lambda *ks:  lambda r, rp, t: sum(k(r, rp, t) for k in ks)
mul  = lambda c, k: lambda r, rp, t: c * k(r, rp, t)

# shift operator, appendix B:  (S_L a)(r) = a(r) - a(r + L e_x)
_eL  = jnp.array([L, 0.0])
S    = lambda k: lambda r, rp, t: k(r, rp, t) - k(r + _eL, rp, t)
Sp   = lambda k: lambda r, rp, t: k(r, rp, t) - k(r, rp + _eL, t)

### Base kernel (Total of 6 kernels and 18 parameters)

k_uu = [[lambda r, rp, t: Product_Squared_Exponential_Kernel(r,  rp, theta(t, 0)),      
         lambda r, rp, t: Product_Squared_Exponential_Kernel(r,  rp, theta(t, 1))],    
        [lambda r, rp, t: Product_Squared_Exponential_Kernel(rp, r,  theta(t, 1)),     
         lambda r, rp, t: Product_Squared_Exponential_Kernel(r,  rp, theta(t, 2))]]     

k_up = [lambda r, rp, t: Product_Squared_Exponential_Kernel(r,  rp, theta(t, 3)),
        lambda r, rp, t: Product_Squared_Exponential_Kernel(r,  rp, theta(t, 4))]
k_pu = k_up

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
k_dsp_s = S(k_ps)

k_ff = lambda a, b: add(D(k_pf(b), a),  mul(-η, Lap(k_uf(a, b))))       
k_fs = lambda a:    add(D(k_ps, a),     mul(-η, Lap(k_us(a))))       

k_ss =              add(*[D(k_us(a), a) for a in range(2)])  

## Buiding K

def kmat(k):
    return jax.jit(jax.vmap(jax.vmap(k, (None, 0, None)), (0, None, None)))

blocks = [
    ("u1", R_u_train),   
    ("u2", R_u_train),      # boundary velocity
    ("Su1", R_dSu1),  
    ("Su2", R_dSu2),     # shifted velocity (periodicity)
    ("Sp", R_sp),                      # shifted pressure
    ("f1", R_f1),    
    ("f2", R_f2),       # force-balance residual points
    ("s", R_s),                        # continuity residual points
    ]

kfn = {
    ("u1","u1"): k_uu[0][0],           ("u1","u2"): k_uu[0][1],
    ("u2","u2"): k_uu[1][1],
    ("u1","Su1"): k_u_dsu(0,0),        ("u1","Su2"): k_u_dsu(0,1),
    ("u2","Su1"): k_u_dsu(1,0),        ("u2","Su2"): k_u_dsu(1,1),
    ("u1","Sp"):  k_u_dsp(0),          ("u2","Sp"):  k_u_dsp(1),
    ("Su1","Su1"): k_dsu_dsu(0,0),     ("Su1","Su2"): k_dsu_dsu(0,1),
    ("Su2","Su1"): k_dsu_dsu(1,0),     ("Su2","Su2"): k_dsu_dsu(1,1),
    ("Su1","Sp"): k_dsu_dsp(0),        ("Su2","Sp"): k_dsu_dsp(1),
    ("Sp","Sp"):  k_dsp_dsp,
    ("u1","f1"): k_uf(0,0), ("u1","f2"): k_uf(0,1),
    ("u2","f1"): k_uf(1,0), ("u2","f2"): k_uf(1,1),
    ("u1","s"):  k_us(0),   ("u2","s"):  k_us(1),
    ("Su1","f1"): k_dsu_f(0,0), ("Su1","f2"): k_dsu_f(0,1),
    ("Su2","f1"): k_dsu_f(1,0), ("Su2","f2"): k_dsu_f(1,1),
    ("Su1","s"):  k_dsu_s(0),   ("Su2","s"):  k_dsu_s(1),
    ("Sp","f1"): k_dsp_f(0),    ("Sp","f2"): k_dsp_f(1),
    ("Sp","s"):  k_dsp_s,
    ("f1","f1"): k_ff(0,0), ("f1","f2"): k_ff(0,1),
    ("f2","f2"): k_ff(1,1),
    ("f1","s"): k_fs(0),    ("f2","s"): k_fs(1),
    ("s","s"):  k_ss,
}


def build_K(theta):
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

# theta_init = jnp.array([
#     1.0, jnp.log(0.50), jnp.log(0.30),   # u1-u1
#    0.8, jnp.log(0.60), jnp.log(0.35),   # u1-u2  
#     1.1, jnp.log(0.55), jnp.log(0.32),   # u2-u2
#    0.0, jnp.log(0.70), jnp.log(0.40),   # u1-p
#    0.0, jnp.log(0.65), jnp.log(0.38),   # u2-p
#     0.7, jnp.log(0.80), jnp.log(0.45),   # p-p
# ])
theta_init = jnp.array([
    1.2, jnp.log(0.50), jnp.log(0.30),   # u1-u1
   1.2, jnp.log(0.60), jnp.log(0.35),   # u1-u2  
    1.2, jnp.log(0.55), jnp.log(0.32),   # u2-u2
   0.0, jnp.log(0.70), jnp.log(0.40),   # u1-p
   0.0, jnp.log(0.65), jnp.log(0.38),   # u2-p
   1.2, jnp.log(0.80), jnp.log(0.45),   # p-p
])


y = jnp.concatenate([
    u1_train,                      # (62,)  no-slip, top wall
    u2_train,                      # (62,)  no-slip, bottom wall
    s_u_1,                     # (15,)  periodicity, u^x
    s_u_2,                     # (15,)  periodicity, u^y
    s_p,                       # (15,)  periodicity, p
    f_1,                       # (340,) = -ΔP/L
    f_2,                       # (340,) = 0
    s,                         # (340,) = 0
])
# print(y)
ntrain = y.shape[0]    

def inspect_K_blocks(theta):
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

@jax.jit
def neg_log_posterior(theta):
    K = build_K(theta) + (EPS_JITTER**2) * jnp.eye(ntrain)
    Lc = jnp.linalg.cholesky(K)
    v  = solve_triangular(Lc, y, lower=True)
    nlp = (0.5*jnp.dot(v, v) + jnp.sum(jnp.log(jnp.diag(Lc))) + 0.5*ntrain*jnp.log(2.0*jnp.pi)) + 0.5*jnp.sum(theta[0::3]**2)/(2.0**2)
    return nlp


fun = neg_log_posterior

def run_nelder_mead(init_params, fun, maxiter=500, tol=1e-2):
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


def config_fingerprint():
    """Hash everything that changes the optimum. A cache keyed only on a
    filename would silently return theta fitted for different physics."""
    cfg = {
        "dim": 2,
        "eta": float(η), "L": float(L), "dP": float(ΔP), "a": float(a),
        "avg_width": float(avg_width), "eps": float(EPS_JITTER),
        "margin": float(MARGIN),
        "N_f": int(N_f), "N_s": int(N_s),
        "n_wall": int(R_wall.shape[0]),
        "n_dSu": int(R_dSu1.shape[0]), "n_Sp": int(R_sp.shape[0]),
        "blocks": [name for name, _ in blocks],
        "block_sizes": [int(R.shape[0]) for _, R in blocks],
        "ntrain": int(ntrain),
        "optimizer": "nelder-mead",
        "nm_iter": int(args.nm_iter),
        "tol": float(args.tol),
        "theta_init": [float(v) for v in theta_init],
        "points_hash": _arr_hash(R_u_train, R_dSu1, R_dSu2, R_sp, R_f1, R_f2, R_s),
        "sparse": bool(args.sparse),
        "n_sparse": int(args.n_sparse) if args.sparse else 0,
        "sparse_seed": int(args.sparse_seed) if args.sparse else 0,
        "y_hash": _arr_hash(y),
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
NX, NY = 120, 41
x_line = jnp.linspace(0.0, L, NX)
eta_line = jnp.linspace(-0.999, 0.999, NY) * 0.5      
XX, EE = jnp.meshgrid(x_line, eta_line, indexing="ij")
YY = EE * width(XX)
R_test = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)      # (NX*NY, 2)

# ---- cross-covariance rows, one per target field --------------------------
row_u1 = [k_uu[0][0], k_uu[0][1], Sp(k_uu[0][0]), Sp(k_uu[0][1]),
          Sp(k_up[0]),  k_uf(0, 0), k_uf(0, 1), k_us(0)]

row_u2 = [k_uu[1][0], k_uu[1][1], Sp(k_uu[1][0]), Sp(k_uu[1][1]),
          Sp(k_up[1]),  k_uf(1, 0), k_uf(1, 1), k_us(1)]

row_p  = [k_pu[0],     k_pu[1],    Sp(k_pu[0]),   Sp(k_pu[1]),
          Sp(k_pp),     k_pf(0),    k_pf(1),      k_ps]

ROWS = {"u1": row_u1, "u2": row_u2, "p": row_p}
KSTAR = {"u1": k_uu[0][0], "u2": k_uu[1][1], "p": k_pp}

def build_K_star_obs(theta, R_star, star):
    """Q, shape (len(R_star), ntrain). Columns follow `blocks` order."""
    cols = [kmat(k)(R_star, R, theta)
            for k, (_, R) in zip(ROWS[star], blocks)]
    return jnp.concatenate(cols, axis=1)

def prior_var_diag(theta, R_star, star):
    """diag(K**) only - the full n_star x n_star matrix is never needed."""
    k = KSTAR[star]
    return jax.vmap(lambda r: k(r, r, theta))(R_star)

@partial(jax.jit, static_argnames=("star",))
def predict(theta, R_star, star="u1"):
    K  = build_K(theta) + (EPS_JITTER**2) * jnp.eye(ntrain)
    Lc = jnp.linalg.cholesky(K)
    Q  = build_K_star_obs(theta, R_star, star)

    alpha = solve_triangular(Lc.T, solve_triangular(Lc, y, lower=True), lower=False)
    mean  = Q @ alpha                                    # (n_star,)

    V    = solve_triangular(Lc, Q.T, lower=True)         # (ntrain, n_star)
    var  = prior_var_diag(theta, R_star, star) - jnp.sum(V**2, axis=0)
    std  = jnp.sqrt(jnp.clip(var, 0.0))
    return mean, std

fp, cfg = config_fingerprint()
print(f"[config] fingerprint {fp}  ({ntrain} training rows, {theta_init.shape[0]} hyperparameters)")

final_params = None
if not args.fit and not args.no_cache:
    final_params = load_cache(fp)
    if final_params is not None:
        print(f"[cache] loaded theta from {CACHE_PATH} (skipping optimization)")

if final_params is None:
    v0 = fun(theta_init)
    print(f"Initial value: {float(v0):.4e}")
    th, best = theta_init, float(v0)
    final_params, _ = run_nelder_mead(th, fun, maxiter=args.nm_iter, tol=args.tol)
    if not args.no_cache:
        save_cache(fp, final_params, fun(final_params))

print(f"\nFinal value: {fun(final_params):.4e}")

report_theta(final_params)

### Predictions

u1_mean, u1_std = predict(final_params, R_test, "u1")
u2_mean, u2_std = predict(final_params, R_test, "u2")
p_mean,  p_std  = predict(final_params, R_test, "p")

for nm, arr in [("u1", u1_mean), ("u2", u2_mean), ("p", p_mean),
                ("u1_std", u1_std)]:
    arr = onp.asarray(arr)
    bad = ~onp.isfinite(arr)
    print(f"{nm}: {bad.sum()}/{arr.size} non-finite", 
          f"finite range [{onp.nanmin(arr):.3f}, {onp.nanmax(arr):.3f}]" if (~bad).any() else "")
    
fem_data = None

if args.fem:
    f1, f2, fpres = fem_stokes.get(
        onp.asarray(R_test), OUTDIR / "fem_reference.npz",
        refit=args.fem_refit,
        L=float(L), a=float(a), avg_width=float(avg_width),
        eta=float(η), body_force_x=float(FBODY[0]),
        nx=args.fem_nx, ny=args.fem_ny)
    fem_data = {"u_x": f1, "u_y": f2, "p": fpres}

ctx = plotting.PlotCtx(
    XX=onp.asarray(XX), YY=onp.asarray(YY), NX=NX, NY=NY,
    L=float(L), a=float(a), avg_width=float(avg_width),
    R_wall=onp.asarray(R_wall), outdir=OUTDIR, tag=fp,
    subdir="sparse" if args.sparse else "base",
    sparse_pts=onp.asarray(R_u_train) if args.sparse else None)

plotting.plot_all(ctx, {
    "u_x": (u1_mean, u1_std),
    "u_y": (u2_mean, u2_std),
    "p":   (p_mean,  p_std),
}, fem=fem_data)
plt.show()