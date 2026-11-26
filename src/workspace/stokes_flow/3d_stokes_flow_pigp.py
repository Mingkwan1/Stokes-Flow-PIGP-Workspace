"""
Replicating Fujita-san work on inferring 3D Stokes flow using PIGP.

Steady Stokes flow in an x-periodic pipe of radius R(x) = width(x) / 2,
width(x) = avg_width + 2 a sin(2 pi x / L).

    python 3d_stokes_flow_pigp.py                          straight pipe (a = 0), vs exact Poiseuille
    python 3d_stokes_flow_pigp.py --geometry sinusoidal    sinusoidal pipe (a = 0.2), vs a 3D FEM reference
                                                      (fem_stokes_3d.py, solved once and cached)
"""

### Importing Libraries and Modules
import argparse
import time
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jrd
import jaxopt
import numpy as onp
from jax.scipy.linalg import solve_triangular

import fem_stokes_3d
import plotting_3d

## Command line

WALL_AMP = {"pipe": 0.0, "sinusoidal": 0.2}               # wall amplitude a per geometry

parser = argparse.ArgumentParser(description="Steady 3D Stokes flow in a periodic pipe with a PIGP.")
parser.add_argument("--geometry", choices=list(WALL_AMP), default="pipe",
                    help="pipe: straight, compared with exact Poiseuille (default); "
                         "sinusoidal: a = 0.2, compared with a 3D FEM reference")
parser.add_argument("--fem-refit", action="store_true",
                    help="re-solve the FEM reference even if a matching cache exists")
args = parser.parse_args()
GEOMETRY = args.geometry

## Output

OUTDIR = Path(__file__).resolve().parent / "output" / GEOMETRY   # next to this script
ERRDIR = OUTDIR / "errors"
ERRDIR.mkdir(parents=True, exist_ok=True)

## Functions

def Product_Squared_Exponential_Kernel(r, rp, theta):
    gamma, log_lx, log_ly, log_lz = theta[0], theta[1], theta[2], theta[3]
    return jnp.exp(gamma - 0.5*((r[0]-rp[0])/jnp.exp(log_lx))**2
                         - 0.5*((r[1]-rp[1])/jnp.exp(log_ly))**2
                         - 0.5*((r[2]-rp[2])/jnp.exp(log_lz))**2)

def u_exact(R):
    """Poiseuille (straight pipe only): u = (F (R0^2 - r^2) / (4 eta), 0, 0)."""
    R  = onp.asarray(R)
    r2 = R[:, 1]**2 + R[:, 2]**2
    ux = float(FBODY[0]) * (float(R0)**2 - r2) / (4 * η)
    return {"u1": ux, "u2": onp.zeros_like(ux), "u3": onp.zeros_like(ux)}

### Setting of the fluid characteristics and wall geometry

def width(x):
    """Width of the pipe at position x."""
    return avg_width + 2 * a * jnp.sin(2 * jnp.pi * x / L)

def radius(x):
    return width(x) / 2

η = 1.0          # Viscosity of the fluid
L = 2.5          # Characteristic Length
avg_width = 1.0  # Average width
a = WALL_AMP[GEOMETRY]           # wall amplitude, set by --geometry
REFERENCE = {"pipe": "exact", "sinusoidal": "fem"}[GEOMETRY]   # what errors are measured against

ΔP = -30.0                                     # Pressure Diff
FBODY = -ΔP/L * jnp.array([1.0, 0.0, 0.0])     # Body force

R0 = radius(0.0)                               # = radius(L)

### Input data

# ---- training points ---------------------------------------------------------
NX_WALL, NT_WALL = 24, 16    # wall rings along x  x  points per ring          -> 384
NR_DISK, NT_DISK = 20, 16    # periodic cross-section: radii x half-turn angles -> 160
DISK_MARGIN      = 0.98       # cross-section points stay inside 0.9 R (off the wall points)
N_F              = 400       # interior collocation points
F_MARGIN         = 0.98      # collocation points stay inside 0.98 R
SEED             = 67

assert NR_DISK % 2 == 0, "NR_DISK must be even (no repeated r = 0 point)"

## 1.) velocity at boundary (no-slip)

x_u_b, boundary_theta = jnp.meshgrid(jnp.linspace(0.0, L, NX_WALL),
                                     jnp.linspace(0.0, 2*jnp.pi, NT_WALL, endpoint=False),
                                     indexing="ij")
R_wall = jnp.stack([x_u_b,
                    radius(x_u_b) * jnp.cos(boundary_theta),
                    radius(x_u_b) * jnp.sin(boundary_theta)], axis=-1).reshape(-1, 3)

## 2.) + 3.) periodic velocity and pressure, cross-section at x = 0 (vs x = L)

radial, theta_around_disk = jnp.meshgrid(jnp.linspace(-DISK_MARGIN*R0, DISK_MARGIN*R0, NR_DISK),
                                         jnp.linspace(0.0, jnp.pi, NT_DISK, endpoint=False),
                                         indexing="ij")
R_disk = jnp.stack([jnp.zeros_like(radial),
                    radial * jnp.cos(theta_around_disk),
                    radial * jnp.sin(theta_around_disk)], axis=-1).reshape(-1, 3)
R_dSu = R_disk
R_sp  = R_disk

## 4.) + 5.) governing equations: momentum f and continuity s (same points)

def sample_interior(n, key, margin):
    """n random points inside the pipe: x uniform, uniform over each cross-section disk."""
    kx, kr, kt = jrd.split(key, 3)
    x = jrd.uniform(kx, (n,), minval=0.0, maxval=L)
    r = margin * radius(x) * jnp.sqrt(jrd.uniform(kr, (n,)))
    t = jrd.uniform(kt, (n,), minval=0.0, maxval=2*jnp.pi)
    return jnp.stack([x, r * jnp.cos(t), r * jnp.sin(t)], axis=-1)

R_f = sample_interior(N_F, jrd.PRNGKey(SEED), F_MARGIN)
R_s = R_f

### Kernel functions

theta = lambda t, i: jax.lax.dynamic_slice(t, (4*i,), (4,))

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

_eL = jnp.array([L, 0.0, 0.0])
S   = lambda k: None if k is None else (lambda r, rp, t: k(r, rp, t) - k(r + _eL, rp, t))
Sp  = lambda k: None if k is None else (lambda r, rp, t: k(r, rp, t) - k(r, rp + _eL, t))

### Base kernels (7 kernels x 4 = 28 parameters, u-p = 0)

_k = lambda i: (lambda r, rp, t: Product_Squared_Exponential_Kernel(r, rp, theta(t, i)))

k_uu = [[_k(0), _k(1), _k(2)],
        [_k(1), _k(3), _k(4)],
        [_k(2), _k(4), _k(5)]]

k_pp = _k(6)

k_up = [None, None, None]
k_pu = [None, None, None]

### Derived kernels

k_u_dsu = lambda a, b: Sp(k_uu[a][b])
k_u_dsp = lambda a:    Sp(k_up[a])
k_uf    = lambda a, b: add(Dp(k_up[a], b), mul(-η, Lapp(k_uu[a][b])))
k_us    = lambda a:    add(*[Dp(k_uu[a][b], b) for b in range(3)])

k_pf    = lambda a: add(Dp(k_pp, a), mul(-η, Lapp(k_pu[a])))
k_ps    =           add(*[Dp(k_pu[b], b) for b in range(3)])

k_dsu_dsu = lambda a, b: Sp(S(k_uu[a][b]))
k_dsu_dsp = lambda a:    Sp(S(k_up[a]))
k_dsu_f   = lambda a, b: S(k_uf(a, b))
k_dsu_s   = lambda a:    S(k_us(a))

k_dsp_dsp = Sp(S(k_pp))
k_dsp_f   = lambda b: S(k_pf(b))
k_dsp_s   = S(k_ps)

k_ff = lambda a, b: add(D(k_pf(b), a), mul(-η, Lap(k_uf(a, b))))
k_fs = lambda a:    add(D(k_ps, a),    mul(-η, Lap(k_us(a))))

k_ss = add(*[D(k_us(a), a) for a in range(3)])

### Kernel Matrix

nu  = lambda a: f"u{a+1}"
nSu = lambda a: f"Su{a+1}"
nf  = lambda a: f"f{a+1}"

kfn = {
    ("u1","u1"): k_uu[0][0],        ("u1","u2"): k_uu[0][1],        ("u1","u3"): k_uu[0][2],
    ("u2","u2"): k_uu[1][1],        ("u2","u3"): k_uu[1][2],
    ("u3","u3"): k_uu[2][2],

    ("u1","Su1"): k_u_dsu(0,0),     ("u1","Su2"): k_u_dsu(0,1),     ("u1","Su3"): k_u_dsu(0,2),
    ("u2","Su1"): k_u_dsu(1,0),     ("u2","Su2"): k_u_dsu(1,1),     ("u2","Su3"): k_u_dsu(1,2),
    ("u3","Su1"): k_u_dsu(2,0),     ("u3","Su2"): k_u_dsu(2,1),     ("u3","Su3"): k_u_dsu(2,2),

    ("u1","Sp"):  k_u_dsp(0),       ("u2","Sp"):  k_u_dsp(1),       ("u3","Sp"):  k_u_dsp(2),

    ("Su1","Su1"): k_dsu_dsu(0,0),  ("Su1","Su2"): k_dsu_dsu(0,1),  ("Su1","Su3"): k_dsu_dsu(0,2),
    ("Su2","Su1"): k_dsu_dsu(1,0),  ("Su2","Su2"): k_dsu_dsu(1,1),  ("Su2","Su3"): k_dsu_dsu(1,2),
    ("Su3","Su1"): k_dsu_dsu(2,0),  ("Su3","Su2"): k_dsu_dsu(2,1),  ("Su3","Su3"): k_dsu_dsu(2,2),

    ("Su1","Sp"):  k_dsu_dsp(0),    ("Su2","Sp"):  k_dsu_dsp(1),    ("Su3","Sp"):  k_dsu_dsp(2),

    ("Sp","Sp"):   k_dsp_dsp,
    ("Sp","f1"):   k_dsp_f(0),      ("Sp","f2"):   k_dsp_f(1),      ("Sp","f3"):   k_dsp_f(2),
    ("Sp","s"):    k_dsp_s,

    ("u1","f1"): k_uf(0,0),         ("u1","f2"): k_uf(0,1),         ("u1","f3"): k_uf(0,2),
    ("u2","f1"): k_uf(1,0),         ("u2","f2"): k_uf(1,1),         ("u2","f3"): k_uf(1,2),
    ("u3","f1"): k_uf(2,0),         ("u3","f2"): k_uf(2,1),         ("u3","f3"): k_uf(2,2),

    ("u1","s"):  k_us(0),           ("u2","s"):  k_us(1),           ("u3","s"):  k_us(2),

    ("Su1","f1"): k_dsu_f(0,0),     ("Su1","f2"): k_dsu_f(0,1),     ("Su1","f3"): k_dsu_f(0,2),
    ("Su2","f1"): k_dsu_f(1,0),     ("Su2","f2"): k_dsu_f(1,1),     ("Su2","f3"): k_dsu_f(1,2),
    ("Su3","f1"): k_dsu_f(2,0),     ("Su3","f2"): k_dsu_f(2,1),     ("Su3","f3"): k_dsu_f(2,2),

    ("Su1","s"):  k_dsu_s(0),       ("Su2","s"):  k_dsu_s(1),       ("Su3","s"):  k_dsu_s(2),

    ("f1","f1"): k_ff(0,0),         ("f1","f2"): k_ff(0,1),         ("f1","f3"): k_ff(0,2),
    ("f2","f1"): k_ff(1,0),         ("f2","f2"): k_ff(1,1),         ("f2","f3"): k_ff(1,2),
    ("f3","f1"): k_ff(2,0),         ("f3","f2"): k_ff(2,1),         ("f3","f3"): k_ff(2,2),

    ("f1","s"):  k_fs(0),           ("f2","s"):  k_fs(1),           ("f3","s"):  k_fs(2),

    ("s","s"):   k_ss,
}

KSTAR = {"u1": k_uu[0][0], "u2": k_uu[1][1], "u3": k_uu[2][2]}   # prior variance at test points

def kmat(k):
    if k is None:
        return lambda R1, R2, t: jnp.zeros((R1.shape[0], R2.shape[0]))
    return jax.vmap(jax.vmap(k, (None, 0, None)), (0, None, None))

def kblock(ki, Ri, kj, Rj, theta):
    if (ki, kj) in kfn:
        return kmat(kfn[(ki, kj)])(Ri, Rj, theta)
    return kmat(kfn[(kj, ki)])(Rj, Ri, theta).T

## Training blocks: (kind, points, values). Order = order of K and y.

n_wall, n_disk = R_wall.shape[0], R_disk.shape[0]

BLOCKS = ([(nu(a),  R_wall, jnp.zeros(n_wall))       for a in range(3)] +   # no-slip
          [(nSu(a), R_dSu,  jnp.zeros(n_disk))       for a in range(3)] +   # u periodic
          [("Sp",   R_sp,   jnp.zeros(n_disk))] +                           # p periodic
          [(nf(a),  R_f,    jnp.full(N_F, FBODY[a])) for a in range(3)] +   # momentum
          [("s",    R_s,    jnp.zeros(N_F))])                               # continuity

y      = jnp.concatenate([v for _, _, v in BLOCKS])
NTRAIN = y.shape[0]

def build_K(theta):
    n = len(BLOCKS)
    rows = [[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            ki, Ri, _ = BLOCKS[i]
            kj, Rj, _ = BLOCKS[j]
            rows[i][j] = kblock(ki, Ri, kj, Rj, theta)
            rows[j][i] = rows[i][j].T
    return jnp.block(rows)

EPS_JITTER = 1e-3

### Posterior Calculation

@jax.jit
def neg_log_posterior(theta):
    K  = build_K(theta) + (EPS_JITTER**2) * jnp.eye(NTRAIN)
    Lc = jnp.linalg.cholesky(K)
    v  = solve_triangular(Lc, y, lower=True)
    jeffreys_nlp = jnp.sum(theta)   # Molina eq. 29 on (A = e^gamma, l): sum(gamma) + sum(log l)
    nlp = (0.5*jnp.dot(v, v) + jnp.sum(jnp.log(jnp.diag(Lc)))
           + 0.5*NTRAIN*jnp.log(2.0*jnp.pi)) + jeffreys_nlp
    return nlp

@jax.jit
def nlp_safe(theta):
    """Cholesky failure (NaN) -> large finite value, so Nelder-Mead moves away."""
    v = neg_log_posterior(theta)
    return jnp.where(jnp.isfinite(v), v, 1e10)

### Optimization

NM_ITER        = 1000
NM_TOL         = 1e-5
NM_PRINT_EVERY = 25
LBFGS_ITER     = 300         # 0 = Nelder-Mead only
LBFGS_TOL      = 1e-10
LBFGS_SCALE    = 0.01        # L-BFGS optimizes z with theta = theta_NM + LBFGS_SCALE * z

GAMMA_AUTO     = 1.2         # initial gamma, u_a-u_a and p-p
GAMMA_CROSS    = 0.0         # initial gamma, u_a-u_b (a != b): must start below GAMMA_AUTO
ELL_INIT       = 0.3         # initial length scale, all blocks and axes

_NM_IT = [0]

def _nm_value_zero_grad(th):
    """Nelder-Mead uses only the value; the zero gradient satisfies jaxopt's interface."""
    return nlp_safe(th), jnp.zeros_like(th)

def _nm_callback(th):
    """Count iterations and print the objective every NM_PRINT_EVERY iterations."""
    _NM_IT[0] += 1
    if _NM_IT[0] % NM_PRINT_EVERY == 0:
        print(f"[NM] iter {_NM_IT[0]:5d}   nlp = {float(nlp_safe(th)):.6e}", flush=True)

@jax.jit
def _lbfgs_value_and_grad(z, th0):
    """theta = th0 + LBFGS_SCALE * z. Failed points: large value, zero gradient."""
    v, g = jax.value_and_grad(neg_log_posterior)(th0 + LBFGS_SCALE * z)
    ok = jnp.isfinite(v) & jnp.all(jnp.isfinite(g))
    return jnp.where(ok, v, 1e10), jnp.where(ok, LBFGS_SCALE * g, 0.0)

NM_SOLVER = jaxopt.ScipyMinimize(
    fun=_nm_value_zero_grad, value_and_grad=True,
    method="Nelder-Mead", tol=NM_TOL, maxiter=NM_ITER,
    callback=_nm_callback, options={"disp": True, "adaptive": True})

LBFGS_SOLVER = jaxopt.ScipyMinimize(
    fun=_lbfgs_value_and_grad, value_and_grad=True, method="L-BFGS-B",
    tol=LBFGS_TOL, maxiter=LBFGS_ITER)

def run_nelder_mead(x0):
    _NM_IT[0] = 0
    res = NM_SOLVER.run(x0)
    return res.params, int(res.state.iter_num)

def run_nm_then_lbfgs(x0):
    ta, nit = run_nelder_mead(x0)
    if LBFGS_ITER <= 0:
        return ta, nit
    f_nm = float(nlp_safe(ta))
    res  = LBFGS_SOLVER.run(jnp.zeros_like(ta), ta)
    tb   = ta + LBFGS_SCALE * res.params
    f_lb = float(nlp_safe(tb))
    ok   = bool(jnp.all(jnp.isfinite(tb))) and f_lb < f_nm
    print(f"[lbfgs] NM {f_nm:.6e} -> LBFGS {f_lb:.6e}  "
          f"({int(res.state.iter_num)} iters)  {'accepted' if ok else 'REJECTED, keeping NM'}")
    return (tb if ok else ta), nit + int(res.state.iter_num)

### Prediction

NX_TEST, NR_TEST = 61, 41    # plane grids: points along x  x  across the pipe
TEST_MARGIN      = 0.98      # plane grids stay inside 0.98 R
N_PROFILE        = 101       # points on each profile line
X_PROFILE        = L / 2     # x position of the profile lines
N_TEST_VOL       = 4000      # random points over the whole volume (error statistics)

# ---- FEM reference (sinusoidal pipe), see fem_stokes_3d.py ---------------------
FEM_DISK_REFS    = 4         # cross-section mesh: 64 wall points per ring, 1024 triangles
FEM_NX           = 64        # slices along x
FEM_CACHE        = OUTDIR / "fem_reference.npz"

@jax.jit
def factorize(th):
    Lc    = jnp.linalg.cholesky(build_K(th) + EPS_JITTER**2 * jnp.eye(NTRAIN))
    alpha = solve_triangular(Lc.T, solve_triangular(Lc, y, lower=True), lower=False)
    return Lc, alpha

@partial(jax.jit, static_argnames=("star",))
def predict(th, Lc, alpha, R_star, star):
    Q     = jnp.concatenate([kblock(star, R_star, kind, R, th) for kind, R, _ in BLOCKS], axis=1)
    mean  = Q @ alpha
    V     = solve_triangular(Lc, Q.T, lower=True)
    prior = jax.vmap(lambda r: KSTAR[star](r, r, th))(R_star)
    std   = jnp.sqrt(jnp.clip(prior - jnp.sum(V**2, axis=0), 0.0))
    return mean, std

@partial(jax.jit, static_argnames=("star",))
def predict_mean(th, alpha, R_star, star):
    """Posterior mean of any training kind at R_star (u1..3, f1..3, s)."""
    Q = jnp.concatenate([kblock(star, R_star, kind, R, th) for kind, R, _ in BLOCKS], axis=1)
    return Q @ alpha

def predict_u(th, Lc, alpha, R_star):
    return {c: tuple(onp.asarray(v) for v in predict(th, Lc, alpha, R_star, c))
            for c in ("u1", "u2", "u3")}

def plane_grid(axis):
    """axis = 1: x-y plane (z = 0);  axis = 2: x-z plane (y = 0)."""
    XX, EE = jnp.meshgrid(jnp.linspace(0.0, L, NX_TEST),
                          jnp.linspace(-TEST_MARGIN, TEST_MARGIN, NR_TEST), indexing="ij")
    HH = EE * radius(XX)
    R  = jnp.zeros(XX.shape + (3,)).at[..., 0].set(XX).at[..., axis].set(HH)
    return onp.asarray(XX), onp.asarray(HH), R.reshape(-1, 3)

def profile_line(axis):
    """Line across the pipe at x = X_PROFILE along y (axis 1) or z (axis 2)."""
    h = jnp.linspace(-1.0, 1.0, N_PROFILE) * radius(X_PROFILE)
    R = jnp.zeros((N_PROFILE, 3)).at[:, 0].set(X_PROFILE).at[:, axis].set(h)
    return onp.asarray(h), R

## Diagnostic

COMPS      = ("u1", "u2", "u3")
COMP_NAME  = {"u1": "u_x", "u2": "u_y", "u3": "u_z"}
STAT_COLS  = ("min", "max", "mean", "mean|.|")

def _stats(e):
    """Statistics over the finite values (NaN = reference point outside the FEM mesh)."""
    e = e[onp.isfinite(e)]
    return {"min": e.min(), "max": e.max(), "mean": e.mean(), "mean|.|": onp.abs(e).mean()}

def error_stats(pred, exact):
    """Per component, two tables against the reference u_ref (exact or FEM):
    absolute error  e     = u_pred - u_ref
    relative error  e_rel = (u_pred - u_ref) / u_ref,  set to 0 where u_ref = 0."""
    absolute, relative = {}, {}
    for c in COMPS:
        e   = pred[c][0] - exact[c]
        nz  = exact[c] != 0
        rel = onp.where(nz, e / onp.where(nz, exact[c], 1.0), 0.0)
        absolute[COMP_NAME[c]] = _stats(e)
        relative[COMP_NAME[c]] = _stats(rel)
    n_skip = int((~onp.isfinite(exact["u1"])).sum())
    note = f"   ({n_skip} points outside the FEM mesh skipped)" if n_skip else ""
    return {"absolute error  e = u_pred - u_ref" + note: absolute,
            "relative error  (u_pred - u_ref) / u_ref   (0 where u_ref = 0)": relative}

def residual_stats(th, alpha, R):
    """PDE residuals of the posterior mean: div u and f_a - F_a (no reference needed)."""
    rows = {"div u": onp.asarray(predict_mean(th, alpha, R, "s"))}
    for i in range(3):
        rows[f"f{i+1} - F{i+1}"] = onp.asarray(predict_mean(th, alpha, R, nf(i))) - float(FBODY[i])
    return {"residual of the posterior mean": {k: _stats(v) for k, v in rows.items()}}

def write_report(path, header, sections):
    """sections: [(title, n_points, {table_title: {row: {stat: value}}})]"""
    lines = [f"# {h}" for h in header] + [""]
    for title, n, tables in sections:
        lines.append(f"[{title}]  n = {n}")
        for table_title, rows in tables.items():
            lines.append(f"  {table_title}")
            lines.append(f"  {'':12s}" + "".join(f"{c:>14s}" for c in STAT_COLS))
            for name, st in rows.items():
                lines.append(f"  {name:12s}" + "".join(f"{st[c]:14.6e}" for c in STAT_COLS))
        lines.append("")
    text = "\n".join(lines)
    path.write_text(text)
    print(text)
    print(f"[save] {path}")

## Reference solution

def reference_solutions(point_sets):
    """Reference velocity {"u1", "u2", "u3"} for each named point set.

    pipe       : exact Poiseuille
    sinusoidal : 3D FEM (scikit-fem), solved once for all points together and
                 cached in FEM_CACHE; the cache is reused while the geometry,
                 mesh and points are unchanged.
    """
    if REFERENCE == "exact":
        return {k: u_exact(R) for k, R in point_sets.items()}
    names = list(point_sets)
    sizes = [point_sets[k].shape[0] for k in names]
    allpts = onp.concatenate([onp.asarray(point_sets[k]) for k in names])
    u1, u2, u3, _ = fem_stokes_3d.get(
        allpts, FEM_CACHE, refit=args.fem_refit,
        L=float(L), a=float(a), avg_width=float(avg_width), eta=float(η),
        body_force_x=float(FBODY[0]), disk_refs=FEM_DISK_REFS, nx=FEM_NX)
    out, i = {}, 0
    for k, n in zip(names, sizes):
        out[k] = {"u1": u1[i:i+n], "u2": u2[i:i+n], "u3": u3[i:i+n]}
        i += n
    return out

REF_NAME = {"exact": "exact Poiseuille",
            "fem": f"3D FEM, scikit-fem Taylor-Hood P2/P1, disk_refs = {FEM_DISK_REFS}, nx = {FEM_NX}"}[REFERENCE]
REF_LABEL = {"exact": "Poiseuille", "fem": "FEM"}[REFERENCE]

### Main Driver

if __name__ == "__main__":
    print(f"[setup] reference: {REF_NAME}")
    print(f"[setup] geometry = {GEOMETRY}   a = {a}   R0 = {float(R0):.3f}   F = {float(FBODY[0]):.3f}")
    print(f"[setup] wall {n_wall} pts, cross-section {n_disk} pts, interior {N_F} pts "
          f"->  ntrain = {NTRAIN}")
    print(f"[setup] blocks: {[kind for kind, _, _ in BLOCKS]}")
    print(f"[setup] output: {OUTDIR}")

    _auto  = [GAMMA_AUTO,  onp.log(ELL_INIT), onp.log(ELL_INIT), onp.log(ELL_INIT)]
    _cross = [GAMMA_CROSS, onp.log(ELL_INIT), onp.log(ELL_INIT), onp.log(ELL_INIT)]
    theta_init = jnp.array(_auto + _cross + _cross       # u1-u1, u1-u2, u1-u3
                           + _auto + _cross              # u2-u2, u2-u3
                           + _auto                       # u3-u3
                           + _auto)                      # p-p

    t0 = time.time()
    nlp0 = float(neg_log_posterior(theta_init))
    print(f"[init] nlp = {nlp0:.6e}   (compile + 1 eval: {time.time() - t0:.1f} s)")
    if not onp.isfinite(nlp0):
        raise RuntimeError("Initial K is not positive definite: increase EPS_JITTER "
                           "or lower GAMMA_CROSS.")

    t0 = time.time()
    theta_opt, n_iter = run_nm_then_lbfgs(theta_init)
    print(f"[opt] {n_iter} iterations in {time.time() - t0:.1f} s   "
          f"nlp = {float(nlp_safe(theta_opt)):.6e}")

    names = ["u1-u1", "u1-u2", "u1-u3", "u2-u2", "u2-u3", "u3-u3", "p-p"]
    th = onp.asarray(theta_opt).reshape(7, 4)
    print(f"{'block':>6} {'A=e^g':>10} {'lx':>10} {'ly':>10} {'lz':>10}")
    for nm, (g, lx, ly, lz) in zip(names, th):
        print(f"{nm:>6} {onp.exp(g):10.4f} {onp.exp(lx):10.4f} {onp.exp(ly):10.4f} {onp.exp(lz):10.4f}")

    Lc, alpha = factorize(theta_opt)
    wall_x = onp.linspace(0.0, L, 300)
    wall_r = onp.asarray(radius(jnp.asarray(wall_x)))

    # ---- evaluation points and their reference velocity -----------------------
    grids = {tag: plane_grid(axis) for axis, tag in [(1, "xy"), (2, "xz")]}
    lines = {tag: profile_line(axis) for axis, tag in [(1, "py"), (2, "pz")]}
    R_vol = sample_interior(N_TEST_VOL, jrd.PRNGKey(SEED + 1), TEST_MARGIN)
    ref = reference_solutions({"vol": R_vol,
                               **{t: g[2] for t, g in grids.items()},
                               **{t: ln[1] for t, ln in lines.items()}})

    # ---- whole volume: errors -------------------------------------------------
    sections = [("error, volume", N_TEST_VOL,
                 error_stats(predict_u(theta_opt, Lc, alpha, R_vol), ref["vol"]))]

    # ---- planes: x-y (z = 0) and x-z (y = 0) ---------------------------------
    results = {}
    for tag, vlabel in [("xy", "$y$"), ("xz", "$z$")]:
        XX, HH, R = grids[tag]
        pred = predict_u(theta_opt, Lc, alpha, R)
        sections.append((f"error, plane {tag}", R.shape[0], error_stats(pred, ref[tag])))
        plotting_3d.plot_plane(XX, HH, pred, ref[tag], vlabel, wall_x, wall_r,
                               OUTDIR / f"plane_{tag}.png")
        results.update({f"{tag}_X": XX, f"{tag}_H": HH,
                        **{f"{tag}_{c}_mean": m for c, (m, _) in pred.items()},
                        **{f"{tag}_{c}_std": s for c, (_, s) in pred.items()},
                        **{f"{tag}_{c}_ref": ref[tag][c] for c in COMPS}})

    # ---- profiles of u_x across the pipe at x = X_PROFILE ---------------------
    profiles = {}
    for tag, vlabel in [("py", "$y$  ($z = 0$)"), ("pz", "$z$  ($y = 0$)")]:
        h, R = lines[tag]
        m, s = (onp.asarray(v) for v in predict(theta_opt, Lc, alpha, R, "u1"))
        profiles[vlabel] = (h, m, s, ref[tag]["u1"])
    plotting_3d.plot_profiles(profiles, X_PROFILE, OUTDIR / "profiles_ux.png", ref_label=REF_LABEL)

    # ---- PDE residuals (no reference needed) -----------------------------------
    sections.append(("PDE residual, volume", N_TEST_VOL, residual_stats(theta_opt, alpha, R_vol)))

    write_report(ERRDIR / "error_stats.txt", [
        f"geometry = {GEOMETRY}   a = {a}   L = {L}   R0 = {float(R0)}   F = {float(FBODY[0])}",
        f"reference = {REF_NAME}",
        f"ntrain = {NTRAIN}   nlp = {float(nlp_safe(theta_opt)):.6e}   iterations = {n_iter}",
        "columns: min, max, mean of the signed value;  mean|.| = mean of its absolute value",
        "absolute error: u_pred - u_ref;  relative error: (u_pred - u_ref) / u_ref, 0 where u_ref = 0",
        "residual: div u and f_a - F_a of the posterior mean (no reference needed)",
    ], sections)

    onp.savez(OUTDIR / "results.npz", theta_init=onp.asarray(theta_init),
              theta_opt=onp.asarray(theta_opt), **results)
    print(f"[save] {OUTDIR / 'results.npz'}")