"""Unsteady 2D Stokes flow in a sinusoidal channel -- numerical GP, backward Euler.

Raissi, Perdikaris & Karniadakis (2017) sec. 2.2-2.5; Stokes kernels and shift
operator S from Molina et al. (2023) eqs. (20)-(26), appendix B.

Prior on (u^n, p^n), k_{u_a p} = 0. Observations at step n:
    u_a = 0                                   on the walls
    S_L u_a = S_L p = 0                       on x = 0
    G_a := u_a + c (d_a p - eta Lap u_a) = u_a^{n-1} + c F_a   at x^{n-1},  c = dt/rho
    s := div u = 0                            at the collocation grid
x^0: initial data u^0 = 0 on the N_f collocation grid (step 1);
x^{n-1}, n > 1: fresh uniform random locations every step, new seed every run.
theta^n: all hyperparameters refit (Nelder-Mead) every step, warm-started from theta^{n-1}.
u^{n-1} ~ N(mu^{n-1}, Sigma^{n-1}) marginalized in prediction, eqs. (13)-(14).
"""
import argparse
import hashlib
import json
import secrets
import time
from functools import partial
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrd
from jax.scipy.linalg import solve_triangular

import numpy as onp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import plotting
import fem_stokes
import evolution_plot

# ============================================================================ CLI
p = argparse.ArgumentParser(description="2D unsteady Stokes numerical GP")
p.add_argument("--kernel", choices=["u1u2", "independent"], default="u1u2",
               help="u1u2: blocks u1u1,u1u2,u2u2,pp (12 params). "
                    "independent: u1u1,u2u2,pp (9 params). k_up = 0 in both")
p.add_argument("--maxiter", type=int, default=500, help="Nelder-Mead iterations per fit")
p.add_argument("--tol", type=float, default=1e-4, help="Nelder-Mead xatol = fatol")
p.add_argument("--nm-step", type=float, default=0.1,
               help="initial simplex edge (in theta units: gamma, log l)")
p.add_argument("--nm-print", type=int, default=50, help="print every k iterations (0 = off)")
p.add_argument("--refit-every", type=int, default=1, help="0 = freeze after step 1")
p.add_argument("--gamma-bounds", type=float, nargs=2, default=[-10.0, 10.0])
p.add_argument("--lu-bounds", type=float, nargs=2, default=[0.05, 1.5],
               help="velocity length-scale bounds")
p.add_argument("--lp-bounds", type=float, nargs=2, default=[0.4, 3.0],
               help="pressure length-scale bounds; small values let p absorb F")
p.add_argument("--l0", type=float, default=0.3, help="prior median, velocity")
p.add_argument("--lp0", type=float, default=1.0, help="prior median, pressure")
p.add_argument("--l-prior-std", type=float, default=0.5, help="0 = bounds only")
p.add_argument("--jitter", type=float, default=1e-3, help="eps; eps^2 added to diag(K)")

p.add_argument("--n-loop", type=int, default=10)
p.add_argument("--dt", type=float, default=0.01)
p.add_argument("--n-artificial", type=int, default=100,
               help="artificial points per step for n > 1 (step 1 uses the N_f grid)")
p.add_argument("--seed", type=int, default=None, help="default: new random seed each run")
p.add_argument("--art-margin", type=float, default=0.95,
               help="artificial points within this fraction of the half-width")
p.add_argument("--plot-every", type=int, default=1)
p.add_argument("--test-margin", type=float, default=0.999)
p.add_argument("--n-snap", type=int, default=5)
p.add_argument("--evolution-cmap", default="RdBu_r")

p.add_argument("--fem", action="store_true")
p.add_argument("--fem-refit", action="store_true")
p.add_argument("--fem-nx", type=int, default=200)
p.add_argument("--fem-ny", type=int, default=80)
args = p.parse_args()

SEED = args.seed if args.seed is not None else secrets.randbits(31)

OUTDIR = Path(__file__).resolve().parent / "outputs"/"outputs_unsteady_stokes_flow_timestep_0.05"
for d in ("plots/base", "states"):
    (OUTDIR / d).mkdir(parents=True, exist_ok=True)

# ======================================================================= physics
η, ρ = 1.0, 1.0
L, a, avg_width = 2.5, 0.2, 1.0
ΔP = -30.0
FBODY = -ΔP / L * jnp.array([1.0, 0.0])
MARGIN = 0.999
Δt = args.dt
c = Δt / ρ
EPS = args.jitter
M = args.n_artificial          # artificial points for n > 1


def width(x):
    return avg_width + 2 * a * jnp.sin(2 * jnp.pi * x / L)


# ================================================================ training points
N_WALL, N_PER, N_F = 62, 15, 340

x_w = jnp.linspace(0.0, L, N_WALL)
R_wall = jnp.concatenate([jnp.stack([x_w,  width(x_w) / 2], -1),
                          jnp.stack([x_w, -width(x_w) / 2], -1)])       # (124, 2)

# endpoints y = ±w(0)/2 excluded: S_L u there = combination of two wall rows
y_per = jnp.linspace(-width(0.0) / 2, width(0.0) / 2, N_PER + 2)[1:-1]
R_per = jnp.stack([jnp.zeros_like(y_per), y_per], -1)                  # (15, 2)


def points_grid(N=N_F, n_cols=20):
    n_rows = int(onp.ceil(N / n_cols))
    x_c = (jnp.arange(n_cols) + 0.5) / n_cols * L
    eta = ((jnp.arange(n_rows) + 0.5) / n_rows - 0.5) * MARGIN
    X, E = jnp.meshgrid(x_c, eta, indexing="ij")
    x = X.ravel()[:N]
    return jnp.stack([x, E.ravel()[:N] * width(x)], -1)


R_s = points_grid()                                                     # (340, 2)


def sample_artificial(key, m=M):
    """Uniform random interior points (paper sec. 2.2)."""
    kx, ke = jrd.split(key)
    x = jrd.uniform(kx, (m,), minval=0.0, maxval=L)
    e = jrd.uniform(ke, (m,), minval=-0.5, maxval=0.5) * args.art_margin
    return jnp.stack([x, e * width(x)], -1)


# ================================================================ kernel algebra
# None = zero kernel: k_up = 0 removes every pressure-velocity derivative term.
_eL = jnp.array([L, 0.0])


def D(k, i):   return None if k is None else (lambda r, rp, t: jax.grad(k, 0)(r, rp, t)[i])
def Dp(k, i):  return None if k is None else (lambda r, rp, t: jax.grad(k, 1)(r, rp, t)[i])
def Lap(k):    return None if k is None else (lambda r, rp, t: jnp.trace(jax.hessian(k, 0)(r, rp, t)))
def Lapp(k):   return None if k is None else (lambda r, rp, t: jnp.trace(jax.hessian(k, 1)(r, rp, t)))
def S(k):      return None if k is None else (lambda r, rp, t: k(r, rp, t) - k(r + _eL, rp, t))
def Sp(k):     return None if k is None else (lambda r, rp, t: k(r, rp, t) - k(r, rp + _eL, t))
def swap(k):   return None if k is None else (lambda r, rp, t: k(rp, r, t))
def mul(s, k): return None if k is None else (lambda r, rp, t: s * k(r, rp, t))


def add(*ks):
    ks = [k for k in ks if k is not None]
    if not ks:
        return None
    if len(ks) == 1:
        return ks[0]
    return lambda r, rp, t: sum(k(r, rp, t) for k in ks)


def se(r, rp, th):
    """exp(gamma - dx^2/(2 lx^2) - dy^2/(2 ly^2)),  th = (gamma, log lx, log ly)."""
    return jnp.exp(th[0] - 0.5 * ((r[0] - rp[0]) / jnp.exp(th[1])) ** 2
                         - 0.5 * ((r[1] - rp[1]) / jnp.exp(th[2])) ** 2)


if args.kernel == "u1u2":
    PARAM_BLOCKS = ["u1u1", "u1u2", "u2u2", "pp"]
    _TH0 = {"u1u1": 1.2, "u1u2": -1.0, "u2u2": 1.2, "pp": 1.2}
else:
    PARAM_BLOCKS = ["u1u1", "u2u2", "pp"]
    _TH0 = {"u1u1": 1.2, "u2u2": 1.2, "pp": 1.2}

_L0 = {b: (args.lp0 if b == "pp" else args.l0) for b in PARAM_BLOCKS}
theta_init = jnp.array(sum([[_TH0[b], onp.log(_L0[b]), onp.log(_L0[b])]
                            for b in PARAM_BLOCKS], []))
LOGL0 = jnp.log(jnp.array([_L0[b] for b in PARAM_BLOCKS] * 2))   # order: all lx, all ly


def blk(name):
    i = PARAM_BLOCKS.index(name)
    return lambda r, rp, t: se(r, rp, t[3 * i:3 * i + 3])


if args.kernel == "u1u2":
    k_uu = [[blk("u1u1"), blk("u1u2")], [swap(blk("u1u2")), blk("u2u2")]]
else:
    k_uu = [[blk("u1u1"), None], [None, blk("u2u2")]]
k_up = [None, None]                      # assumption: u-p cross-covariance = 0
k_pu = [None, None]
k_pp = blk("pp")

# f_a := d_a p - eta Lap u_a,  s := div u,  G_a := u_a + c f_a
k_uf = lambda i, j: add(Dp(k_up[i], j), mul(-η, Lapp(k_uu[i][j])))     # <u_i f_j'>
k_fu = lambda i, j: add(D(k_pu[j], i),  mul(-η, Lap(k_uu[i][j])))      # <f_i u_j'>
k_pf = lambda j:    add(Dp(k_pp, j),    mul(-η, Lapp(k_pu[j])))        # <p f_j'>
k_ff = lambda i, j: add(D(k_pf(j), i),  mul(-η, Lap(k_uf(i, j))))      # <f_i f_j'>
k_us = lambda i:    add(*[Dp(k_uu[i][j], j) for j in range(2)])        # <u_i s'>
k_ps =              add(*[Dp(k_pu[j], j) for j in range(2)])           # <p s'>  (= 0)
k_fs = lambda i:    add(D(k_ps, i),     mul(-η, Lap(k_us(i))))         # <f_i s'>
k_ss =              add(*[D(k_us(i), i) for i in range(2)])            # <s s'>

k_uG = lambda i, j: add(k_uu[i][j], mul(c, k_uf(i, j)))
k_pG = lambda j:    add(k_pu[j],    mul(c, k_pf(j)))
k_GG = lambda i, j: add(k_uu[i][j], mul(c, k_uf(i, j)), mul(c, k_fu(i, j)),
                        mul(c ** 2, k_ff(i, j)))
k_Gs = lambda i:    add(k_us(i),    mul(c, k_fs(i)))

_RANK = {"u": 0, "p": 1, "G": 2, "s": 3}


def cov(A, B):
    """<A(r) B(r')> for A, B in {("u",i), ("p",), ("G",i), ("s",), ("S", X)}."""
    if A[0] == "S":
        return S(cov(A[1], B))
    if B[0] == "S":
        return Sp(cov(A, B[1]))
    if _RANK[A[0]] > _RANK[B[0]]:
        return swap(cov(B, A))
    i = A[1] if len(A) > 1 else None
    j = B[1] if len(B) > 1 else None
    return {("u", "u"): lambda: k_uu[i][j], ("u", "p"): lambda: k_up[i],
            ("u", "G"): lambda: k_uG(i, j), ("u", "s"): lambda: k_us(i),
            ("p", "p"): lambda: k_pp,       ("p", "G"): lambda: k_pG(j),
            ("p", "s"): lambda: k_ps,       ("G", "G"): lambda: k_GG(i, j),
            ("G", "s"): lambda: k_Gs(i),    ("s", "s"): lambda: k_ss}[(A[0], B[0])]()


# ======================================================================= blocks
OBS = [("u1", ("u", 0), "wall"), ("u2", ("u", 1), "wall"),
       ("Su1", ("S", ("u", 0)), "per"), ("Su2", ("S", ("u", 1)), "per"),
       ("Sp", ("S", ("p",)), "per"),
       ("G1", ("G", 0), "G"), ("G2", ("G", 1), "G"), ("s", ("s",), "s")]
BLOCK_NAMES = [o[0] for o in OBS]
STAR = {"u1": ("u", 0), "u2": ("u", 1), "p": ("p",)}

KFN = {(i, j): cov(OBS[i][1], OBS[j][1]) for i in range(len(OBS)) for j in range(i, len(OBS))}
KROW = {s: [cov(o, ob[1]) for ob in OBS] for s, o in STAR.items()}
KDIAG = {s: cov(o, o) for s, o in STAR.items()}
K_UU_JOINT = [cov(STAR["u1"], STAR["u1"]), cov(STAR["u1"], STAR["u2"]),
              cov(STAR["u2"], STAR["u2"])]

_SIZE = {"wall": R_wall.shape[0], "per": R_per.shape[0], "s": R_s.shape[0]}
G_START = sum(_SIZE[o[2]] for o in OBS[:BLOCK_NAMES.index("G1")])


def n_train(m):
    """Rows of K when the G blocks hold m points each."""
    return sum(m if o[2] == "G" else _SIZE[o[2]] for o in OBS)


def g_rows(R_G):
    """Rows of K holding u^{n-1} (static under jit: depends on shape only)."""
    return slice(G_START, G_START + 2 * R_G.shape[0])


def make_blocks(R_G):
    return [{"wall": R_wall, "per": R_per, "G": R_G, "s": R_s}[o[2]] for o in OBS]


def kmat(k, R1, R2, th):
    if k is None:
        return jnp.zeros((R1.shape[0], R2.shape[0]))
    return jax.vmap(jax.vmap(k, (None, 0, None)), (0, None, None))(R1, R2, th)


def build_K(th, R_G):
    Rs = make_blocks(R_G)
    n = len(Rs)
    rows = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            rows[i][j] = kmat(KFN[(i, j)], Rs[i], Rs[j], th)
            rows[j][i] = rows[i][j].T
    return jnp.block(rows)


def make_y(u1_prev, u2_prev):
    return jnp.concatenate([jnp.zeros(G_START),
                            u1_prev + c * FBODY[0],
                            u2_prev + c * FBODY[1],
                            jnp.zeros(R_s.shape[0])])


# ================================================================ objective / fit
def log_prior(th):
    """-log p(theta): N(0, 2^2) on gamma, N(log l0, s^2) on log l."""
    lp = 0.5 * jnp.sum(th[0::3] ** 2) / 2.0 ** 2
    if args.l_prior_std > 0:
        logl = jnp.concatenate([th[1::3], th[2::3]])
        lp = lp + 0.5 * jnp.sum((logl - LOGL0) ** 2) / args.l_prior_std ** 2
    return lp


@jax.jit
def neg_log_posterior(th, y, R_G):
    """Training uses the mean of u^{n-1} (paper footnote 2)."""
    K = build_K(th, R_G) + EPS ** 2 * jnp.eye(y.shape[0])
    Lc = jnp.linalg.cholesky(K)
    v = solve_triangular(Lc, y, lower=True)
    return (0.5 * v @ v + jnp.sum(jnp.log(jnp.diag(Lc)))
            + 0.5 * y.shape[0] * jnp.log(2 * jnp.pi) + log_prior(th))


BAD = 1e12
BOUNDS = []
for _b in PARAM_BLOCKS:
    _lo, _hi = args.lp_bounds if _b == "pp" else args.lu_bounds
    _lb = (float(onp.log(_lo)), float(onp.log(_hi)))
    BOUNDS += [tuple(args.gamma_bounds), _lb, _lb]
LO, HI = onp.array(BOUNDS).T
U_GAMMA_IDX = [3 * k for k, b in enumerate(PARAM_BLOCKS) if b in ("u1u1", "u2u2")]


def initial_simplex(x0, h):
    """x0 plus one step h per coordinate, stepping inward when x0 sits near a bound."""
    sim = onp.tile(x0, (x0.size + 1, 1))
    for k in range(x0.size):
        sim[k + 1, k] += h if x0[k] + h <= HI[k] else -h
    return sim


def fit_theta(th0, y, R_G, maxiter, label):
    """Nelder-Mead from th0. Non-finite NLP -> BAD. Never returns worse than th0."""
    x_in = onp.asarray(th0, float)
    x0 = onp.clip(x_in, LO, HI)
    if not onp.allclose(x0, x_in):
        print(f"      [{label}] warm start clipped into bounds")

    def f(x):
        v = float(neg_log_posterior(jnp.asarray(x), y, R_G))
        return v if onp.isfinite(v) else BAD

    it = [0]

    def cb(intermediate_result):
        it[0] += 1
        if args.nm_print and it[0] % args.nm_print == 0:
            print(f"      [{label}] NM iter {it[0]:4d}  nlp {intermediate_result.fun:.6e}",
                  flush=True)

    f0 = f(x0)
    res = minimize(f, x0, method="Nelder-Mead", bounds=BOUNDS, callback=cb,
                   options=dict(maxiter=maxiter, xatol=args.tol, fatol=args.tol,
                                adaptive=True, initial_simplex=initial_simplex(x0, args.nm_step)))
    if not onp.isfinite(res.fun) or res.fun >= BAD or res.fun > f0:
        return jnp.asarray(x0), f0, res.nit, res.nfev, False
    return jnp.asarray(res.x), float(res.fun), res.nit, res.nfev, True


@jax.jit
def factorize(th, y, R_G):
    Lc = jnp.linalg.cholesky(build_K(th, R_G) + EPS ** 2 * jnp.eye(y.shape[0]))
    alpha = solve_triangular(Lc.T, solve_triangular(Lc, y, lower=True), lower=False)
    return Lc, alpha


# ==================================================================== posterior
def build_Q(th, R_star, star, R_G):
    return jnp.concatenate([kmat(k, R_star, R, th)
                            for k, R in zip(KROW[star], make_blocks(R_G))], axis=1)


@partial(jax.jit, static_argnames=("star",))
def predict(Lc, alpha, th, R_star, R_G, Sigma_prev, star):
    """Eq. (13): k** - q^T K^-1 q + q^T K^-1 [0;Sigma] K^-1 q (diagonal)."""
    Q = build_Q(th, R_star, star, R_G)
    mean = Q @ alpha
    V = solve_triangular(Lc, Q.T, lower=True)
    W = solve_triangular(Lc.T, V, lower=False)[g_rows(R_G)]   # (2m, n*)
    var = (jax.vmap(lambda r: KDIAG[star](r, r, th))(R_star)
           - jnp.sum(V ** 2, 0) + jnp.sum(W * (Sigma_prev @ W), 0))
    return mean, jnp.sqrt(jnp.clip(var, 0.0))


@jax.jit
def predict_artificial(Lc, alpha, th, R_new, R_G, Sigma_prev):
    """Joint (u1, u2) posterior at the next artificial locations, eq. (14)."""
    Q = jnp.concatenate([build_Q(th, R_new, "u1", R_G),
                         build_Q(th, R_new, "u2", R_G)], 0)
    mean = Q @ alpha
    K11, K12, K22 = (kmat(k, R_new, R_new, th) for k in K_UU_JOINT)
    V = solve_triangular(Lc, Q.T, lower=True)
    W = solve_triangular(Lc.T, V, lower=False)[g_rows(R_G)]
    Sig = jnp.block([[K11, K12], [K12.T, K22]]) - V.T @ V + W.T @ Sigma_prev @ W
    m = R_new.shape[0]
    return mean[:m], mean[m:], 0.5 * (Sig + Sig.T)


def report_theta(th):
    print(f"  {'block':<6}{'gamma':>9}{'lx':>9}{'ly':>9}")
    for i, name in enumerate(PARAM_BLOCKS):
        g, lx, ly = float(th[3 * i]), float(jnp.exp(th[3 * i + 1])), float(jnp.exp(th[3 * i + 2]))
        x, sl = onp.asarray(th[3 * i:3 * i + 3]), slice(3 * i, 3 * i + 3)
        at = onp.isclose(x, LO[sl], atol=1e-6) | onp.isclose(x, HI[sl], atol=1e-6)
        flag = "  <- at bound" if at.any() else ""
        print(f"  {name:<6}{g:>9.3f}{lx:>9.3f}{ly:>9.3f}{flag}")


# ======================================================================== setup
cfg = dict(kernel=args.kernel, dt=Δt, M=M, seed=SEED, eps=EPS, maxiter=args.maxiter,
           tol=args.tol, step=args.nm_step, lu=args.lu_bounds, lp=args.lp_bounds,
           g=args.gamma_bounds, l0=args.l0, lp0=args.lp0, lstd=args.l_prior_std)
TAG = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
print(f"[config] tag {TAG}  seed={SEED}  kernel={args.kernel}  n_theta={theta_init.size}"
      f"  ntrain(step 1)={n_train(R_s.shape[0])}  ntrain(n>1)={n_train(M)}  dt={Δt}  NM maxiter={args.maxiter}  refit_every={args.refit_every}")

NX, NY = 120, 41
TEST_MARGIN = min(args.test_margin, 0.98) if args.fem else args.test_margin
XX, EE = jnp.meshgrid(jnp.linspace(0.0, L, NX),
                      jnp.linspace(-TEST_MARGIN, TEST_MARGIN, NY) * 0.5, indexing="ij")
YY = EE * width(XX)
R_test = jnp.stack([XX.ravel(), YY.ravel()], -1)

SNAP_STEPS = sorted({max(1, int(round(k * args.n_loop / args.n_snap)))
                     for k in range(1, args.n_snap + 1)})

F1_test = fem_good = None
if args.fem:
    f1, _, _ = fem_stokes.get(onp.asarray(R_test), OUTDIR / "fem_reference.npz",
                              refit=args.fem_refit, L=L, a=a, avg_width=avg_width, eta=η,
                              body_force_x=float(FBODY[0]), nx=args.fem_nx, ny=args.fem_ny)
    F1_test = onp.asarray(f1).reshape(NX, NY)
    fem_good = onp.isfinite(F1_test)

# ========================================================================= march
key = jrd.PRNGKey(SEED)
R_G = R_s                              # initial data {x^0, u^0}: the N_f grid
N0 = R_G.shape[0]
u1_prev = jnp.zeros(N0)                # t = 0: rest, known exactly
u2_prev = jnp.zeros(N0)
Sigma_prev = jnp.zeros((2 * N0, 2 * N0))
theta = theta_init
history, theta_hist, snapshots = [], [], []
u1_test_prev = onp.zeros(NX * NY)

HEADER = (f"{'step':>5}{'t':>7}{'ntrain':>7}{'nlp':>12}{'it':>5}{'fev':>6}{'ux_max':>10}{'ux_ctr':>10}"
          f"{'uy_absmax':>11}{'|du|':>10}{'std_max':>10}{'sec':>7}")
print("\n" + HEADER)

for n in range(1, args.n_loop + 1):
    t_wall = time.time()
    y = make_y(u1_prev, u2_prev)

    # 1) hyperparameters (step 1 from theta_init, then warm start)
    nit = nfev = 0
    if n == 1 or (args.refit_every > 0 and (n - 1) % args.refit_every == 0):
        theta, _, nit, nfev, ok = fit_theta(theta, y, R_G, args.maxiter, f"step {n}")
        if not ok:
            print(f"      [step {n}] no improvement / non-finite -> kept previous theta")
        if any(float(theta[i]) <= args.gamma_bounds[0] + 1e-6 for i in U_GAMMA_IDX):
            print(f"      [warn] step {n}: velocity amplitude at lower bound -> u collapsing")
    nlp = float(neg_log_posterior(theta, y, R_G))

    # 2) factorize
    Lc, alpha = factorize(theta, y, R_G)
    if not bool(jnp.all(jnp.isfinite(Lc))):
        raise FloatingPointError(f"Cholesky failed at step {n}; increase --jitter")

    # 3) marginalized posterior on the test grid
    u1_mean, u1_std = predict(Lc, alpha, theta, R_test, R_G, Sigma_prev, "u1")
    u2_mean, u2_std = predict(Lc, alpha, theta, R_test, R_G, Sigma_prev, "u2")
    p_mean, p_std = predict(Lc, alpha, theta, R_test, R_G, Sigma_prev, "p")

    # 4) new random locations and marginalized artificial data {x^n, u^n}
    key, kn = jrd.split(key)
    R_G_next = sample_artificial(kn)
    u1_next, u2_next, Sigma_next = predict_artificial(Lc, alpha, theta, R_G_next, R_G,
                                                      Sigma_prev)

    # ---- diagnostics
    U1 = onp.asarray(u1_mean).reshape(NX, NY)
    U2 = onp.asarray(u2_mean).reshape(NX, NY)
    du = float(onp.abs(onp.asarray(u1_mean) - u1_test_prev).max())
    u1_test_prev = onp.asarray(u1_mean)
    t_now = n * Δt
    if args.nm_print:
        print(HEADER)
    print(f"{n:>5}{t_now:>7.3f}{y.shape[0]:>7d}{nlp:>12.4e}{nit:>5}{nfev:>6}{U1.max():>10.5f}"
          f"{U1[:, NY // 2].mean():>10.5f}{onp.abs(U2).max():>11.2e}{du:>10.2e}"
          f"{float(u1_std.max()):>10.2e}{time.time() - t_wall:>7.1f}", flush=True)

    history.append(dict(step=n, t=t_now, nlp=nlp, ux_max=float(U1.max()),
                        ux_ctr=float(U1[:, NY // 2].mean()), ux_wall=float(U1[:, 0].mean()),
                        uy_absmax=float(onp.abs(U2).max()), dmax=du,
                        std_max=float(u1_std.max())))
    theta_hist.append(onp.asarray(theta))

    if n in SNAP_STEPS:
        snapshots.append((n, t_now, U1.copy(), U2.copy()))
        if F1_test is not None:
            good = fem_good & onp.isfinite(U1)
            rel = onp.linalg.norm(U1[good] - F1_test[good]) / onp.linalg.norm(F1_test[good])
            print(f"      [fem] t={t_now:.3f}  rel L2 vs steady = {rel:.4f}")
            history[-1]["fem_rel_l2"] = rel

    if n == args.n_loop or (args.plot_every and n % args.plot_every == 0):
        ctx = plotting.PlotCtx(XX=onp.asarray(XX), YY=onp.asarray(YY), NX=NX, NY=NY,
                               L=L, a=a, avg_width=avg_width, R_wall=onp.asarray(R_wall),
                               outdir=OUTDIR, tag=f"{TAG}_t{n:04d}", subdir="base",
                               sparse_pts=onp.asarray(R_G))
        plt.close(plotting.plot_all(ctx, {"u_x": (u1_mean, u1_std), "u_y": (u2_mean, u2_std),
                                          "p": (p_mean, p_std)}, fem=None))

    onp.savez(OUTDIR / "states" / f"state_{TAG}_t{n:04d}.npz",
              t=t_now, seed=SEED, theta=onp.asarray(theta), nlp=nlp,
              u1=U1, u2=U2, p=onp.asarray(p_mean).reshape(NX, NY),
              u1_std=onp.asarray(u1_std).reshape(NX, NY),
              u2_std=onp.asarray(u2_std).reshape(NX, NY),
              p_std=onp.asarray(p_std).reshape(NX, NY),
              R_G=onp.asarray(R_G), R_G_next=onp.asarray(R_G_next),
              u1_artificial=onp.asarray(u1_next), u2_artificial=onp.asarray(u2_next),
              Sigma_artificial=onp.asarray(Sigma_next))

    u1_prev, u2_prev, Sigma_prev, R_G = u1_next, u2_next, Sigma_next, R_G_next

print("\n[theta] final")
report_theta(theta)

# ======================================================================= summary
KEYS = sorted({k for h in history for k in h})
H = {k: onp.array([h.get(k, onp.nan) for h in history], dtype=float) for k in KEYS}
TH = onp.stack(theta_hist)

fig, ax = plt.subplots(1, 4, figsize=(20, 4))
ax[0].plot(H["t"], H["ux_ctr"], "o-", label="centreline mean")
ax[0].plot(H["t"], H["ux_max"], "s-", ms=3, label="domain max")
ax[0].set(xlabel="$t$", ylabel="$u_x$", title="spin-up from rest"); ax[0].legend(fontsize=8)
ax[1].semilogy(H["t"], onp.maximum(H["dmax"], 1e-16), "o-")
ax[1].set(xlabel="$t$", ylabel=r"$\max|u_x^n-u_x^{n-1}|$", title="step-to-step change")
ax[2].semilogy(H["t"], onp.abs(H["ux_wall"]) + 1e-16, "o-")
ax[2].set(xlabel="$t$", ylabel=r"$|u_x|$ at wall row", title="no-slip residual")
for i, name in enumerate(PARAM_BLOCKS):
    l, = ax[3].plot(H["t"], TH[:, 3 * i], "-", label=f"{name} gamma")
    ax[3].plot(H["t"], TH[:, 3 * i + 1], "--", color=l.get_color())
    ax[3].plot(H["t"], TH[:, 3 * i + 2], ":", color=l.get_color())
ax[3].set(xlabel="$t$", ylabel="theta", title="gamma (solid), log lx (--), log ly (:)")
ax[3].legend(fontsize=7)
for a_ in ax:
    a_.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTDIR / f"march_summary_{TAG}.png", dpi=140, bbox_inches="tight")
onp.savez(OUTDIR / f"march_history_{TAG}.npz", theta=TH, seed=SEED, **H)

fem_results = None
if args.fem:
    fem_results = fem_stokes.get_unsteady(
        onp.asarray(R_test), SNAP_STEPS, OUTDIR / "fem_unsteady_reference.npz",
        refit=args.fem_refit, L=L, a=a, avg_width=avg_width, eta=η, rho=ρ,
        body_force_x=float(FBODY[0]), dt=Δt, nx=args.fem_nx, ny=args.fem_ny)

evolution_plot.plot_evolution_vs_fem(
    snapshots, OUTDIR, XX=onp.asarray(XX), YY=onp.asarray(YY), L=L, a=a,
    avg_width=avg_width, fem=fem_results, cmap=args.evolution_cmap, tag=TAG, dt=Δt,
    n_artificial=M)
plt.show()