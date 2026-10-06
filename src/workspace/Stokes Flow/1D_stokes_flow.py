from functools import wraps
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jrd

from jax.scipy.linalg import solve_triangular

import optax
import optax.tree

import numpy as onp

import matplotlib.pyplot as plt 

import argparse
import hashlib
import json
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = OUTDIR / "final_params.npz"

parser = argparse.ArgumentParser(description="1D Stokes flow PIGP")
parser.add_argument("--fit", action="store_true",
                    help="force re-optimization even if a valid cache exists")
parser.add_argument("--no-cache", action="store_true",
                    help="do not read or write the cache")
parser.add_argument("--max-iter", type=int, default=300)
parser.add_argument("--nm-iter", type=int, default=100)
parser.add_argument("--tol", type=float, default=1e-4)
args = parser.parse_args()

@wraps(onp.subtract.outer)
def subtract_outer(a,b,out=None):
    if out:
        raise NotImplementedError("The 'out' argument to outer is not supported")
    dtype = jnp.result_type(a, b)
    a = jnp.asarray(a, dtype=dtype)
    b = jnp.asarray(b, dtype=dtype)

    return jnp.ravel(a)[:,None] - jnp.ravel(b)

def K_Linear(x1, x2, l=1.0):
    return jnp.outer(x1, x2)

def Squared_Exponential_Kernel(t_slice):
    alpha, log_l = t_slice[0], (t_slice[1])
    return lambda x1, x2: jnp.exp(alpha - 0.5*((x1- x2)/jnp.exp(log_l))**2)

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
FBODY = -ΔP/L # Body force
MARGIN = 0.99


## 1.) velocity at bounday u_b (62 points)

x_u = jnp.linspace(0.0, L, 2)

u = jnp.zeros_like(x_u)

## 4.) Governing equations f (340 points)

N_f = N_s = 340
x_f = jnp.linspace(0.0, L, N_f)
x_s = x_f 

# x_f, y_f = points_grid(N=N_f)

x_f = x_f

f = jnp.full(N_f, -ΔP / L)

## 5.) Governing equations s (337 points)

x_s= x_f
s = jnp.zeros(N_s)

###

theta = lambda t, i: jax.lax.dynamic_slice(t, (2*i,), (2,))

D    = lambda k: lambda r, rp, t: jax.grad(k, 0)(r, rp, t)
Dp = lambda k: lambda r, rp, t: jax.grad(k, 1)(r, rp, t)
Lap  = lambda k:    lambda r, rp, t: jnp.trace(jnp.atleast_2d(jax.hessian(k, 0)(r, rp, t)))
Lap_p = lambda k: lambda r, rp, t: jnp.trace(jnp.atleast_2d(jax.hessian(k, 1)(r, rp, t)))
add  = lambda *ks:  lambda r, rp, t: sum(k(r, rp, t) for k in ks)
mul  = lambda c, k: lambda r, rp, t: c * k(r, rp, t)

### Base kernel (Total of 6 kernels and 18 parameters)

k_uu = lambda r, rp, t: Squared_Exponential_Kernel(theta(t, 0))(r, rp)

k_up = lambda r, rp, t: Squared_Exponential_Kernel(theta(t, 1))(r, rp)
# k_pu = k_up

k_pp = lambda r, rp, t: Squared_Exponential_Kernel(theta(t, 2))(r, rp)

### Derived kernels 


k_uf = add(Dp(k_up), mul(-η, Lap_p(k_uu)))

k_pf = add(Dp(k_pp), mul(-η, Lap_p(k_up)))

k_ff = add(D(k_pf), mul(-η, Lap(k_uf)))


## Buiding K

def kmat(k):
    return jax.jit(jax.vmap(jax.vmap(k, (None, 0, None)), (0, None, None)))

# ---- block order: pick once, matches paper eqs. (16)-(19) ------------------
blocks = [
    ("u", x_u),   
    ("f", x_f),    
    ]

# ---- lookup table: kernel fn for every ordered pair of block names ----------
kfn = {
    ("u","u"): k_uu,           
    ("u","f"): k_uf, 
    ("f","f"): k_ff,
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
#    0.8, jnp.log(0.60), jnp.log(0.35),   # u1-u2  (weak cross-correlation)
#     1.1, jnp.log(0.55), jnp.log(0.32),   # u2-u2
#    0.0, jnp.log(0.70), jnp.log(0.40),   # u1-p
#    0.0, jnp.log(0.65), jnp.log(0.38),   # u2-p
#     0.7, jnp.log(0.80), jnp.log(0.45),   # p-p
# ])

theta_init = jnp.array([
    1.2, -1.2,   # u-u
   0.0, -1.2,   # u-p
    1.2, -1.2,    # p-p
])
EPS_JITTER = 1e-3

y = jnp.concatenate([
    u,                      # (2,)  no-slip, top wall
    f,                       # (340,) = 0
])
# print(y)
ntrain = y.shape[0]    

K = build_K(theta_init) + (EPS_JITTER**2) * jnp.eye(ntrain)
print("symmetric:", jnp.allclose(K, K.T, atol=1e-8))
print("min eig:", float(jnp.linalg.eigvalsh(K).min()))

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

@jax.jit
def neg_log_posterior(theta):
    K = build_K(theta) + (EPS_JITTER**2) * jnp.eye(ntrain)
    Lc = jnp.linalg.cholesky(K)
    v  = solve_triangular(Lc, y, lower=True)
    nlp = (0.5*jnp.dot(v, v) + jnp.sum(jnp.log(jnp.diag(Lc))) + 0.5*ntrain*jnp.log(2.0*jnp.pi))
    return nlp

class InfoState(NamedTuple):
  iter_num: jax.typing.ArrayLike


def print_info():
  def init_fn(params):
    del params
    return InfoState(iter_num=0)

  def update_fn(updates, state, params, *, value, grad, **extra_args):
    del extra_args

    jax.debug.print(
        'Iteration: {i}, Value: {v}, Gradient norm: {e}, theta: {p}',
        i=state.iter_num,
        v=value,
        e=optax.tree.norm(grad),
        p=params,
    )
    return updates, InfoState(iter_num=state.iter_num + 1)

  return optax.GradientTransformationExtraArgs(init_fn, update_fn)

def run_opt(init_params, fun, opt, max_iter, tol):
  value_and_grad_fun = optax.value_and_grad_from_state(fun)

  def step(carry):
    params, state = carry
    value, grad = value_and_grad_fun(params, state=state)
    updates, state = opt.update(
        grad, state, params, value=value, grad=grad, value_fn=fun
    )
    params = optax.apply_updates(params, updates)
    return params, state

  def continuing_criterion(carry):
    _, state = carry
    iter_num = optax.tree.get(state, 'count')
    grad = optax.tree.get(state, 'grad')
    err = optax.tree.norm(grad)
    return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

  init_carry = (init_params, opt.init(init_params))
  final_params, final_state = jax.lax.while_loop(
      continuing_criterion, step, init_carry
  )
  return final_params, final_state

opt = optax.chain(print_info(), optax.lbfgs())

fun = neg_log_posterior

# v0, g0 = jax.value_and_grad(neg_log_posterior)(theta_init)
# print(v0, jnp.linalg.norm(g0), jnp.any(jnp.isnan(g0)))

import jaxopt

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

# print(
#     f'Initial value: {fun(theta_init):.2e} '
#     f'Initial gradient norm: {optax.tree.norm(jax.grad(fun)(theta_init)):.2e}'
# )
# th_nm, state_nm = run_nelder_mead(theta_init, fun, maxiter=100)

# final_params, _ = run_opt(th_nm, fun, opt, max_iter=300, tol=1e-4)
# print(
#     f'Final value: {fun(final_params):.2e}, '
#     f'Final gradient norm: {optax.tree.norm(jax.grad(fun)(final_params)):.2e}'
# )

def config_fingerprint():
    """Hash everything that changes the optimum. A cache keyed only on a
    filename would silently return theta fitted for different physics."""
    cfg = {
        "eta": float(η), "L": float(L), "dP": float(ΔP), "a": float(a),
        "avg_width": float(avg_width), "eps": float(EPS_JITTER),
        "N_f": int(N_f), "N_s": int(N_s), "n_u": int(x_u.shape[0]),
        "blocks": [name for name, _ in blocks],
        "theta_init": [float(v) for v in theta_init],
        "x_f": [float(x_f[0]), float(x_f[-1]), int(x_f.shape[0])],
        "x_u": [float(v) for v in x_u],
        "y_sum": float(jnp.sum(y)),
    }
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    return h, cfg


def load_cache(fingerprint):
    if not CACHE_PATH.exists():
        return None
    d = onp.load(CACHE_PATH, allow_pickle=False)
    if str(d["fingerprint"]) != fingerprint:
        print(f"[cache] fingerprint mismatch (config changed) -> refitting")
        return None
    return jnp.asarray(d["theta"])


def save_cache(fingerprint, theta, value):
    onp.savez(CACHE_PATH,
              theta=onp.asarray(theta),
              fingerprint=onp.asarray(fingerprint),
              value=onp.asarray(float(value)))
    print(f"[cache] wrote {CACHE_PATH}")


def fit():
    print(f"Initial value: {fun(theta_init):.2e} "
          f"Initial gradient norm: {optax.tree.norm(jax.grad(fun)(theta_init)):.2e}")
    th_nm, _ = run_nelder_mead(theta_init, fun, maxiter=args.nm_iter)
    th, _ = run_opt(th_nm, fun, opt, max_iter=args.max_iter, tol=args.tol)
    return th


fp, cfg = config_fingerprint()
print(f"[config] fingerprint {fp}")

final_params = None
if not args.fit and not args.no_cache:
    final_params = load_cache(fp)
    if final_params is not None:
        print(f"[cache] loaded theta from {CACHE_PATH} (skipping optimization)")

if final_params is None:
    final_params = fit()
    if not args.no_cache:
        save_cache(fp, final_params, fun(final_params))

print(f"Final value: {fun(final_params):.2e}, "
      f"Final gradient norm: {optax.tree.norm(jax.grad(fun)(final_params)):.2e}")
print(f"theta = {onp.asarray(final_params)}")
#### Predidcting the final value

x_test = jnp.linspace(0,L, 500)

u_exact = (-FBODY / (2*η)) * ((L/2- onp.asarray(x_test))**2)+9.4

kfn_star = {
    ("u","u"): k_uu,
    ("u","f"): k_uf,
}

def build_K_star_obs(theta, x_star, star="u"):
    """Q, shape (len(x_star), ntrain). Columns follow `blocks` order."""
    cols = [kmat(kfn_star[(star, name)])(x_star, R, theta) for name, R in blocks]
    return jnp.concatenate(cols, axis=1)

def build_K_star_star(theta, x_star, star="u"):
    """K**, shape (len(x_star), len(x_star))."""
    return kmat(kfn_star[(star, star)])(x_star, x_star, theta)

@jax.jit
def predict(theta, x_star, star="u"):
    K  = build_K(theta) + (EPS_JITTER**2) * jnp.eye(ntrain)
    Lc = jnp.linalg.cholesky(K)
    Q  = build_K_star_obs(theta, x_star, star)

    alpha = solve_triangular(Lc.T, solve_triangular(Lc, y, lower=True), lower=False)
    mean  = Q @ alpha                                   # (n_star,)

    V   = solve_triangular(Lc, Q.T, lower=True)         # (ntrain, n_star)
    cov = build_K_star_star(theta, x_star, star) - V.T @ V
    std = jnp.sqrt(jnp.clip(jnp.diag(cov), 0.0))
    return mean, std

u_mean, u_std = predict(final_params, x_test)

# with open("theta_log.csv", "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=["iter", "loss"] + param_names)
#     writer.writeheader()
#     writer.writerows(log_rows)

# print("\nFinal hyperparameters:")
# for i, name in enumerate(param_names):
#     val = final_params[i]
#     if "log_l" in name:
#         print(f"  {name.replace('log_','')} = {float(jnp.exp(val)):.4f}  (raw log = {float(val):.4f})")
#     else:
#         print(f"  {name} = {float(val):.4f}")

xt = onp.asarray(x_test)
m  = onp.asarray(u_mean)
sd = onp.asarray(u_std)

fig, ax = plt.subplots(figsize=(9, 5))

# --- posterior mean + 2σ credible band -------------------------------------
ax.fill_between(xt, m - 2*sd, m + 2*sd,
                alpha=0.25, color="tab:blue", lw=0,
                label=r"posterior $\pm 2\sigma$")
ax.plot(xt, m, color="tab:blue", lw=2, label=r"predicted $u(x)$")

# --- boundary training points (these ARE velocities) -----------------------
ax.scatter(onp.asarray(x_u), onp.asarray(u),
           s=90, c="k", marker="o", zorder=5,
           label=f"$u_b$ training ({x_u.shape[0]} pts)")

# --- collocation locations for f and s (NOT velocities -> shown as rugs) ---
ylo, yhi = ax.get_ylim()
span = yhi - ylo
ax.plot(onp.asarray(x_f), onp.full(x_f.shape[0], ylo + 0.03*span),
        "|", ms=10, c="tab:red", alpha=0.6,
        label=f"$f$ collocation ({N_f} pts)")
ax.plot(onp.asarray(x_s), onp.full(x_s.shape[0], ylo + 0.08*span),
        "|", ms=10, c="tab:green", alpha=0.6,
        label=f"$s$ collocation ({N_s} pts)")
ax.plot(onp.asarray(x_test), u_exact, "k--", lw=1.5, label="analytic")
ax.set_xlabel("cross-channel position $y$")
ax.set_xlabel("distance $x$")
ax.set_ylabel("velocity $u$")
ax.set_title("1D Stokes: GP posterior velocity vs. training points")
ax.axhline(0.0, color="gray", lw=0.6, ls=":")
ax.legend(loc="best", framealpha=0.9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("u_prediction.png", dpi=150)
plt.show()

plot_path = OUTDIR / f"plots/1D_u_prediction_{fp}.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
fig.savefig(OUTDIR / "u_prediction_latest.png", dpi=150, bbox_inches="tight")
print(f"[plot] saved {plot_path}")

if not args.no_show:
    plt.show()
plt.close(fig)