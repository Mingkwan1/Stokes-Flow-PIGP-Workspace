"""Steady Stokes-flow PIGP reference, imported by the unsteady main script.

Solves the STEADY problem with the same kernels, hyper-prior, jitter and
training points as the unsteady march, but without time stepping:

    grad p - eta*lap(u) = F,   div(u) = 0,   no-slip walls,   x-periodic

Blocks: u1, u2, Su1, Su2, Sp, f1, f2, s   (f1/f2 replace the march's G1/G2)

The result (u_x, u_y, p and their std on the test grid) is the PIGP's own
steady state. The unsteady PIGP march is compared against it to find
t*_PIGP, in the same way the FEM march is compared against the steady FEM.

Caching (independent of dt / n_loop, so one fit serves every run):
  cache_dir/steady_pigp_<fp>.npz
    theta                      -- keyed by the training fingerprint <fp>
    u1, u2, p, u1_std, u2_std  -- reused only if the test grid hash matches;
                                  otherwise re-predicted from cached theta
"""
import hashlib
import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import jaxopt
import numpy as onp

_VERSION = 1   # bump to invalidate every cache if the model changes


# ============================================================ utilities
def _hash(*arrays):
    h = hashlib.sha256()
    for A in arrays:
        h.update(onp.ascontiguousarray(onp.asarray(A), dtype=onp.float64).tobytes())
    return h.hexdigest()[:16]


def rel_l2_velocity(u1, u2, u1_ref, u2_ref):
    """||(u1,u2) - (u1_ref,u2_ref)||_2 / ||(u1_ref,u2_ref)||_2 over points
    where every array is finite. Pass u2=u2_ref=None for u_x only."""
    a = [onp.asarray(u1, float).ravel()]
    b = [onp.asarray(u1_ref, float).ravel()]
    if u2 is not None:
        a.append(onp.asarray(u2, float).ravel())
        b.append(onp.asarray(u2_ref, float).ravel())
    good = onp.ones_like(a[0], dtype=bool)
    for arr in a + b:
        good &= onp.isfinite(arr)
    num = sum(onp.sum((x[good] - y[good])**2) for x, y in zip(a, b))
    den = sum(onp.sum(y[good]**2) for y in b)
    return float(onp.sqrt(num / den))


# ============================================================ kernels
def _se(r, rp, th):
    return jnp.exp(th[0] - 0.5*((r[0]-rp[0])/jnp.exp(th[1]))**2
                         - 0.5*((r[1]-rp[1])/jnp.exp(th[2]))**2)


def _build_kernels(eta, L, zero_up):
    th = lambda t, i: jax.lax.dynamic_slice(t, (3*i,), (3,))
    D    = lambda k, c: None if k is None else (lambda r, rp, t: jax.grad(k, 0)(r, rp, t)[c])
    Dp   = lambda k, c: None if k is None else (lambda r, rp, t: jax.grad(k, 1)(r, rp, t)[c])
    Lap  = lambda k: None if k is None else (lambda r, rp, t: jnp.trace(jax.hessian(k, 0)(r, rp, t)))
    Lapp = lambda k: None if k is None else (lambda r, rp, t: jnp.trace(jax.hessian(k, 1)(r, rp, t)))
    mul  = lambda c, k: None if k is None else (lambda r, rp, t: c*k(r, rp, t))

    def add(*ks):
        ks = [k for k in ks if k is not None]
        if not ks:
            return None
        if len(ks) == 1:
            return ks[0]
        return lambda r, rp, t: sum(k(r, rp, t) for k in ks)

    eL = jnp.array([L, 0.0])
    S  = lambda k: None if k is None else (lambda r, rp, t: k(r, rp, t) - k(r + eL, rp, t))
    Sp = lambda k: None if k is None else (lambda r, rp, t: k(r, rp, t) - k(r, rp + eL, t))

    k_uu = [[lambda r, rp, t: _se(r,  rp, th(t, 0)), lambda r, rp, t: _se(r, rp, th(t, 1))],
            [lambda r, rp, t: _se(rp, r,  th(t, 1)), lambda r, rp, t: _se(r, rp, th(t, 2))]]
    k_up = [lambda r, rp, t: _se(r, rp, th(t, 3)), lambda r, rp, t: _se(r, rp, th(t, 4))]
    if zero_up:
        k_up = [None, None]
    k_pu = k_up
    k_pp = lambda r, rp, t: _se(r, rp, th(t, 5))

    k_uf = lambda a, b: add(Dp(k_up[a], b), mul(-eta, Lapp(k_uu[a][b])))
    k_us = lambda a: add(*[Dp(k_uu[a][b], b) for b in range(2)])
    k_pf = lambda a: add(Dp(k_pp, a), mul(-eta, Lapp(k_pu[a])))
    k_ps = add(*[Dp(k_pu[b], b) for b in range(2)])
    k_ff = lambda a, b: add(D(k_pf(b), a), mul(-eta, Lap(k_uf(a, b))))
    k_fs = lambda a: add(D(k_ps, a), mul(-eta, Lap(k_us(a))))
    k_ss = add(*[D(k_us(a), a) for a in range(2)])

    kfn = {
        ("u1","u1"): k_uu[0][0], ("u1","u2"): k_uu[0][1], ("u2","u2"): k_uu[1][1],
        ("u1","Su1"): Sp(k_uu[0][0]), ("u1","Su2"): Sp(k_uu[0][1]),
        ("u2","Su1"): Sp(k_uu[1][0]), ("u2","Su2"): Sp(k_uu[1][1]),
        ("u1","Sp"): Sp(k_up[0]), ("u2","Sp"): Sp(k_up[1]),
        ("Su1","Su1"): Sp(S(k_uu[0][0])), ("Su1","Su2"): Sp(S(k_uu[0][1])),
        ("Su2","Su2"): Sp(S(k_uu[1][1])),
        ("Su1","Sp"): Sp(S(k_up[0])), ("Su2","Sp"): Sp(S(k_up[1])),
        ("Sp","Sp"): Sp(S(k_pp)),
        ("u1","f1"): k_uf(0,0), ("u1","f2"): k_uf(0,1),
        ("u2","f1"): k_uf(1,0), ("u2","f2"): k_uf(1,1),
        ("u1","s"): k_us(0), ("u2","s"): k_us(1),
        ("Su1","f1"): S(k_uf(0,0)), ("Su1","f2"): S(k_uf(0,1)),
        ("Su2","f1"): S(k_uf(1,0)), ("Su2","f2"): S(k_uf(1,1)),
        ("Su1","s"): S(k_us(0)), ("Su2","s"): S(k_us(1)),
        ("Sp","f1"): S(k_pf(0)), ("Sp","f2"): S(k_pf(1)), ("Sp","s"): S(k_ps),
        ("f1","f1"): k_ff(0,0), ("f1","f2"): k_ff(0,1), ("f2","f2"): k_ff(1,1),
        ("f1","s"): k_fs(0), ("f2","s"): k_fs(1),
        ("s","s"): k_ss,
    }
    rows = {
        "u1": [k_uu[0][0], k_uu[0][1], Sp(k_uu[0][0]), Sp(k_uu[0][1]), Sp(k_up[0]),
               k_uf(0,0), k_uf(0,1), k_us(0)],
        "u2": [k_uu[1][0], k_uu[1][1], Sp(k_uu[1][0]), Sp(k_uu[1][1]), Sp(k_up[1]),
               k_uf(1,0), k_uf(1,1), k_us(1)],
        "p":  [k_pu[0], k_pu[1], Sp(k_pu[0]), Sp(k_pu[1]), Sp(k_pp),
               k_pf(0), k_pf(1), k_ps],
    }
    kstar = {"u1": k_uu[0][0], "u2": k_uu[1][1], "p": k_pp}
    return kfn, rows, kstar


def _kmat(k):
    if k is None:
        return lambda R1, R2, t: jnp.zeros((R1.shape[0], R2.shape[0]))
    return jax.jit(jax.vmap(jax.vmap(k, (None, 0, None)), (0, None, None)))


# ============================================================ model
class _SteadyPIGP:
    def __init__(self, blocks, kfn, rows, kstar, eps_jitter, zero_up):
        self.blocks, self.kfn, self.rows, self.kstar = blocks, kfn, rows, kstar
        self.eps = eps_jitter
        dead = (3, 4) if zero_up else ()
        self.active = jnp.array([i for i in range(18) if (i // 3) not in dead])
        self.n = sum(R.shape[0] for _, R in blocks)

    def K(self, theta):
        nb = len(self.blocks)
        M = [[None]*nb for _ in range(nb)]
        for i in range(nb):
            for j in range(i, nb):
                (ni, Ri), (nj, Rj) = self.blocks[i], self.blocks[j]
                M[i][j] = _kmat(self.kfn[(ni, nj)])(Ri, Rj, theta)
                M[j][i] = M[i][j].T
        return jnp.block(M) + (self.eps**2)*jnp.eye(self.n)

    def nlp(self, theta, y):
        Lc = jnp.linalg.cholesky(self.K(theta))
        v = solve_triangular(Lc, y, lower=True)
        return (0.5*jnp.dot(v, v) + jnp.sum(jnp.log(jnp.diag(Lc)))
                + 0.5*self.n*jnp.log(2.0*jnp.pi)
                + jnp.sum(theta[self.active]))           # Jeffreys prior, as in main

    def predict(self, theta, y, R_star, star, chunk=2000):
        Lc = jnp.linalg.cholesky(self.K(theta))
        alpha = solve_triangular(Lc.T, solve_triangular(Lc, y, lower=True), lower=False)
        kd = jax.vmap(lambda r: self.kstar[star](r, r, theta))
        means, stds = [], []
        for s in range(0, R_star.shape[0], chunk):
            Rs = R_star[s:s+chunk]
            Q = jnp.concatenate([_kmat(k)(Rs, R, theta)
                                 for k, (_, R) in zip(self.rows[star], self.blocks)], axis=1)
            V = solve_triangular(Lc, Q.T, lower=True)
            means.append(Q @ alpha)
            stds.append(jnp.sqrt(jnp.clip(kd(Rs) - jnp.sum(V**2, axis=0), 0.0)))
        return onp.asarray(jnp.concatenate(means)), onp.asarray(jnp.concatenate(stds))


def _nelder_mead(fun, x0, maxiter, tol, verbose):
    it = [0]

    def cb(xk):
        it[0] += 1
        if verbose and (it[0] % 10 == 0 or it[0] == 1):
            print(f"  [steady-PIGP] NM {it[0]:<5} nlp = {float(fun(xk)):.6e}", flush=True)

    res = jaxopt.ScipyMinimize(fun=fun, method="Nelder-Mead", tol=tol, maxiter=maxiter,
                               callback=cb, options={"adaptive": True}).run(x0)
    return res.params


# ============================================================ public API
def get_steady_reference(R_test, *, R_wall, R_dSu1, R_dSu2, R_sp, R_f, R_s,
                         body_force, eta, L, a, avg_width, theta_init,
                         eps_jitter, zero_up, nm_iter, nm_tol, cache_dir,
                         refit=False, verbose=True):
    """Steady PIGP mean/std of u_x, u_y, p at R_test (cached).

    Returns dict with 1-D arrays u1, u2, p, u1_std, u2_std (length
    R_test.shape[0]), plus theta, nlp and fingerprint.
    """
    R_test = jnp.asarray(R_test)
    blocks = [("u1", R_wall), ("u2", R_wall), ("Su1", R_dSu1), ("Su2", R_dSu2),
              ("Sp", R_sp), ("f1", R_f), ("f2", R_f), ("s", R_s)]
    body_force = onp.asarray(body_force, float)
    y = jnp.concatenate([
        jnp.zeros(R_wall.shape[0]), jnp.zeros(R_wall.shape[0]),     # no-slip
        jnp.zeros(R_dSu1.shape[0]), jnp.zeros(R_dSu2.shape[0]),     # periodic u
        jnp.zeros(R_sp.shape[0]),                                   # periodic p
        jnp.full(R_f.shape[0], body_force[0]),                      # f1 = F_x
        jnp.full(R_f.shape[0], body_force[1]),                      # f2 = F_y
        jnp.zeros(R_s.shape[0]),                                    # div u = 0
    ])

    cfg = {"v": _VERSION, "eta": float(eta), "L": float(L), "a": float(a),
           "avg_width": float(avg_width), "F": body_force.tolist(),
           "eps": float(eps_jitter), "zero_up": bool(zero_up),
           "theta_init": [float(v) for v in onp.asarray(theta_init)],
           "nm_iter": int(nm_iter), "nm_tol": float(nm_tol),
           "points": _hash(R_wall, R_dSu1, R_dSu2, R_sp, R_f, R_s)}
    fp = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    test_hash = _hash(R_test)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"steady_pigp_{fp}.npz"

    d = dict(onp.load(path, allow_pickle=False)) if (path.exists() and not refit) else None
    if d is not None and str(d.get("test_hash", "")) == test_hash:
        if verbose:
            print(f"[steady-PIGP] loaded reference from {path}")
        return {k: d[k] for k in ("u1", "u2", "p", "u1_std", "u2_std", "theta", "nlp")} | {"fingerprint": fp}

    kfn, rows, kstar = _build_kernels(float(eta), float(L), bool(zero_up))
    model = _SteadyPIGP(blocks, kfn, rows, kstar, float(eps_jitter), bool(zero_up))
    obj = jax.jit(lambda th: model.nlp(th, y))

    if d is not None:
        theta = jnp.asarray(d["theta"])
        if verbose:
            print(f"[steady-PIGP] loaded theta from {path}; re-predicting on new test grid")
    else:
        if verbose:
            print(f"[steady-PIGP] fitting steady reference (ntrain={model.n}, fp={fp}) ...")
            print(f"  [steady-PIGP] initial nlp = {float(obj(jnp.asarray(theta_init))):.6e}", flush=True)
        t0 = time.time()
        theta = _nelder_mead(obj, jnp.asarray(theta_init), nm_iter, nm_tol, verbose)
        if verbose:
            print(f"  [steady-PIGP] final nlp = {float(obj(theta)):.6e}  [{time.time()-t0:.0f}s]")

    u1, u1_std = model.predict(theta, y, R_test, "u1")
    u2, u2_std = model.predict(theta, y, R_test, "u2")
    p,  _      = model.predict(theta, y, R_test, "p")
    p = p - onp.mean(p)                       # pressure only defined up to a constant
    out = {"u1": u1, "u2": u2, "p": p, "u1_std": u1_std, "u2_std": u2_std,
           "theta": onp.asarray(theta), "nlp": onp.asarray(float(obj(theta)))}
    onp.savez(path, **out, fingerprint=onp.asarray(fp), test_hash=onp.asarray(test_hash))
    if verbose:
        print(f"[steady-PIGP] u_x max = {u1.max():.5f}   max|u_y| = {onp.abs(u2).max():.5f}")
        print(f"[steady-PIGP] wrote {path}")
    return out | {"fingerprint": fp}