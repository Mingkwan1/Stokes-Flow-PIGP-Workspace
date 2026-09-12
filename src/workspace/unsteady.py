import time, sys
from functools import partial
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as onp

# ------------------------------------------------------------------ physics
eta = 1.0
Q = 1.0
L = 2.5
a = 0.2
avg_width = 1.0
dP = -30.0
FBODY = -dP / L * jnp.array([1.0, 0.0])     # [12.0, 0.0]
MARGIN = 0.999
EPS_JITTER = 1e-3
DT = 0.01

def width(x):
    return avg_width + 2 * a * jnp.sin(2 * jnp.pi * x / L)

def PSE(r, rp, th):
    g, lx, ly = th[0], th[1], th[2]
    return jnp.exp(g - 0.5*((r[0]-rp[0])/jnp.exp(lx))**2
                     - 0.5*((r[1]-rp[1])/jnp.exp(ly))**2)

# ------------------------------------------------------------------- points
x_u1_b = jnp.linspace(0.0, L, 62); R_u1 = jnp.stack([x_u1_b,  width(x_u1_b)/2], -1)
x_u2_b = jnp.linspace(0.0, L, 62); R_u2 = jnp.stack([x_u2_b, -width(x_u2_b)/2], -1)
R_wall = jnp.concatenate([R_u1, R_u2], 0)
R_u_train = R_wall

y_s = jnp.linspace(-avg_width/2, avg_width/2, 15)
R_dSu1 = jnp.stack([jnp.zeros_like(y_s), y_s], -1)
R_dSu2 = R_dSu1
R_sp   = R_dSu1

N_f = N_s = 340
def points_grid(N=N_f, n_cols=20):
    n_rows = int(jnp.ceil(N / n_cols))
    x_c = (jnp.arange(n_cols) + 0.5) / n_cols * L
    e = ((jnp.arange(n_rows) + 0.5) / n_rows - 0.5) * (1 - 2*MARGIN)
    X, E = jnp.meshgrid(x_c, e, indexing="ij")
    x = jnp.asarray(X.ravel()[:N])
    return x, E.ravel()[:N] * width(x)
x_f, y_f = points_grid()
R_f1 = R_f2 = R_s = jnp.stack([x_f, y_f], -1)
R_G1 = R_G2 = R_f1

# ---------------------------------------------------------------- operators
theta = lambda t, i: jax.lax.dynamic_slice(t, (3*i,), (3,))
D    = lambda k, a: lambda r, rp, t: jax.grad(k, 0)(r, rp, t)[a]
Dp   = lambda k, a: lambda r, rp, t: jax.grad(k, 1)(r, rp, t)[a]
Lap  = lambda k:    lambda r, rp, t: jnp.trace(jax.hessian(k, 0)(r, rp, t))
Lapp = lambda k:    lambda r, rp, t: jnp.trace(jax.hessian(k, 1)(r, rp, t))
add  = lambda *ks:  lambda r, rp, t: sum(k(r, rp, t) for k in ks)
mul  = lambda c, k: lambda r, rp, t: c * k(r, rp, t)
_eL  = jnp.array([L, 0.0])
S    = lambda k: lambda r, rp, t: k(r, rp, t) - k(r + _eL, rp, t)
Sp   = lambda k: lambda r, rp, t: k(r, rp, t) - k(r, rp + _eL, t)

k_uu = [[lambda r, rp, t: PSE(r,  rp, theta(t, 0)),
         lambda r, rp, t: PSE(r,  rp, theta(t, 1))],
        [lambda r, rp, t: PSE(rp, r,  theta(t, 1)),
         lambda r, rp, t: PSE(r,  rp, theta(t, 2))]]
k_up = [lambda r, rp, t: PSE(r, rp, theta(t, 3)),
        lambda r, rp, t: PSE(r, rp, theta(t, 4))]
k_pu = k_up
k_pp = lambda r, rp, t: PSE(r, rp, theta(t, 5))

k_u_dsu = lambda a,b: Sp(k_uu[a][b])
k_u_dsp = lambda a:   Sp(k_up[a])
k_uf = lambda a,b: add(Dp(k_up[a], b), mul(-eta, Lapp(k_uu[a][b])))
k_us = lambda a:   add(*[Dp(k_uu[a][b], b) for b in range(2)])
k_pf = lambda a:   add(Dp(k_pp, a), mul(-eta, Lapp(k_pu[a])))
k_ps =             add(*[Dp(k_pu[b], b) for b in range(2)])
k_dsu_dsu = lambda a,b: Sp(S(k_uu[a][b]))
k_dsu_dsp = lambda a:   Sp(S(k_up[a]))
k_dsu_s = lambda a: S(k_us(a))
k_dsp_dsp = Sp(S(k_pp))
k_dsp_s = S(k_ps)
k_ff = lambda a,b: add(D(k_pf(b), a), mul(-eta, Lap(k_uf(a, b))))
k_fs = lambda a:   add(D(k_ps, a),    mul(-eta, Lap(k_us(a))))
k_ss =             add(*[D(k_us(a), a) for a in range(2)])

# ---- G = I + dt * F -------------------------------------------------------
k_fu = lambda a,b: add(D(k_pu[b], a), mul(-eta, Lap(k_uu[a][b])))
k_uG = lambda a,b: add(k_uu[a][b], mul(DT, k_uf(a,b)))
k_pG = lambda b:   add(k_pu[b],    mul(DT, k_pf(b)))
k_Gs = lambda a:   add(k_us(a),    mul(DT, k_fs(a)))
k_GG = lambda a,b: add(k_uu[a][b], mul(DT, k_uf(a,b)),
                       mul(DT, k_fu(a,b)), mul(DT**2, k_ff(a,b)))
k_dsu_G = lambda a,b: S(k_uG(a,b))
k_dsp_G = lambda b:   S(k_pG(b))

# -------------------------------------------------------------------- prior
def kmat(k):
    return jax.jit(jax.vmap(jax.vmap(k, (None, 0, None)), (0, None, None)))

blocks = [("u1", R_u_train), ("u2", R_u_train),
          ("Su1", R_dSu1), ("Su2", R_dSu2), ("Sp", R_sp),
          ("G1", R_G1), ("G2", R_G2), ("s", R_s)]

kfn = {
 ("u1","u1"): k_uu[0][0],      ("u1","u2"): k_uu[0][1],  ("u2","u2"): k_uu[1][1],
 ("u1","Su1"): k_u_dsu(0,0),   ("u1","Su2"): k_u_dsu(0,1),
 ("u2","Su1"): k_u_dsu(1,0),   ("u2","Su2"): k_u_dsu(1,1),
 ("u1","Sp"): k_u_dsp(0),      ("u2","Sp"): k_u_dsp(1),
 ("Su1","Su1"): k_dsu_dsu(0,0),("Su1","Su2"): k_dsu_dsu(0,1),
 ("Su2","Su1"): k_dsu_dsu(1,0),("Su2","Su2"): k_dsu_dsu(1,1),
 ("Su1","Sp"): k_dsu_dsp(0),   ("Su2","Sp"): k_dsu_dsp(1),
 ("Sp","Sp"): k_dsp_dsp,
 ("u1","G1"): k_uG(0,0),       ("u1","G2"): k_uG(0,1),
 ("u2","G1"): k_uG(1,0),       ("u2","G2"): k_uG(1,1),
 ("u1","s"): k_us(0),          ("u2","s"): k_us(1),
 ("Su1","G1"): k_dsu_G(0,0),   ("Su1","G2"): k_dsu_G(0,1),
 ("Su2","G1"): k_dsu_G(1,0),   ("Su2","G2"): k_dsu_G(1,1),
 ("Su1","s"): k_dsu_s(0),      ("Su2","s"): k_dsu_s(1),
 ("Sp","G1"): k_dsp_G(0),      ("Sp","G2"): k_dsp_G(1),
 ("Sp","s"): k_dsp_s,
 ("G1","G1"): k_GG(0,0),       ("G1","G2"): k_GG(0,1),  ("G2","G2"): k_GG(1,1),
 ("G1","s"): k_Gs(0),          ("G2","s"): k_Gs(1),
 ("s","s"): k_ss,
}

def build_K(th):
    n = len(blocks)
    rows = [[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            ni, Ri = blocks[i]; nj, Rj = blocks[j]
            rows[i][j] = kmat(kfn[(ni, nj)])(Ri, Rj, th)
            rows[j][i] = rows[i][j].T
    return jnp.block(rows)

theta_init = jnp.array([
    1.0, jnp.log(0.50), jnp.log(0.30),
    0.8, jnp.log(0.60), jnp.log(0.35),
    1.1, jnp.log(0.55), jnp.log(0.32),
    0.0, jnp.log(0.70), jnp.log(0.40),
    0.0, jnp.log(0.65), jnp.log(0.38),
    0.7, jnp.log(0.80), jnp.log(0.45),
])

def make_y(u1_prev, u2_prev):
    return jnp.concatenate([
        jnp.zeros(R_u_train.shape[0]), jnp.zeros(R_u_train.shape[0]),
        jnp.zeros(15), jnp.zeros(15), jnp.zeros(15),
        u1_prev + DT * FBODY[0],
        u2_prev + DT * FBODY[1],
        jnp.zeros(N_s),
    ])

ntrain = sum(R.shape[0] for _, R in blocks)
print(f"ntrain = {ntrain}")

@jax.jit
def neg_log_posterior(th, y):
    K  = build_K(th) + (EPS_JITTER**2) * jnp.eye(ntrain)
    Lc = jnp.linalg.cholesky(K)
    v  = solve_triangular(Lc, y, lower=True)
    return (0.5*jnp.dot(v, v) + jnp.sum(jnp.log(jnp.diag(Lc)))
            + 0.5*ntrain*jnp.log(2.0*jnp.pi)
            + 0.5*jnp.sum(th[0::3]**2)/(2.0**2))

# ---------------------------------------------------------------- posterior
row_u1 = [k_uu[0][0], k_uu[0][1], Sp(k_uu[0][0]), Sp(k_uu[0][1]), Sp(k_up[0]),
          k_uG(0,0), k_uG(0,1), k_us(0)]
row_u2 = [k_uu[1][0], k_uu[1][1], Sp(k_uu[1][0]), Sp(k_uu[1][1]), Sp(k_up[1]),
          k_uG(1,0), k_uG(1,1), k_us(1)]
row_p  = [k_pu[0], k_pu[1], Sp(k_pu[0]), Sp(k_pu[1]), Sp(k_pp),
          k_pG(0), k_pG(1), k_ps]
ROWS  = {"u1": row_u1, "u2": row_u2, "p": row_p}
KSTAR = {"u1": k_uu[0][0], "u2": k_uu[1][1], "p": k_pp}

def build_Q(th, R_star, star):
    return jnp.concatenate([kmat(k)(R_star, R, th)
                            for k, (_, R) in zip(ROWS[star], blocks)], axis=1)

@partial(jax.jit, static_argnames=("star",))
def predict(th, y, R_star, star="u1"):
    K  = build_K(th) + (EPS_JITTER**2) * jnp.eye(ntrain)
    Lc = jnp.linalg.cholesky(K)
    Q  = build_Q(th, R_star, star)
    alpha = solve_triangular(Lc.T, solve_triangular(Lc, y, lower=True), lower=False)
    mean  = Q @ alpha
    V     = solve_triangular(Lc, Q.T, lower=True)
    pv    = jax.vmap(lambda r: KSTAR[star](r, r, th))(R_star)
    std   = jnp.sqrt(jnp.clip(pv - jnp.sum(V**2, axis=0), 0.0))
    return mean, std

if __name__ == "__main__":
    y1 = make_y(jnp.zeros(R_G1.shape[0]), jnp.zeros(R_G2.shape[0]))
    print("y[G1] unique:", onp.unique(onp.asarray(y1[278:618])))
    t0 = time.time()
    v0 = neg_log_posterior(theta_init, y1)
    print(f"nlp(theta_init) = {float(v0):.6e}   [compile+eval {time.time()-t0:.1f}s]")
    t0 = time.time()
    v0 = neg_log_posterior(theta_init + 0.01, y1)
    print(f"second eval {time.time()-t0:.2f}s")