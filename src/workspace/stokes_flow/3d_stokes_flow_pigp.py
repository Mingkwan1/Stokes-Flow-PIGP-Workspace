"""
Replicating Fujita-san work on inferring  3D Stokes flow using PIGP.

There will be one option for a 3D spatial domain
another for a 2D spatial + temporal domains
"""

### Importing Libraries and Modules
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jrd

## Functions

def Product_Squared_Exponential_Kernel(r, rp, theta):
    gamma, log_lx ,log_ly, log_lz = theta[0],theta[1],theta[2],theta[3]
    return jnp.exp(gamma - 0.5*((r[0]-rp[0])/jnp.exp(log_lx))**2 - 0.5*((r[1]-rp[1])/jnp.exp(log_ly))**2 - 0.5*((r[2]-rp[2])/jnp.exp(log_lz))**2)

@jax.jit
def Cholesky(K, ϵ):
    try:
        L = jnp.linalg.cholesky(K + jnp.diag(ϵ))
        return L
    except:
        print("Unexpected Error")
        raise


### Setting of the fluid characteristics and wall geometry

def width(x):
    """Width of the pipe at position x."""
    return avg_width + 2 * a * jnp.sin (2 * jnp.pi * x / L)

η = 1.0 # Viscosity of the fluid
ρ = 1.0 # Density rho
Q = 1.0 # Flow rate
L = 2.5 # Characteristic Length
avg_width = 1 # Average width
a = 0.2

# a = 0.2 if args.geometry == "sinusoidal" else 0.0   # wall amplitude
# if args.geometry == "plates":
#     print("[config] flat parallel plates: a=0, "f"constant half-width {avg_width/2:.3f}")

ΔP = -30 # Pressure Diff
FBODY = -ΔP/L * jnp.array([1.0, 0.0]) # Body force
MARGIN = 0.999

# Δt = args.dt

# EPS_JITTER = args.eps_jitter

### Input data

## 1.) velocity at bounday u_b (62 x 3 points)

x_u_b = jnp.linspace(0.0, L, 62)
boundary_theta = jnp.array(0.0, 2*jnp.pi, 32,endpoint=False) 

x_u_b , boundary_theta = jnp.meshgrid(x_u_b, boundary_theta, indexing='ij') # make a grid of x and theta values ij to make the x come first

y_u_b = width(x_u_b) / 2 * jnp.cos(boundary_theta)
z_u_b = width(x_u_b) / 2 * jnp.sin(boundary_theta)

coor_u_b = jnp.stack([x_u_b, y_u_b, z_u_b], axis=-1)

u1_b = jnp.zeros_like(x_u_b)
u2_b = jnp.zeros_like(y_u_b)
u3_b = jnp.zeros_like(z_u_b)

u_wall = jnp.concatenate([u1_b, u2_b, u3_b], axis=0)    

# print(x_u_b.shape(), y_u_b.shape(), z_u_b.shape(), coor_wall.shape())

R0 = average_width / 2

def disk_plane(n_r=15, n_theta=16, x0=0.0):
    r = jnp.linspace(-R0, R0, n_r)
    theta = jnp.linspace(0.0, jnp.pi, n_theta, endpoint=False)   # half turn, no double cover
    r, theta = jnp.meshgrid(r, theta, indexing='ij')              # (15, 16)
    x = jnp.full_like(r, x0)
    y = r * jnp.cos(theta)
    z = r * jnp.sin(theta)
    return jnp.stack([x, y, z], axis=-1).reshape(-1, 3)           # (240, 3)

## 2.) Slip periodic velocity (15 x 16 = 240 points, 3 components = 720 targets)
coor_dSu1 = disk_plane()                                          # (240, 3)

s_u1 = jnp.zeros(coor_dSu1.shape[0])
s_u2 = jnp.zeros(coor_dSu1.shape[0])
s_u3 = jnp.zeros(coor_dSu1.shape[0])
s_u = jnp.concatenate([s_u1, s_u2, s_u3], axis=0)                 # (720,)

## 3.) Slip periodic pressure (15 x 16 = 240 points)
coor_sp = disk_plane()                                            # (240, 3)
s_p = jnp.zeros(coor_sp.shape[0])                                 # (240,)

## 4.) Governing equations f,g (340 x 3 points)

N_f = N_s = 340

def points_random(N=N_f, key=jax.random.PRNGKey(0)):
    kx, kr, kt = jax.random.split(key, 3)
    x = jax.random.uniform(kx, (N,), minval=0.0, maxval=L)
    u = jax.random.uniform(kr, (N,))
    theta = jax.random.uniform(kt, (N,), minval=0.0, maxval=2*jnp.pi)
    R_max = width(x) / 2 * (1 - MARGIN)
    r = R_max * jnp.sqrt(u)
    return x, r * jnp.cos(theta), r * jnp.sin(theta)

x_f, y_f, z_f = points_random(N=N_f)
coor_f = jnp.stack([x_f, y_f, z_f], axis=-1)          # (340, 3), same layout as coor_u_b

# momentum residual targets, one per velocity component
f_1 = jnp.full(N_f, -ΔP / L)                          # x-momentum  (340,)
f_2 = jnp.zeros(N_f)                                  # y-momentum  (340,)
f_3 = jnp.zeros(N_f)                                  # z-momentum  (340,)

f = jnp.concatenate([f_1, f_2, f_3], axis=0)          # (1020,)  block ordering [f_1 | f_2 | f_3]

## 5.) Governing equations s (340 points)

coor_s = coor_f                                       # same collocation points, (340, 3)
s = jnp.zeros(N_s)                                    # continuity (div u = 0), one value per point (340,)


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

_eL  = jnp.array([L, 0.0])
S    = lambda k: None if k is None else (lambda r, rp, t: k(r, rp, t) - k(r + _eL, rp, t))
Sp   = lambda k: None if k is None else (lambda r, rp, t: k(r, rp, t) - k(r, rp + _eL, t))

### Base kernel (Total of 6 kernels and 18 parameters)

_k = lambda i: (lambda r, rp, t: Product_Squared_Exponential_Kernel(r, rp, theta(t, i)))

k_uu = [[_k(0), _k(1), _k(2)],
        [_k(1), _k(3), _k(4)],
        [_k(2), _k(4), _k(5)]]

k_pp = _k(6)

k_up = [None, None]
k_pu = [None, None]
    
# k_pp = lambda r, rp, t: Product_Squared_Exponential_Kernel(r, rp, theta(t, 3))

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

k_fu = lambda a, b: add(D(k_pu[b], a), mul(-η, Lap(k_uu[a][b])))

### Kernel Matrix

## Buiding K

BLOCK_NAMES = ["u1", "u2","u3", "Su1", "Su2","Su3", "Sp", "f1", "f2","f3", "s"]


theta_init = jnp.array([
    1.2, jnp.log(0.30), jnp.log(0.30),   # u1-u1
   1.2, jnp.log(0.30), jnp.log(0.30),   # u1-u2  
   1.2, jnp.log(0.30), jnp.log(0.30),   # u1-u3  
    1.2, jnp.log(0.30), jnp.log(0.30),   # u2-u2
    1.2, jnp.log(0.30), jnp.log(0.30),   # u2-u3
    1.2, jnp.log(0.30), jnp.log(0.30),   # u3-u3
   0.0, jnp.log(0.30), jnp.log(0.30),   # u1-p
   0.0, jnp.log(0.30), jnp.log(0.30),   # u2-p
   0.0, jnp.log(0.30), jnp.log(0.30),   # u3-p
    1.2, jnp.log(0.30), jnp.log(0.30),   # p-p
])

def kmat(k):
    if k is None:
        return lambda R1, R2, t: jnp.zeros((R1.shape[0], R2.shape[0]))
    return jax.jit(jax.vmap(jax.vmap(k, (None, 0, None)), (0, None, None)))

def make_blocks(R_f, R_O):
    """Block layout. Only the G blocks move between time steps."""
    b = []
    if USE_WALL: b += [("u1w", "u1", R_wall), # no-slip, u_x
                        ("u2w", "u2", R_wall),# no-slip, u_y
                        ("u3w", "u3", R_wall),# no-slip, u_z
                        ] 
    if USE_OBS:  b += [("u1o", "u1", R_O),     # no-slip, u_x
                        ("u2o", "u2", R_O),
                        ("u3o", "u3", R_O)] # no-slip, u_y
    return b + [("Su1", "Su1", R_dSu1), # periodicity, u_x
                ("Su2", "Su2", R_dSu2), # periodicity, u_y
                ("Su3", "Su3", R_dSu3), # periodicity, u_z
                ("Sp", "Sp", R_sp),# periodicity, p
                ("G1", "G1", R_f), # backward-Euler momentum, x
                ("G2", "G2", R_f), # backward-Euler momentum, y
                ("G3", "G3", R_f), # backward-Euler momentum, z
                ("s", "s", R_s)] # continuity

def block_slices(R_f, R_O):
    """Start/end offsets of each named block in the K / y ordering."""
    offsets = {}
    start = 0
    for name, _, R in make_blocks(R_f, R_O):
        offsets[name] = (start, start + R.shape[0])
        start += R.shape[0]
    return offsets

kfn = {
    ("u1","u1"): k_uu[0][0],           ("u1","u2"): k_uu[0][1],             ("u1","u3"): k_uu[0][2],
    ("u2","u2"): k_uu[1][1],           ("u2","u3"): k_uu[1][2],
    ("u3","u3"): k_uu[2][2],

    ("u1","Su1"): k_u_dsu(0,0),        ("u1","Su2"): k_u_dsu(0,1),         ("u1","Su3"): k_u_dsu(0,2),
    ("u2","Su1"): k_u_dsu(1,0),        ("u2","Su2"): k_u_dsu(1,1),         ("u2","Su3"): k_u_dsu(1,2),
    ("u3","Su1"): k_u_dsu(2,0),        ("u3","Su2"): k_u_dsu(2,1),         ("u3","Su3"): k_u_dsu(2,2),

    ("u1","Sp"):  k_u_dsp(0),          ("u2","Sp"):  k_u_dsp(1),           ("u3","Sp"):  k_u_dsp(2),

    ("Su1","Su1"): k_dsu_dsu(0,0),     ("Su1","Su2"): k_dsu_dsu(0,1),     ("Su1","Su3"): k_dsu_dsu(0,2), 
    ("Su2","Su1"): k_dsu_dsu(1,0),     ("Su2","Su2"): k_dsu_dsu(1,1),     ("Su2","Su3"): k_dsu_dsu(1,2),
    ("Su3","Su1"): k_dsu_dsu(2,0),     ("Su3","Su2"): k_dsu_dsu(2,1),     ("Su3","Su3"): k_dsu_dsu(2,2),

    ("Su1","Sp"):  k_dsu_dsp(0),       ("Su2","Sp"):  k_dsu_dsp(1),       ("Su3","Sp"):  k_dsu_dsp(2),

    ("Sp","Sp"):   k_dsp_dsp,

    ("u1","f1"): k_uf(0,0),            ("u1","f2"): k_uf(0,1),          ("u1","f3"): k_uf(0,2),
    ("u2","f1"): k_uf(1,0),            ("u2","f2"): k_uf(1,1),          ("u2","f3"): k_uf(1,2),
    ("u3","f1"): k_uf(2,0),            ("u3","f2"): k_uf(2,1),          ("u3","f3"): k_uf(2,2),

    ("u1","s"):  k_us(0),              ("u2","s"):  k_us(1),            ("u3","s"):  k_us(2),

    ("Su1","f1"): k_dsu_f(0,0),        ("Su1","f2"): k_dsu_f(0,1),        ("Su1","f3"): k_dsu_f(0,2),
    ("Su2","f1"): k_dsu_f(1,0),        ("Su2","f2"): k_dsu_f(1,1),        ("Su2","f3"): k_dsu_f(1,2),
    ("Su3","f1"): k_dsu_f(2,0),        ("Su3","f2"): k_dsu_f(2,1),        ("Su3","f3"): k_dsu_f(2,2),

    ("Su1","s"):  k_dsu_s(0),          ("Su2","s"):  k_dsu_s(1),          ("Su3","s"):  k_dsu_s(2),

    ("f1","f1"): k_f(0,0),            ("f1","f2"): k_f(0,1),            ("f1","f3"): k_f(0,2),   
    ("f2","f1"): k_f(1,0),            ("f2","f2"): k_f(1,1),            ("f2","f3"): k_f(1,2),   
    ("f3","f1"): k_f(2,0),            ("f3","f2"): k_f(2,1),            ("f3","f3"): k_f(2,2),   

    ("f1","s"):  k_fs(0),              ("f2","s"):  k_fs(1),            ("f3","s"):  k_fs(2),

    ("s","s"):   k_ss,
}

def kblock(ki, Ri, kj, Rj, theta):
    if (ki, kj) in kfn:
        return kmat(kfn[(ki, kj)])(Ri, Rj, theta)
    return kmat(kfn[(kj, ki)])(Rj, Ri, theta).T

def build_K(theta,R_f, R_O):
    blocks = make_blocks(R_f, R_O)
    n = len(blocks)
    rows = [[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            _, ki, Ri = blocks[i]
            _, kj, Rj = blocks[j]
            rows[i][j] = kblock(ki, Ri, kj, Rj, theta)
            rows[j][i] = rows[i][j].T
    return jnp.block(rows)

#Make K

y = jnp.concatenate([
    u_wall,                      # 
    s_u,                     # 
    s_p,                       # 
    f,                       # 
    s,                         #
])

### Posterior Calculation

@jax.jit
def neg_log_posterior(theta):
    K = build_K(theta) + (EPS_JITTER**2) * jnp.eye(ntrain)
    Lc = jnp.linalg.cholesky(K)
    v  = solve_triangular(Lc, y, lower=True)
    nlp = (0.5*jnp.dot(v, v) + jnp.sum(jnp.log(jnp.diag(Lc))) + 0.5*ntrain*jnp.log(2.0*jnp.pi)) + 0.5*jnp.sum(theta[0::3]**2)/(2.0**2)
    return nlp

fun = neg_log_posterior

### Optimization


NM_SOLVER = jaxopt.ScipyMinimize(
    fun=_nlp_value_zero_grad, value_and_grad=True,
    method="Nelder-Mead", tol=args.tol, maxiter=args.nm_iter,
    callback=_nm_callback, options={"disp": True, "adaptive": True})

LBFGS_SOLVER = jaxopt.ScipyMinimize(
    fun=nlp_active, method="L-BFGS-B",
    tol=args.lbfgs_tol, maxiter=args.lbfgs_iter)

def run_nelder_mead(x0, y, R_G, R_O, N_G):
    _NM_STATE["it"], _NM_STATE["data"] = 0, (y, R_G, R_O, N_G)
    res = NM_SOLVER.run(x0, y, R_G, R_O, N_G)
    return res.params, int(res.state.iter_num)

def run_nm_then_lbfgs(x0, y, R_G, R_O, N_G):
    ta, nit = run_nelder_mead(x0, y, R_G, R_O, N_G)
    if args.lbfgs_iter <= 0:
        return ta, nit
    f_nm = float(nlp_active(ta, y, R_G, R_O, N_G))
    res = LBFGS_SOLVER.run(ta, y, R_G, R_O, N_G)         
    tb = res.params
    f_lb = float(nlp_active(tb, y, R_G, R_O, N_G))
    ok = bool(jnp.all(jnp.isfinite(tb))) and onp.isfinite(f_lb) and f_lb < f_nm
    print(f"[lbfgs] NM {f_nm:.6e} -> LBFGS {f_lb:.6e}  "
          f"({int(res.state.iter_num)} iters)  {'accepted' if ok else 'REJECTED, keeping NM'}")
    return (tb if ok else ta), nit + int(res.state.iter_num)

### Prediction

# ---- test points: a grid inside the channel -------------------------------
NX, NY, NZ= 41, 41, 41
x_line = jnp.linspace(0.0, L, NX)
eta_line = jnp.linspace(-0.999, 0.999, NY) * 0.5      
XX, EE = jnp.meshgrid(x_line, eta_line, indexing="ij")
YY = EE * width(XX)
R_test = jnp.stack([XX.ravel(), YY.ravel()], axis=-1) 

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

u1_mean, u1_std = predict(final_params, R_test, "u1")
u2_mean, u2_std = predict(final_params, R_test, "u2")
u3_mean, u3_std = predict(final_params, R_test, "u3")

### PLotting

plotting.plot_all(ctx, {
    "u_x": (u1_mean, u1_std),
    "u_y": (u2_mean, u2_std),
    "u_z": (u3_mean, u3_std),
}, fem=fem_data)
plt.show()

