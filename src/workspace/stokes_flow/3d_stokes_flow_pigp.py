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


### Kernel Matrix

### Posterior Calculation

### Optimization

### PLotting


### Main Function + Arguments

## Parse arguments