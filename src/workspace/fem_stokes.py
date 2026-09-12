"""Reference Stokes solution in a sinusoidal channel, via scikit-fem.

Self-contained: imports nothing from the PIGP script. Pure Python (no MPI,
no PETSc), so it installs and runs natively on Windows with `uv add scikit-fem`.

Taylor-Hood P2/P1, no-slip on the sinusoidal walls, exactly x-periodic via a
DOF-identification matrix, driven by a uniform body force. This matches the
PIGP setup: same geometry, same BCs, same forcing, and pressure determined
only up to a constant.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------
def _arr_hash(*arrays):
    h = hashlib.sha256()
    for A in arrays:
        h.update(np.ascontiguousarray(np.asarray(A), dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _fingerprint(cfg, sample_points):
    d = dict(cfg)
    d["points_hash"] = _arr_hash(sample_points)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# periodic constraint
# --------------------------------------------------------------------------
def _periodic_prolongation(basis, L, tol=1e-9):
    """Build P (n_full x n_reduced) identifying dofs at x=L with those at x=0.

    scikit-fem 12 has no built-in periodic basis, so we reduce the system
    ourselves:  K_red = P.T @ K @ P,  F_red = P.T @ F.
    """
    from scipy.sparse import coo_matrix

    dl = basis.doflocs
    n = dl.shape[1]

    left = np.flatnonzero(np.abs(dl[0] - 0.0) < tol)
    right = np.flatnonzero(np.abs(dl[0] - L) < tol)
    if left.size != right.size:
        raise RuntimeError(
            f"periodic pairing failed: {left.size} dofs at x=0 but "
            f"{right.size} at x={L}. The mesh must be a tensor grid so the "
            f"two boundary columns line up.")

    # pair by y; for vector elements two dofs share a location, so break the
    # tie with the dof's index within its location group
    def key(idx):
        y = dl[1, idx]
        order = np.lexsort((idx, np.round(y, 12)))
        return idx[order]

    left, right = key(left), key(right)
    if not np.allclose(dl[1, left], dl[1, right], atol=1e-9):
        raise RuntimeError("periodic pairing failed: y-coordinates disagree.")

    # reduced numbering: drop every right dof, redirect it to its left partner
    keep = np.ones(n, dtype=bool)
    keep[right] = False
    new = np.full(n, -1, dtype=np.int64)
    new[keep] = np.arange(keep.sum())
    new[right] = new[left]

    P = coo_matrix((np.ones(n), (np.arange(n), new)),
                   shape=(n, int(keep.sum()))).tocsr()
    return P, keep

def _build_mesh_and_bases(L, a, avg_width, nx, ny):
    from skfem import MeshTri, Basis, ElementVector, ElementTriP2, ElementTriP1

    def wid(x):
        return avg_width + 2.0*a*np.sin(2.0*np.pi*x/L)

    m0 = MeshTri.init_tensor(np.linspace(0.0, L, nx + 1),
                             np.linspace(-0.5, 0.5, ny + 1))
    p = m0.p.copy()
    p[1] = p[1] * wid(p[0]) / avg_width
    m = MeshTri(p, m0.t)

    uelem = ElementVector(ElementTriP2())
    ub = Basis(m, uelem, intorder=4)
    pb = ub.with_element(ElementTriP1())
    return m, ub, pb, wid


def _wall_dofs(ub, wid, L, nx, verbose):
    wall_tol = 1e-3
    wall = ub.get_dofs(
        lambda x: np.abs(np.abs(x[1]) - wid(x[0])/2.0) < wall_tol).all()
    if verbose:
        expect = 2*(2*(2*nx))
        print(f"[fem] {wall.size} wall dofs (expect ~{expect})")
    if wall.size < 0.5*2*(2*(2*nx)):
        raise RuntimeError(
            f"only {wall.size} wall dofs found - the no-slip condition would "
            f"be under-applied. Increase wall_tol.")
    return wall


def _sample(basis, vals, pts, chunk=250):
    itp = basis.interpolator(vals)
    return np.concatenate(
        [itp(pts[:, k:k+chunk]) for k in range(0, pts.shape[1], chunk)],
        axis=-1)
# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
def solve(sample_points, *, L, a, avg_width, eta, body_force_x,
          nx=240, ny=64, verbose=True):
    """Returns (u1, u2, p) sampled at `sample_points`, shape (N, 2).

    Pressure is mean-subtracted: the periodic system fixes it only up to a
    constant, exactly as the PIGP model does.
    """
    try:
        from skfem import (MeshTri, Basis, ElementVector, ElementTriP2,
                           ElementTriP1, BilinearForm, LinearForm, condense,
                           solve as skfem_solve)
        from skfem.helpers import grad, div
    except ImportError as e:
        raise ImportError(
            "scikit-fem is required for the FEM reference solve. "
            "Install it with:  uv add scikit-fem") from e

    import scipy.sparse as sp

    def wid(x):
        return avg_width + 2.0*a*np.sin(2.0*np.pi*x/L)

    # ---- mesh: tensor grid on [0,L]x[-1/2,1/2], y scaled by local width ---
    # Tensor structure matters: it guarantees the x=0 and x=L dof columns
    # line up, which the periodic pairing relies on.
    m0 = MeshTri.init_tensor(np.linspace(0.0, L, nx + 1),
                             np.linspace(-0.5, 0.5, ny + 1))
    p = m0.p.copy()
    p[1] = p[1] * wid(p[0]) / avg_width
    m = MeshTri(p, m0.t)

    # ---- Taylor-Hood P2/P1 (shared quadrature) ---------------------------
    uelem = ElementVector(ElementTriP2())
    ub = Basis(m, uelem, intorder=4)
    pb = ub.with_element(ElementTriP1())      # must share ub's quadrature

    @BilinearForm
    def vector_lap(u, v, w):
        return eta*np.einsum('ij...,ij...', grad(u), grad(v))

    @BilinearForm
    def divergence(u, q, w):
        return -q*div(u)

    @LinearForm
    def body(v, w):
        return body_force_x*v[0]

    A = vector_lap.assemble(ub)
    B = divergence.assemble(ub, pb)
    f = body.assemble(ub)

    nu, npr = ub.N, pb.N
    K = sp.bmat([[A, B.T], [B, None]], format='csr')
    F = np.concatenate([f, np.zeros(npr)])

    # ---- exact x-periodicity ---------------------------------------------
    Pu, _ = _periodic_prolongation(ub, L)
    Pp, _ = _periodic_prolongation(pb, L)
    P = sp.block_diag([Pu, Pp], format='csr')
    Kr = (P.T @ K @ P).tocsr()
    Fr = P.T @ F

    # ---- no-slip walls, in reduced numbering ------------------------------
    # P2 edge-midpoint dofs sit on the chord, not the sine curve, so the
    # predicate needs a tolerance larger than the sagitta (~4e-5 here) but
    # smaller than the first interior row (~1.2e-2).
    wall_tol = 1e-3
    wall = ub.get_dofs(
        lambda x: np.abs(np.abs(x[1]) - wid(x[0])/2.0) < wall_tol).all()
    if verbose:
        expect = 2*(2*(2*nx))          # 2 components x 2 walls x (nodes+midpoints)
        print(f"[fem] mesh {m.p.shape[1]} vertices, {m.t.shape[1]} triangles; "
              f"{wall.size} wall dofs (expect ~{expect})")
    if wall.size < 0.5*2*(2*(2*nx)):
        raise RuntimeError(
            f"only {wall.size} wall dofs found - the no-slip condition would "
            f"be under-applied. Increase wall_tol.")

    # reduced index of a full dof i is the column of P in row i
    full_to_red = np.asarray(P.argmax(axis=1)).ravel()
    D = np.unique(full_to_red[wall])
    D = np.concatenate([D, [full_to_red[nu]]])   # pin one pressure dof

    xr = skfem_solve(*condense(Kr, Fr, D=D))
    x = P @ xr
    u, pr = x[:nu], x[nu:]

    # ---- sample -----------------------------------------------------------
    pts = np.asarray(sample_points).T          # (2, N)

    def _sample(basis, vals, chunk=250):
        itp = basis.interpolator(vals)         # brute-force search: chunk it
        return np.concatenate(
            [itp(pts[:, k:k+chunk]) for k in range(0, pts.shape[1], chunk)],
            axis=-1)

    uv = _sample(ub, u)
    pv = _sample(pb, pr)
    pv = pv - np.nanmean(pv)

    if verbose:
        print(f"[fem] peak u_x = {np.nanmax(uv[0]):.4f}   "
              f"min u_x = {np.nanmin(uv[0]):.4f}   "
              f"max|u_y| = {np.nanmax(np.abs(uv[1])):.4f}")
    return uv[0], uv[1], pv


# --------------------------------------------------------------------------
# unsteady solver -- backward Euler, SAME scheme as the PIGP march
# --------------------------------------------------------------------------
def solve_unsteady(sample_points, snap_steps, *, L, a, avg_width, eta,
                   rho, body_force_x, dt, nx=240, ny=64, verbose=True):
    """March backward-Euler Stokes in FEM space, from u^0 = 0 at rest.
 
        rho*(u^n - u^{n-1})/dt = -grad p^n + eta*lap u^n + F
        div u^n = 0
 
    which rearranges to the same saddle-point system solved at every step:
 
        (rho/dt * M + A) u^n + B^T p^n = rho/dt * M u^{n-1} + f
        B u^n = 0
 
    The LHS is IDENTICAL every step (same dt, same mesh), so it is
    factorized ONCE and every step is a single sparse solve -- the FEM
    analogue of the frozen-Cholesky trick used in the PIGP march.
 
    Parameters
    ----------
    snap_steps : sorted iterable of step indices (1-based) to return, e.g.
                 the PIGP script's SNAP_STEPS. Must not exceed max(snap_steps).
 
    Returns
    -------
    dict: {step: (u1, u2, p)}, each sampled at `sample_points`, one entry
    per requested step. u^0 = 0 is not included (it is not the solution of
    a linear solve here, it is the imposed initial condition).
    """
    try:
        from skfem import BilinearForm, LinearForm, condense
        from skfem.helpers import grad, div, dot
    except ImportError as e:
        raise ImportError(
            "scikit-fem is required for the FEM reference solve. "
            "Install it with:  uv add scikit-fem") from e
 
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
 
    snap_steps = sorted(set(int(s) for s in snap_steps))
    n_steps = max(snap_steps)
 
    m, ub, pb, wid = _build_mesh_and_bases(L, a, avg_width, nx, ny)
    if verbose:
        print(f"[fem-unsteady] mesh {m.p.shape[1]} vertices, "
              f"{m.t.shape[1]} triangles, dt={dt}, {n_steps} steps, "
              f"snapshots at {snap_steps}")
 
    @BilinearForm
    def vector_lap(u, v, w):
        return eta*np.einsum('ij...,ij...', grad(u), grad(v))
 
    @BilinearForm
    def divergence(u, q, w):
        return -q*div(u)
 
    @BilinearForm
    def mass(u, v, w):
        return dot(u, v)
 
    @LinearForm
    def body(v, w):
        return body_force_x*v[0]
 
    A = vector_lap.assemble(ub)
    B = divergence.assemble(ub, pb)
    M = mass.assemble(ub)
    f = body.assemble(ub)
 
    nu, npr = ub.N, pb.N
    c = rho/dt
 
    # LHS: (c*M + A)  B^T ; B  0.  Same every step.
    Klhs = sp.bmat([[c*M + A, B.T], [B, None]], format='csr')
 
    Pu, _ = _periodic_prolongation(ub, L)
    Pp, _ = _periodic_prolongation(pb, L)
    P = sp.block_diag([Pu, Pp], format='csr')
    Kr = (P.T @ Klhs @ P).tocsr()
 
    wall = _wall_dofs(ub, wid, L, nx, verbose)
    full_to_red = np.asarray(P.argmax(axis=1)).ravel()
    D = np.unique(full_to_red[wall])
    D = np.concatenate([D, [full_to_red[nu]]])   # pin one pressure dof
    keep = np.ones(Kr.shape[0], dtype=bool)
    keep[D] = False
    idx_free = np.flatnonzero(keep)
 
    # factorize the constrained LHS ONCE -- reused at every time step
    K_free = Kr[idx_free][:, idx_free].tocsc()
    lu = splu(K_free)
    if verbose:
        print(f"[fem-unsteady] factorized {K_free.shape[0]}x{K_free.shape[0]} "
              f"system once, reused for all {n_steps} steps")
 
    pts = np.asarray(sample_points).T
    u_prev_full = np.zeros(nu)     # u^0 = 0, fluid at rest -- matches PIGP IC
 
    results = {}
    for n in range(1, n_steps + 1):
        Frhs = np.concatenate([c*(M @ u_prev_full) + f, np.zeros(npr)])
        Frhs_r = P.T @ Frhs
 
        xr_free = lu.solve(Frhs_r[idx_free])
        xr = np.zeros(Kr.shape[0])
        xr[idx_free] = xr_free
        # xr[D] left at 0: homogeneous no-slip + pinned pressure dof
 
        x = P @ xr
        u_full, pr = x[:nu], x[nu:]
        u_prev_full = u_full
 
        if n in snap_steps:
            uv = _sample(ub, u_full, pts)
            pv = _sample(pb, pr, pts)
            pv = pv - np.nanmean(pv)
            results[n] = (uv[0], uv[1], pv)
            if verbose:
                print(f"[fem-unsteady] step {n:>4}  t={n*dt:.4f}  "
                      f"u_x max={np.nanmax(uv[0]):.4f}")
 
    return results

def get(sample_points, cache_path, *, refit=False, **kwargs):
    """Cached wrapper around solve(). kwargs pass straight through."""
    cache_path = Path(cache_path)
    fp = _fingerprint(kwargs, sample_points)

    if not refit and cache_path.exists():
        d = np.load(cache_path, allow_pickle=False)
        if str(d["fingerprint"]) == fp:
            print(f"[fem] loaded cached solution from {cache_path}")
            return d["u1"], d["u2"], d["p"]
        print("[fem] fingerprint mismatch -> re-solving")

    print("[fem] solving Stokes with scikit-fem ...")
    u1, u2, p = solve(sample_points, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, u1=u1, u2=u2, p=p, fingerprint=np.asarray(fp))
    print(f"[fem] wrote {cache_path}")
    return u1, u2, p

def get_unsteady(sample_points, snap_steps, cache_path, *, refit=False, **kwargs):
    """Cached wrapper around solve_unsteady()."""
    cache_path = Path(cache_path)
    cfg = dict(kwargs)
    cfg["snap_steps"] = sorted(set(int(s) for s in snap_steps))
    fp = _fingerprint(cfg, sample_points)

    if not refit and cache_path.exists():
        d = np.load(cache_path, allow_pickle=False)
        if str(d["fingerprint"]) == fp:
            print(f"[fem-unsteady] loaded cached march from {cache_path}")
            steps = d["steps"]
            return {int(s): (d[f"u1_{s}"], d[f"u2_{s}"], d[f"p_{s}"])
                    for s in steps}
        print("[fem-unsteady] fingerprint mismatch -> re-solving")

    print("[fem-unsteady] marching backward-Euler Stokes with scikit-fem ...")
    results = solve_unsteady(sample_points, snap_steps, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fingerprint": np.asarray(fp),
              "steps": np.asarray(sorted(results.keys()))}
    for s, (u1, u2, p) in results.items():
        payload[f"u1_{s}"] = u1
        payload[f"u2_{s}"] = u2
        payload[f"p_{s}"] = p
    np.savez(cache_path, **payload)
    print(f"[fem-unsteady] wrote {cache_path}")
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FEM reference Stokes solve")
    ap.add_argument("--out", default="outputs_2D/fem_reference.npz")
    ap.add_argument("--nx", type=int, default=240)
    ap.add_argument("--ny", type=int, default=64)
    args = ap.parse_args()

    # must match the PIGP test grid exactly
    L_, a_, w_, NX_, NY_ = 2.5, 0.2, 1.0, 120, 41
    xl = np.linspace(0.0, L_, NX_)
    el = np.linspace(-0.98, 0.98, NY_)*0.5
    XX, EE = np.meshgrid(xl, el, indexing="ij")
    YY = EE*(w_ + 2*a_*np.sin(2*np.pi*XX/L_))
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)

    get(pts, args.out, refit=True, L=L_, a=a_, avg_width=w_,
        eta=1.0, body_force_x=30.0/2.5, nx=args.nx, ny=args.ny)