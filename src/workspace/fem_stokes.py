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