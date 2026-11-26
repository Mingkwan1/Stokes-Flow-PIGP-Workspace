"""Reference steady Stokes solution in a 3D x-periodic pipe, via scikit-fem.

Same problem the PIGP infers (3d_stokes_pigp.py):
    grad p - eta lap u = F,   div u = 0
    u = 0 on the wall r = R(x),  u and p periodic between x = 0 and x = L,
    R(x) = width(x) / 2,  width(x) = avg_width + 2 a sin(2 pi x / L).

Method
    mesh      triangulated disk (MeshTri.init_circle) extruded along x into
              prisms, each split into 3 tetrahedra; then y, z scaled by R(x)/R0.
              Both end faces are identical, so periodic dofs pair exactly.
    elements  Taylor-Hood P2 (velocity) / P1 (pressure) on tetrahedra.
    periodic  dofs at x = L identified with their partners at x = 0
              (prolongation matrix P, system reduced to P^T K P).
    no-slip   all dofs on the wall facets (found topologically, not by distance).
    pressure  fixed at one dof (periodic BCs leave a constant free), then the
              sampled pressure is shifted to zero mean.
    solver    MINRES with a block-diagonal preconditioner
              (algebraic multigrid on the velocity block, lumped pressure mass).

Run directly to check the solver on the straight pipe against Poiseuille:
    python fem_stokes_3d.py
"""
import hashlib
import json
import time
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
# mesh
# --------------------------------------------------------------------------
def build_mesh(*, L, a, avg_width, disk_refs, nx):
    """Tetrahedral mesh of the periodic pipe.

    disk_refs : refinements of the cross-section disk (3 -> 32 wall points per
                ring, 256 triangles; 4 -> 64 wall points, 1024 triangles)
    nx        : number of slices along x
    """
    from skfem import MeshTet, MeshTri

    disk = MeshTri.init_circle(disk_refs)           # unit disk
    p2, t2 = disk.p, disk.t
    n2 = p2.shape[1]

    xs = np.linspace(0.0, L, nx + 1)
    # nodes: slice k holds the disk nodes at x = xs[k]
    P = np.empty((3, n2 * (nx + 1)))
    for k, x in enumerate(xs):
        P[0, k*n2:(k+1)*n2] = x
        P[1, k*n2:(k+1)*n2] = p2[0]
        P[2, k*n2:(k+1)*n2] = p2[1]

    # prism (a, b, c) -> (a', b', c') split into 3 tets. With a < b < c by
    # global index, every shared quad face gets the same diagonal (smaller
    # bottom vertex to larger top vertex), so the mesh is conforming.
    tri = np.sort(t2, axis=0)
    tets = []
    for k in range(nx):
        a0, b0, c0 = tri + k*n2
        a1, b1, c1 = tri + (k + 1)*n2
        tets += [np.vstack([a0, b0, c0, c1]),
                 np.vstack([a0, b0, b1, c1]),
                 np.vstack([a0, a1, b1, c1])]
    T = np.hstack(tets)

    # disk of radius 1 -> radius R(x)
    R = (avg_width + 2.0*a*np.sin(2.0*np.pi*P[0]/L)) / 2.0
    P[1] *= R
    P[2] *= R
    return MeshTet(P, T)


# --------------------------------------------------------------------------
# periodic constraint
# --------------------------------------------------------------------------
def _periodic_prolongation(basis, comp, L, tol=1e-9):
    """P (n_full x n_reduced) identifying dofs at x = L with those at x = 0.

    comp : component index of every dof (0 for a scalar basis), so that the
           u_x dof at a point pairs with the u_x dof of its partner.
    """
    from scipy.sparse import coo_matrix

    dl = basis.doflocs
    n = dl.shape[1]
    left = np.flatnonzero(np.abs(dl[0]) < tol)
    right = np.flatnonzero(np.abs(dl[0] - L) < tol)
    if left.size != right.size:
        raise RuntimeError(f"periodic pairing failed: {left.size} dofs at x=0, "
                           f"{right.size} at x=L")

    def order(idx):
        return idx[np.lexsort((comp[idx], np.round(dl[2, idx], 10), np.round(dl[1, idx], 10)))]

    left, right = order(left), order(right)
    if not (np.allclose(dl[1:, left], dl[1:, right], atol=1e-9)
            and np.array_equal(comp[left], comp[right])):
        raise RuntimeError("periodic pairing failed: end faces do not match")

    keep = np.ones(n, dtype=bool)
    keep[right] = False
    new = np.full(n, -1, dtype=np.int64)
    new[keep] = np.arange(keep.sum())
    new[right] = new[left]
    return coo_matrix((np.ones(n), (np.arange(n), new)),
                      shape=(n, int(keep.sum()))).tocsr()


def _vector_components(ub):
    """Component (0, 1, 2) of every dof of a vector P2 basis."""
    comp = np.zeros(ub.N, dtype=np.int64)
    for i in range(3):
        comp[ub.nodal_dofs[i]] = i
        comp[ub.edge_dofs[i]] = i
    return comp


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
def solve(sample_points, *, L, a, avg_width, eta, body_force_x,
          disk_refs=3, nx=32, tol=1e-10, maxiter=5000, verbose=True):
    """Returns (u1, u2, u3, p) sampled at `sample_points`, shape (N, 3).

    Points the mesh does not contain come back as NaN.
    """
    try:
        from skfem import (Basis, BilinearForm, LinearForm, ElementVector,
                           ElementTetP1, ElementTetP2, condense)
        from skfem.helpers import ddot, div, grad
    except ImportError as e:
        raise ImportError("scikit-fem is required for the FEM reference solve. "
                          "Install it with:  uv add scikit-fem pyamg") from e
    import scipy.sparse as sp
    from scipy.sparse.linalg import LinearOperator, minres
    import pyamg

    t0 = time.time()
    m = build_mesh(L=L, a=a, avg_width=avg_width, disk_refs=disk_refs, nx=nx)
    ub = Basis(m, ElementVector(ElementTetP2()), intorder=3)
    pb = ub.with_element(ElementTetP1())

    @BilinearForm
    def visc(u, v, w):
        return eta * ddot(grad(u), grad(v))

    @BilinearForm
    def divergence(u, q, w):
        return -q * div(u)

    @BilinearForm
    def pmass(p, q, w):
        return p * q

    @LinearForm
    def body(v, w):
        return body_force_x * v[0]

    A = visc.assemble(ub)
    B = divergence.assemble(ub, pb)
    Mp = pmass.assemble(pb)
    f = body.assemble(ub)
    nu, npr = ub.N, pb.N

    # ---- periodic reduction ------------------------------------------------
    Pu = _periodic_prolongation(ub, _vector_components(ub), L)
    Pp = _periodic_prolongation(pb, np.zeros(npr, dtype=np.int64), L)
    Ar, Br, Mpr = (Pu.T @ A @ Pu).tocsr(), (Pp.T @ B @ Pu).tocsr(), (Pp.T @ Mp @ Pp).tocsr()
    fr = Pu.T @ f
    nur, npr_r = Ar.shape[0], Mpr.shape[0]

    # ---- no-slip: dofs on wall facets (boundary facets not on x = 0 or x = L)
    bf = m.boundary_facets()
    xm = m.p[0, m.facets[:, bf]].mean(axis=0)
    wall_facets = bf[(xm > 1e-9) & (xm < L - 1e-9)]
    wall = ub.get_dofs(facets=wall_facets).all()
    u_to_red = np.asarray(Pu.argmax(axis=1)).ravel()
    Du = np.unique(u_to_red[wall])
    p_pin = np.asarray(Pp.argmax(axis=1)).ravel()[0]

    free_u = np.setdiff1d(np.arange(nur), Du)
    free_p = np.setdiff1d(np.arange(npr_r), [p_pin])
    Aff = Ar[free_u][:, free_u].tocsr()
    Bff = Br[free_p][:, free_u].tocsr()
    rhs = np.concatenate([fr[free_u], np.zeros(free_p.size)])
    K = sp.bmat([[Aff, Bff.T], [Bff, None]], format="csr")
    if verbose:
        print(f"[fem3d] mesh {m.p.shape[1]} vertices, {m.t.shape[1]} tets; "
              f"unknowns {K.shape[0]} (velocity {free_u.size}, pressure {free_p.size}); "
              f"{Du.size} wall dofs")

    # ---- MINRES, block-diagonal preconditioner -----------------------------
    # one AMG V-cycle is a fixed symmetric operator, as MINRES requires
    amg = pyamg.smoothed_aggregation_solver(Aff, symmetry="symmetric",
                                            max_coarse=500).aspreconditioner(cycle="V")
    Mp_diag = np.asarray(Mpr[free_p][:, free_p].sum(axis=1)).ravel() / eta
    nf = free_u.size

    def prec(x):
        y = np.empty_like(x)
        y[:nf] = amg @ x[:nf]
        y[nf:] = x[nf:] / Mp_diag
        return y

    M = LinearOperator(K.shape, matvec=prec)
    its = [0]
    sol, info = minres(K, rhs, M=M, rtol=tol, maxiter=maxiter,
                       callback=lambda xk: its.__setitem__(0, its[0] + 1))
    res = np.linalg.norm(K @ sol - rhs) / np.linalg.norm(rhs)
    if verbose:
        print(f"[fem3d] MINRES {its[0]} iterations, relative residual {res:.2e}, "
              f"{time.time() - t0:.1f} s")
    if info != 0 or res > 1e-5:
        raise RuntimeError(f"FEM solve did not converge (info={info}, residual={res:.2e})")

    ur = np.zeros(nur)
    ur[free_u] = sol[:nf]
    pr = np.zeros(npr_r)
    pr[free_p] = sol[nf:]
    u, p = Pu @ ur, Pp @ pr

    # ---- sample --------------------------------------------------------------
    pts = np.asarray(sample_points, dtype=np.float64).T          # (3, N)
    cells = _locate(m, pts)
    uv = _sample(ub, u, pts, 3, cells)
    pv = _sample(pb, p, pts, 1, cells)[0]
    pv = pv - np.nanmean(pv)
    n_out = int(np.isnan(uv[0]).sum())
    if verbose:
        peak_gb = _peak_memory_gb()
        mem = f", peak memory {peak_gb:.1f} GB" if peak_gb is not None else ""
        print(f"[fem3d] total {time.time() - t0:.1f} s{mem}")
        print(f"[fem3d] peak u_x = {np.nanmax(uv[0]):.4f}   max|u_y| = {np.nanmax(np.abs(uv[1])):.4f}   "
              f"max|u_z| = {np.nanmax(np.abs(uv[2])):.4f}   points outside the mesh: {n_out}")
    return uv[0], uv[1], uv[2], pv


def _peak_memory_gb():
    """Peak memory of this process in GB, or None if it cannot be read.
    `resource` exists only on Linux/macOS; on Windows psutil is used if installed."""
    try:
        import resource
        import sys
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024**3 if sys.platform == "darwin" else kb / 1024**2   # macOS reports bytes
    except ImportError:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / 1024**3          # Windows
    except (ImportError, AttributeError):
        return None


def _locate(mesh, pts, chunk=4000, tol=1e-10):
    """Tetrahedron containing each point (3, N); -1 if none.

    Candidates = the tets with the nearest centroids (40, then 400 for points
    not found); a point is inside when all 4 barycentric coordinates >= -tol.
    scikit-fem's own finder tests every point against every candidate of the
    chunk, which runs out of memory on fine meshes.
    """
    from scipy.spatial import cKDTree

    p, t = mesh.p, mesh.t
    v0 = p[:, t[0]].T                                              # (ntet, 3)
    J = np.stack([p[:, t[1]] - p[:, t[0]],
                  p[:, t[2]] - p[:, t[0]],
                  p[:, t[3]] - p[:, t[0]]], axis=-1).transpose(1, 0, 2)   # (ntet, 3, 3)
    Jinv = np.linalg.inv(J)
    tree = cKDTree(p[:, t].mean(axis=1).T)

    cells = np.full(pts.shape[1], -1, dtype=np.int64)
    for k in (40, 400):
        todo = np.flatnonzero(cells < 0)
        for s0 in range(0, todo.size, chunk):
            idx = todo[s0:s0 + chunk]
            X = pts[:, idx].T                                       # (n, 3)
            _, cand = tree.query(X, k=min(k, t.shape[1]))          # (n, k)
            lam = np.einsum("nkij,nkj->nki", Jinv[cand], X[:, None, :] - v0[cand])
            inside = (lam >= -tol).all(-1) & (1.0 - lam.sum(-1) >= -tol)
            hit = inside.any(axis=1)
            cells[idx[hit]] = cand[hit, inside[hit].argmax(axis=1)]
    return cells


def _sample(basis, vals, pts, ncomp, cells):
    """Evaluate a FEM field at points (3, N) -> (ncomp, N); NaN where cells < 0.

    Same as scikit-fem's Basis.probes, with the containing cells given.
    """
    out = np.full((ncomp, pts.shape[1]), np.nan)
    ok = np.flatnonzero(cells >= 0)
    if ok.size == 0:
        return out
    x, c = pts[:, ok], cells[ok]
    loc = basis.mapping.invF(x[:, :, np.newaxis], tind=c)
    phis = np.array([basis.elem.gbasis(basis.mapping, loc, k, tind=c)[0]
                     for k in range(basis.Nbfun)]).flatten()
    rows = np.tile(np.arange(ncomp * ok.size), basis.Nbfun)
    cols = basis.element_dofs[:, np.tile(c, ncomp)].flatten()
    from scipy.sparse import coo_matrix
    probe = coo_matrix((phis, (rows, cols)), shape=(ncomp * ok.size, basis.N)).tocsr()
    out[:, ok] = (probe @ vals).reshape(ncomp, ok.size)
    return out


# --------------------------------------------------------------------------
# cached entry point
# --------------------------------------------------------------------------
def get(sample_points, cache_path, *, refit=False, **kwargs):
    """Cached wrapper around solve(). kwargs pass straight through."""
    cache_path = Path(cache_path)
    fp = _fingerprint(kwargs, sample_points)
    if not refit and cache_path.exists():
        d = np.load(cache_path, allow_pickle=False)
        if str(d["fingerprint"]) == fp:
            print(f"[fem3d] loaded cached solution from {cache_path}")
            return d["u1"], d["u2"], d["u3"], d["p"]
        print("[fem3d] fingerprint mismatch -> re-solving")
    print("[fem3d] solving 3D Stokes with scikit-fem ...")
    u1, u2, u3, p = solve(sample_points, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, u1=u1, u2=u2, u3=u3, p=p, fingerprint=np.asarray(fp))
    print(f"[fem3d] wrote {cache_path}")
    return u1, u2, u3, p


# --------------------------------------------------------------------------
# check: straight pipe against exact Poiseuille
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Check the 3D FEM solver on the straight pipe "
                                             "against exact Poiseuille flow.")
    ap.add_argument("--disk-refs", type=int, default=3)
    ap.add_argument("--nx", type=int, default=32)
    args = ap.parse_args()

    L_, w_, eta_, F_ = 2.5, 1.0, 1.0, 30.0 / 2.5
    R0_ = w_ / 2
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.uniform(0, L_, n)
    r = 0.98 * R0_ * np.sqrt(rng.uniform(0, 1, n))
    t = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack([x, r * np.cos(t), r * np.sin(t)], axis=-1)

    u1, u2, u3, _ = solve(pts, L=L_, a=0.0, avg_width=w_, eta=eta_, body_force_x=F_,
                          disk_refs=args.disk_refs, nx=args.nx)
    ux = F_ * (R0_**2 - r**2) / (4 * eta_)
    ok = np.isfinite(u1)
    e = u1[ok] - ux[ok]
    print(f"[check] straight pipe, {ok.sum()} points: max|u_x - exact| = {np.abs(e).max():.3e} "
          f"({np.abs(e).max() / ux.max():.2%} of peak), "
          f"max|u_y| = {np.nanmax(np.abs(u2)):.2e}, max|u_z| = {np.nanmax(np.abs(u3)):.2e}")