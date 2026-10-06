"""Find the steady-state time t* of the unsteady Stokes start-up problem
with FEM, using the SAME setup as the PIGP march:

    rho*(u^n - u^{n-1})/dt = -grad p^n + eta*lap u^n + F,   div u^n = 0
    u^0 = 0 (rest), no-slip walls, x-periodic, F = (-dP/L, 0)

Backward Euler, Taylor-Hood P2/P1 (scikit-fem), exact periodicity via a
DOF-identification matrix. The time-stepping LHS is factorized ONCE.

t* = first time the relative L2 distance to the STEADY FEM solution drops
below --tol and stays below it for --window-t time units (dt-invariant).

Usage:
    python fem_steady_time.py --dt 0.01
    python fem_steady_time.py --dt 0.01 0.005 --geometry plates
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from skfem import (MeshTri, Basis, ElementVector, ElementTriP2, ElementTriP1,
                   BilinearForm, LinearForm)
from skfem.helpers import grad, div, dot

# ---------------------------------------------------------------- settings
# identical to the PIGP script
ETA = 1.0
RHO = 1.0
L = 2.5
AVG_WIDTH = 1.0
DP = -30.0
FX = -DP / L                      # body force x = 12
TEST_MARGIN = 0.98                # PIGP caps the test grid at 0.98 with --fem
NX_TEST, NY_TEST = 120, 41        # PIGP test grid

parser = argparse.ArgumentParser(description="FEM steady time t* (PIGP setup)")
parser.add_argument("--dt", type=float, nargs="+", default=[0.01],
                    help="one or more time steps, e.g. --dt 0.01 0.005")
parser.add_argument("--geometry", choices=["sinusoidal", "plates"],
                    default="sinusoidal")
parser.add_argument("--tol", type=float, default=1e-2,
                    help="rel L2 to steady FEM below which flow is steady")
parser.add_argument("--window-t", type=float, default=0.05,
                    help="time span the criterion must hold")
parser.add_argument("--t-max", type=float, default=3.0,
                    help="give up after this time")
parser.add_argument("--nx", type=int, default=200)
parser.add_argument("--ny", type=int, default=80)
parser.add_argument("--outdir", type=str, default="Stokes-Flow-PIGP-Workspace/src/workspace/outputs/outputs_fem_steady")
args = parser.parse_args()

A_WALL = 0.2 if args.geometry == "sinusoidal" else 0.0


def wid(x):
    return AVG_WIDTH + 2.0 * A_WALL * np.sin(2.0 * np.pi * x / L)


# ---------------------------------------------------------------- mesh
def build_mesh(nx, ny):
    m0 = MeshTri.init_tensor(np.linspace(0.0, L, nx + 1),
                             np.linspace(-0.5, 0.5, ny + 1))
    p = m0.p.copy()
    p[1] = p[1] * wid(p[0]) / AVG_WIDTH
    return MeshTri(p, m0.t)


def periodic_prolongation(basis, n_comp, tol=1e-8):
    """P (N_full x N_red): dofs at x=L are copies of the matching dof at x=0
    (same y, same vector component). u_full = P @ u_red."""
    X = basis.doflocs
    N = basis.N
    comp = np.arange(N) % n_comp          # skfem ElementVector interleaves
    left = np.where(np.abs(X[0]) < tol)[0]
    right = np.where(np.abs(X[0] - L) < tol)[0]

    master = np.arange(N)
    for r in right:
        c = left[(comp[left] == comp[r]) & (np.abs(X[1, left] - X[1, r]) < tol)]
        if c.size != 1:
            raise RuntimeError(f"periodic match failed for dof {r} ({c.size} hits)")
        master[r] = c[0]

    keep = np.setdiff1d(np.arange(N), right)
    red = -np.ones(N, dtype=int)
    red[keep] = np.arange(keep.size)
    cols = red[master]
    P = sp.csr_matrix((np.ones(N), (np.arange(N), cols)), shape=(N, keep.size))
    return P, cols


def main_for_dt(dt, outdir):
    t0 = time.time()
    m = build_mesh(args.nx, args.ny)
    ub = Basis(m, ElementVector(ElementTriP2()), intorder=4)
    pb = ub.with_element(ElementTriP1())
    nu, npr = ub.N, pb.N

    # sanity check on the interleaved-component assumption
    d = ub.get_dofs(lambda x: x[0] > -1.0)
    assert np.all(d.nodal["u^1"] % 2 == 0) and np.all(d.nodal["u^2"] % 2 == 1)

    @BilinearForm
    def vlap(u, v, w):
        return ETA * np.einsum("ij...,ij...", grad(u), grad(v))

    @BilinearForm
    def mass(u, v, w):
        return dot(u, v)

    @BilinearForm
    def divergence(u, q, w):
        return -q * div(u)

    @LinearForm
    def body(v, w):
        return FX * v[0]

    A = vlap.assemble(ub)
    M = mass.assemble(ub)
    B = divergence.assemble(ub, pb)
    f = body.assemble(ub)
    c = RHO / dt

    K_ss = sp.bmat([[A, B.T], [B, None]], format="csr")
    K_dt = sp.bmat([[c * M + A, B.T], [B, None]], format="csr")

    Pu, cu = periodic_prolongation(ub, 2)
    Pp, cp = periodic_prolongation(pb, 1)
    P = sp.block_diag([Pu, Pp], format="csr")
    full_to_red = np.concatenate([cu, Pu.shape[1] + cp])

    # Dirichlet: no-slip walls (all components) + pin one pressure dof
    wall = ub.get_dofs(
        lambda x: np.abs(np.abs(x[1]) - wid(x[0]) / 2.0) < 1e-3).all()
    D = np.unique(np.concatenate([full_to_red[wall], [full_to_red[nu]]]))
    n_red = P.shape[1]
    I = np.setdiff1d(np.arange(n_red), D)

    def reduce_factor(K):
        Kr = (P.T @ K @ P).tocsr()
        return splu(Kr[I][:, I].tocsc())

    def solve(lu, F):
        xr = np.zeros(n_red)
        xr[I] = lu.solve((P.T @ F)[I])
        return P @ xr

    F0 = np.concatenate([f, np.zeros(npr)])
    u_s = solve(reduce_factor(K_ss), F0)[:nu]
    lu = reduce_factor(K_dt)

    # PIGP test grid (u_x only, same metric as PIGP's fem_rel_l2)
    x_line = np.linspace(0.0, L, NX_TEST)
    e_line = np.linspace(-TEST_MARGIN, TEST_MARGIN, NY_TEST) * 0.5
    XX, EE = np.meshgrid(x_line, e_line, indexing="ij")
    pts = np.stack([XX.ravel(), (EE * wid(XX)).ravel()])
    Pr = ub.probes(pts).tocsr()[: pts.shape[1]]          # u_x rows
    ux_s = Pr @ u_s
    norm_grid = np.linalg.norm(ux_s)
    norm_M = np.sqrt(u_s @ (M @ u_s))

    print(f"\n[dt={dt}] {args.geometry}  mesh {args.nx}x{args.ny}  "
          f"{n_red} reduced dofs  setup {time.time()-t0:.1f}s")
    print(f"[steady] u_x max on grid = {ux_s.max():.5f}")
    print(f"{'step':>6}{'t':>9}{'rel_grid':>12}{'rel_M':>12}{'max|du|/max|u|':>16}")

    hist = []
    u = np.zeros(nu)
    n_max = int(round(args.t_max / dt))
    n_win = int(np.ceil(args.window_t / dt - 1e-9))
    t_star, n_below = None, 0
    pr_every = max(1, int(round(0.05 / dt)))

    for n in range(1, n_max + 1):
        F = np.concatenate([c * (M @ u) + f, np.zeros(npr)])
        u_new = solve(lu, F)[:nu]
        e = u_new - u_s
        rel_grid = np.linalg.norm(Pr @ u_new - ux_s) / norm_grid
        rel_M = np.sqrt(e @ (M @ e)) / norm_M
        step_chg = np.abs(u_new - u).max() / max(np.abs(u_new).max(), 1e-14)
        u = u_new
        t = n * dt
        hist.append((n, t, rel_grid, rel_M, step_chg))
        if n % pr_every == 0:
            print(f"{n:>6}{t:>9.4f}{rel_grid:>12.4e}{rel_M:>12.4e}{step_chg:>16.4e}")

        n_below = n_below + 1 if rel_grid < args.tol else 0
        if n_below == 1:
            t_cand = t
        if n_below >= n_win + 1:          # held for >= window_t
            t_star = t_cand
            break

    print("-" * 55)
    if t_star is None:
        print(f"[dt={dt}] NOT steady within t_max={args.t_max}")
    else:
        print(f"[dt={dt}] t* = {t_star:.4f}  (step {int(round(t_star/dt))}), "
              f"rel_grid < {args.tol:g} held for {args.window_t:g}")

    tag = f"{args.geometry}_dt{dt}_tol{args.tol:g}"
    with open(outdir / f"fem_steady_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "t", "rel_grid_ux", "rel_M_u", "step_change"])
        w.writerows(hist)
    return t_star, np.array(hist)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(__file__).resolve().parent / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    summary = []
    for dt in args.dt:
        t_star, h = main_for_dt(dt, outdir)
        summary.append((dt, t_star))
        ax.semilogy(h[:, 1], h[:, 2], label=rf"$\Delta t$={dt}"
                    + ("" if t_star is None else rf", $t^*$={t_star:.3f}"))
        if t_star is not None:
            ax.axvline(t_star, ls=":", lw=1)
    ax.axhline(args.tol, color="k", ls="--", lw=1, label=f"tol = {args.tol:g}")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|u_x^n - u_x^{\rm steady}\|_2 / \|u_x^{\rm steady}\|_2$")
    ax.set_title(f"FEM start-up, {args.geometry}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"fem_steady_{args.geometry}_tol{args.tol:g}.png", dpi=150)

    print("\n" + "=" * 40)
    print(f"{'dt':>10}{'t*':>12}{'steps':>10}")
    for dt, ts in summary:
        print(f"{dt:>10g}" + (f"{'n/a':>12}{'n/a':>10}" if ts is None
              else f"{ts:>12.4f}{int(round(ts/dt)):>10}"))
    print("=" * 40)