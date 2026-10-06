"""dt-invariant steady-state time from FINISHED runs -- no re-run needed.

Steady = rel L2 distance to the STEADY FEM solution stays below --tol for
--window units of PHYSICAL time (not steps), for PIGP and for the same-dt
FEM march alike.

    uv run python steady_from_saved.py DIR_A DIR_B DIR_C
"""
import argparse
import re
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("outdirs", nargs="+")
ap.add_argument("--tol", type=float, default=2e-2)
ap.add_argument("--window", type=float, default=0.05)
ap.add_argument("--report-every", type=float, default=0.05)
A = ap.parse_args()


def rel_l2(U, F):
    g = np.isfinite(U) & np.isfinite(F)
    return float(np.linalg.norm(U[g] - F[g]) / np.linalg.norm(F[g]))


def first_sustained(t, v, tol, window, dt):
    need = max(1, int(round(window / dt)))
    for i in range(len(v) - need + 1):
        if np.all(v[i:i + need] < tol):
            return float(t[i])
    return None


rows = []
for d in map(Path, A.outdirs):
    steady = np.load(d / "fem_reference.npz")["u1"].ravel()

    files = sorted(d.glob("state_t*.npz"),
                   key=lambda p: int(re.search(r"t(\d+)", p.name).group(1)))
    t_p, e_p = [], []
    for f in files:
        z = np.load(f)
        t_p.append(float(z["t"]))
        e_p.append(rel_l2(z["u1"].ravel(), steady))
    t_p, e_p = np.array(t_p), np.array(e_p)
    dt = t_p[0]                                  # step 1 sits at t = dt

    t_f = e_f = None
    fu = d / "fem_unsteady_reference.npz"
    if fu.exists():
        z = np.load(fu)
        steps = sorted(int(s) for s in z["steps"])
        if steps != list(range(1, steps[-1] + 1)):
            print(f"[warn] {d.name}: FEM cache is missing steps -- "
                  f"window test unreliable")
        t_f = np.array(steps) * dt
        e_f = np.array([rel_l2(z[f"u1_{s}"].ravel(), steady) for s in steps])

    ts_p = first_sustained(t_p, e_p, A.tol, A.window, dt)
    ts_f = (first_sustained(t_f, e_f, A.tol, A.window, dt)
            if e_f is not None else None)
    rows.append((dt, ts_p, ts_f))

    print(f"\n=== {d.name}   (dt = {dt:g}) ===")
    k = max(1, int(round(A.report_every / dt)))
    print(f"  {'t':>7}{'PIGP relL2':>13}{'FEM relL2':>12}")
    for i in range(k - 1, len(t_p), k):
        fem_s = (f"{e_f[i]:12.4e}" if e_f is not None and i < len(e_f)
                 else f"{'--':>12}")
        print(f"  {t_p[i]:7.3f}{e_p[i]:13.4e}{fem_s}")

fmt = lambda v: "not reached" if v is None else f"{v:.4f}"
print("\n" + "=" * 60)
print(f"STEADY TIME  (rel L2 to steady FEM < {A.tol:g}, held {A.window:g})")
print("=" * 60)
print(f"  {'dt':>8}{'PIGP':>16}{'FEM (same dt)':>18}")
for dt, tp, tf in sorted(rows):
    print(f"  {dt:8g}{fmt(tp):>16}{fmt(tf):>18}")
print("=" * 60)