import argparse
import re
import traceback
from pathlib import Path

import numpy as onp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fem_stokes
import evolution_plot

# 20260927_usf_{dt}_{n_loop}_{n_artificial}artificial_{geometry}_nm_...
NAME_RE = re.compile(
    r"usf_(?P<dt>[\d.]+)_(?P<nloop>\d+)_(?P<nart>\d+)artificial_(?P<geom>sinusoidal|plates)_")

p = argparse.ArgumentParser()
p.add_argument("--root", default=str(Path(__file__).resolve().parent / "outputs"))
p.add_argument("--n-snap", type=int, default=3)
p.add_argument("--skip-existing", action="store_true",
               help="skip runs that already have replot_n{n_snap}/")
args = p.parse_args()

L, avg_width, eta, rho = 2.5, 1.0, 1.0, 1.0
FBODY_X = 30.0 / L
NX, NY, MARGIN = 120, 41, 0.98          # --fem caps the test margin at 0.98


def replot(run: Path):
    m = NAME_RE.search(run.name)
    if m is None:
        print(f"[skip] name not recognised: {run.name}")
        return
    dt, n_loop = float(m["dt"]), int(m["nloop"])
    n_art, geom = int(m["nart"]), m["geom"]
    a = 0.2 if geom == "sinusoidal" else 0.0

    out2 = run / f"replot_n{args.n_snap}"
    if args.skip_existing and out2.exists():
        print(f"[skip] exists: {out2}")
        return

    hist = list(run.glob("march_history_*.npz"))
    if not hist:
        print(f"[skip] no march_history (run unfinished?): {run.name}")
        return
    fp = hist[0].stem.replace("march_history_", "")

    width = lambda x: avg_width + 2*a*onp.sin(2*onp.pi*x/L)
    XX, EE = onp.meshgrid(onp.linspace(0, L, NX),
                          onp.linspace(-MARGIN, MARGIN, NY)*0.5, indexing="ij")
    YY = EE * width(XX)
    R_test = onp.stack([XX.ravel(), YY.ravel()], axis=-1)

    steps = sorted({max(1, int(round(k*n_loop/args.n_snap)))
                    for k in range(1, args.n_snap+1)})
    snaps = []
    for n in steps:
        d = onp.load(run / f"state_t{n:04d}.npz")
        snaps.append((n, float(d["t"]), d["u1"], d["u2"]))

    fem = fem_stokes.get_unsteady(
        R_test, list(range(1, n_loop+1)), run / "fem_unsteady_reference.npz",
        refit=False, L=L, a=a, avg_width=avg_width, eta=eta, rho=rho,
        body_force_x=FBODY_X, dt=dt, nx=200, ny=80)

    out2.mkdir(exist_ok=True)
    figs = evolution_plot.plot_evolution_vs_fem(
        snaps, out2, XX=XX, YY=YY, L=L, a=a, avg_width=avg_width,
        fem=fem, cmap="RdBu_r", tag=fp, dt=dt, n_artificial=n_art)
    for f in figs:
        if f is not None:
            plt.close(f)
    print(f"[done] {run.parent.name}/{run.name}  (steps {steps})")


runs = sorted(r for r in Path(args.root).glob("*/*") if r.is_dir())
print(f"[info] {len(runs)} run folders under {args.root}")
failed = []
for run in runs:
    try:
        replot(run)
    except Exception:
        traceback.print_exc()
        failed.append(run)
print(f"\n[summary] {len(runs)-len(failed)} ok, {len(failed)} failed")
for r in failed:
    print(f"  FAILED: {r}")