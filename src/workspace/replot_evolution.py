import argparse
from pathlib import Path
import numpy as onp
import fem_stokes, evolution_plot

p = argparse.ArgumentParser()
p.add_argument("--outdir", required=True, help="the original OUTDIR of the run")
p.add_argument("--dt", type=float, default=0.01)
p.add_argument("--n-loop", type=int, default=40)
p.add_argument("--n-snap", type=int, default=3)
p.add_argument("--n-artificial", type=int, default=340)
a_ = p.parse_args()

OUT = Path(a_.outdir)
L, a, avg_width, eta, rho = 2.5, 0.2, 1.0, 1.0, 1.0   # sinusoidal
FBODY_X = 30.0 / L
NX, NY, MARGIN = 120, 41, 0.98                         # --fem caps margin at 0.98

def width(x): return avg_width + 2*a*onp.sin(2*onp.pi*x/L)
x_line = onp.linspace(0, L, NX)
eta_line = onp.linspace(-MARGIN, MARGIN, NY) * 0.5
XX, EE = onp.meshgrid(x_line, eta_line, indexing="ij")
YY = EE * width(XX)
R_test = onp.stack([XX.ravel(), YY.ravel()], axis=-1)

# same snapshot rule as the main script
steps = sorted({max(1, int(round(k*a_.n_loop/a_.n_snap))) for k in range(1, a_.n_snap+1)})
snapshots = []
for n in steps:
    d = onp.load(OUT / f"state_t{n:04d}.npz")
    U1, U2, S1 = d["u1"], d["u2"], d["u1_std"]
    snapshots.append((n, float(d["t"]), U1, U2, S1, onp.zeros_like(S1)))  # u2_std not saved

fem_march = fem_stokes.get_unsteady(
    R_test, list(range(1, a_.n_loop+1)), OUT / "fem_unsteady_reference.npz",
    refit=False, L=L, a=a, avg_width=avg_width, eta=eta, rho=rho,
    body_force_x=FBODY_X, dt=a_.dt, nx=200, ny=80)

fp = next(OUT.glob("march_history_*.npz")).stem.replace("march_history_", "")
out2 = OUT / "replot_n3"; out2.mkdir(exist_ok=True)

evolution_plot.plot_evolution_vs_fem(
    snapshots, out2, XX=XX, YY=YY, L=L, a=a, avg_width=avg_width,
    fem=fem_march, cmap="RdBu_r", tag=fp, dt=a_.dt, n_artificial=a_.n_artificial)