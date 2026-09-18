"""
Compare PIGP centreline u_x(t) against the analytical unsteady plane-Poiseuille
solution (Eqs. 59-62), using the SAME physical parameters as the PIGP run.

Usage:
    python compare_pigp_analytical.py \
        --history outputs/usf_dt0.05_10loops_270artificial_plates/march_history_<fp>.npz \
        --eta 1.0 --rho 1.0 --avg-width 1.0 --dP -30 --L 2.5
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--history", required=True, help="path to march_history_*.npz")
p.add_argument("--eta", type=float, required=True, help="viscosity, matches PIGP eta")
p.add_argument("--rho", type=float, required=True, help="density, matches PIGP rho")
p.add_argument("--avg-width", type=float, required=True, help="channel height b, matches PIGP avg_width")
p.add_argument("--dP", type=float, required=True, help="pressure drop, matches PIGP DeltaP")
p.add_argument("--L", type=float, required=True, help="channel length, matches PIGP L")
p.add_argument("--N", type=int, default=400, help="number of Fourier modes")
p.add_argument("--n-snap", type=int, default=5,
               help="number of evenly-spaced snapshots to show (0 = use every "
                    "point in the history file)")
p.add_argument("--out", default="pigp_vs_analytical.png")
args = p.parse_args()

# ---- physical parameters, derived exactly as in the PIGP script -----------
b    = args.avg_width
mu   = args.eta
rho  = args.rho
nu   = mu / rho
dpdx = args.dP / args.L          # NOT dP directly -- this is the conversion PIGP uses internally (FBODY[0] = -dP/L)
U    = 0.0                        # both walls stationary for Poiseuille

print(f"[params] b={b}  mu={mu}  rho={rho}  nu={nu:.4f}  dp/dx={dpdx:.4f}")
print(f"[params] diffusive time scale b^2/nu = {b**2/nu:.4f}")
print(f"[params] steady centreline u(b/2) = {-dpdx*b**2/(8*mu):.5f}  "
      f"<- should match your FEM 'steady reference' printout")


# ---- analytical solution, Eqs. (59)-(62) -----------------------------------
def u_steady(y, U, b, mu, dpdx):
    return y * U / b - (y / (2.0 * mu)) * dpdx * (b - y)

def A_n(n, U, b, mu, dpdx):
    sign = (-1.0) ** n
    return (2.0 * U * sign / (n * np.pi)
            + (2.0 / (b * mu)) * dpdx * (b / (n * np.pi)) ** 3 * (1.0 - sign))

def u_analytical(y, t, U, b, mu, rho, dpdx, N=400):
    nu = mu / rho
    yy = np.atleast_1d(y)[None, :, None]
    tt = np.atleast_1d(t)[:, None, None]
    n = np.arange(1, N + 1, dtype=float)[None, None, :]
    transient = np.sum(
        A_n(n, U, b, mu, dpdx) * np.exp(-(n**2) * np.pi**2 * nu * tt / b**2)
        * np.sin(n * np.pi * yy / b), axis=-1)
    return u_steady(np.atleast_1d(y), U, b, mu, dpdx)[None, :] + transient


# ---- load PIGP history ------------------------------------------------------
d = np.load(args.history)
t_pigp   = d["t"]
ctr_pigp = d["ux_ctr"]         # PIGP centreline-row mean, from your print loop

if args.n_snap and args.n_snap < len(t_pigp):
    # mirrors the PIGP script's own SNAP_STEPS rule (1-indexed steps ->
    # 0-indexed array positions), so these line up with your evolution_*.png
    n_loop = len(t_pigp)
    steps = sorted({max(1, round(k * n_loop / args.n_snap)) for k in range(1, args.n_snap + 1)})
    idx = [s - 1 for s in steps]
    t_pigp, ctr_pigp = t_pigp[idx], ctr_pigp[idx]
    print(f"[snap] showing steps {steps} of {n_loop} "
          f"(t = {', '.join(f'{tt:.3f}' for tt in t_pigp)})")

y_centre = np.array([b / 2.0])
ctr_analytical = u_analytical(y_centre, t_pigp, U, b, mu, rho, dpdx, N=args.N)[:, 0]

rel_err = np.abs(ctr_pigp - ctr_analytical) / np.abs(ctr_analytical)

print(f"\n{'t':>7}{'PIGP':>12}{'analytical':>14}{'rel err':>10}")
for tt, cp, ca, re in zip(t_pigp, ctr_pigp, ctr_analytical, rel_err):
    print(f"{tt:>7.3f}{cp:>12.5f}{ca:>14.5f}{re:>10.2%}")

# ---- plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(t_pigp, ctr_pigp,       "o-", label="PIGP")
ax.plot(t_pigp, ctr_analytical, "s--", label="Analytical (Eqs. 59-62)")
ax.axhline(-dpdx * b**2 / (8*mu), color="gray", ls=":", lw=1, label="steady value")
ax.set_xlabel("$t$")
ax.set_ylabel("centreline $u_x$")
ax.set_title("PIGP vs analytical unsteady Poiseuille flow")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"\n[plot] saved {args.out}")