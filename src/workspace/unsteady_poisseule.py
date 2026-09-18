"""
Channel-style visualisation of impulsively-started plane Poiseuille flow.

Colour field is u_x(y,t) from Eqs. (59)-(62) of Munoz et al. (arXiv:1203.1037),
plotted over an (x, y) channel. NOTE: the analytical solution is independent
of x, so every vertical slice is identical by construction.

Coordinate convention: the paper uses y in [0, b] with walls at y = 0 and y = b.
Here the result is shifted to y in [-b/2, +b/2] to match the centred channel
layout of the target figure.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- parameters
b    = 1.0          # channel height
mu   = 1.0         # dynamic viscosity
rho  = 1.0          # density
nu   = mu / rho     # kinematic viscosity
dpdx = -12.0        # constant streamwise pressure gradient
U    = 0.0          # both walls stationary -> pure Poiseuille
N    = 400          # Fourier modes

Lx   = 2.5          # streamwise extent shown (arbitrary: solution is x-invariant)
times = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])


# ---------------------------------------------------------------- solution
def u_steady(y, U, b, mu, dpdx):
    """Eq. (59), y measured from the lower wall."""
    return y * U / b - (y / (2.0 * mu)) * dpdx * (b - y)


def A_n(n, U, b, mu, dpdx):
    """Eq. (61) evaluated analytically (sign-corrected form of Eq. 62)."""
    sign = (-1.0) ** n
    return (2.0 * U * sign / (n * np.pi)
            + (2.0 / (b * mu)) * dpdx * (b / (n * np.pi)) ** 3 * (1.0 - sign))


def u_total(y, t, U, b, mu, rho, dpdx, N=400):
    """Eqs. (59)+(60). Returns array of shape (len(t), len(y))."""
    nu = mu / rho
    yy = np.atleast_1d(y)[None, :, None]
    tt = np.atleast_1d(t)[:, None, None]
    n = np.arange(1, N + 1, dtype=float)[None, None, :]
    transient = np.sum(
        A_n(n, U, b, mu, dpdx)
        * np.exp(-(n ** 2) * np.pi ** 2 * nu * tt / b ** 2)
        * np.sin(n * np.pi * yy / b),
        axis=-1,
    )
    return u_steady(np.atleast_1d(y), U, b, mu, dpdx)[None, :] + transient


# ---------------------------------------------------------------- evaluate
y_wall = np.linspace(0.0, b, 400)          # paper coordinate, 0 .. b
y_plot = y_wall - b / 2.0                  # centred coordinate, -b/2 .. +b/2
x = np.linspace(0.0, Lx, 200)

U_t = u_total(y_wall, times, U, b, mu, rho, dpdx, N)      # (Nt, Ny)
u_inf = u_steady(y_wall, U, b, mu, dpdx)
vmax = np.abs(u_inf).max()

# ---------------------------------------------------------------- plot
fig, axes = plt.subplots(1, len(times), figsize=(15, 3.2), sharey=True)

for k, ax in enumerate(axes):
    field = np.tile(U_t[k][:, None], (1, x.size))         # (Ny, Nx)
    im = ax.pcolormesh(x, y_plot, field, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)

    # walls
    ax.axhline(-b / 2, color="k", lw=1.5)
    ax.axhline(+b / 2, color="k", lw=1.5)

    frac = 100.0 * np.trapezoid(U_t[k], y_wall) / np.trapezoid(u_inf, y_wall)
    ax.set_title(f"$t$ = {times[k]:.2f}\n{frac:.1f}% of steady flux", fontsize=9)
    ax.set_xlabel("$x$")
    ax.set_xticks([0, 1, 2, 2.5])
    ax.set_ylim(-0.75, 0.75)

axes[0].set_ylabel("$y$")
fig.suptitle("Unsteady plane Poiseuille flow  —  $u_x$  (analytical, Eqs. 59–62)",
             fontsize=11, y=1.04)

cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.015)
cbar.set_label("$u_x$")

fig.savefig("src/workspace/outputs/unsteady_poi/poiseuille_channel.png", dpi=170, bbox_inches="tight")
print("steady centreline u_max =", u_inf.max())
print("diffusive time scale b^2/nu =", b ** 2 / nu)
for k, t in enumerate(times):
    print(f"  t={t:6.2f}   centreline u = {U_t[k].max():.5f}")