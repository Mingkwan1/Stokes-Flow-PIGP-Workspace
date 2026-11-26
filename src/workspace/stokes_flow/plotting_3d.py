"""
Plots for main_steady_stokes_3d.py.

    plot_plane     u_x, u_y, u_z on a slice (x-y at z = 0, or x-z at y = 0):
                   mean, std, and |error| when an exact solution is given
    plot_profiles  u_x across the pipe at a fixed x, with +-2 sigma band
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp

COMP_LABEL = {"u1": "$u_x$", "u2": "$u_y$", "u3": "$u_z$"}


def plot_plane(XX, HH, pred, exact, vlabel, wall_x, wall_r, out):
    """
    XX, HH  : (nx, nh) grid, x and the transverse coordinate (y or z)
    pred    : {"u1": (mean, std), "u2": ..., "u3": ...}, flat arrays of size nx*nh
    exact   : reference {"u1": arr, "u2": arr, "u3": arr} or None (no error column);
              NaN entries (outside the FEM mesh) are left blank
    vlabel  : y-axis label, e.g. "$y$" or "$z$"
    wall_x, wall_r : wall outline, drawn at +-wall_r
    out     : output path
    Rows: u_x, u_y, u_z.  Columns: mean, std, |error|.
    """
    ncol = 3 if exact is not None else 2
    fig, axs = plt.subplots(3, ncol, figsize=(4.2 * ncol, 6.5), squeeze=False,
                            sharex=True, sharey=True)
    for i, c in enumerate(("u1", "u2", "u3")):
        mean, std = (onp.asarray(v).reshape(XX.shape) for v in pred[c])
        panels = [(mean, "RdBu_r", "mean"), (std, "viridis", "std")]
        if exact is not None:
            panels.append((onp.abs(mean - onp.asarray(exact[c]).reshape(XX.shape)),
                           "magma", "|error|"))
        for j, (Z, cmap, title) in enumerate(panels):
            ax = axs[i, j]
            if cmap == "RdBu_r":
                vmax = max(float(onp.abs(Z).max()), 1e-12)
                im = ax.pcolormesh(XX, HH, Z, cmap=cmap, vmin=-vmax, vmax=vmax,
                                   shading="gouraud")
            else:
                im = ax.pcolormesh(XX, HH, Z, cmap=cmap, shading="gouraud")
            ax.plot(wall_x, wall_r, "k", lw=1)
            ax.plot(wall_x, -wall_r, "k", lw=1)
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, shrink=0.8)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"{COMP_LABEL[c]}\n\n{vlabel}")
            if i == 2:
                ax.set_xlabel("$x$")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_profiles(profiles, x_profile, out, ref_label="Poiseuille"):
    """
    profiles  : {xlabel: (h, mean, std, reference_or_None)}, one panel per entry
    x_profile : x position of the lines (for the title)
    out       : output path
    ref_label : legend name of the reference curve (e.g. "Poiseuille", "FEM")
    """
    fig, axs = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4),
                            sharey=True, squeeze=False)
    for ax, (xlabel, (h, mean, std, ex)) in zip(axs[0], profiles.items()):
        ax.fill_between(h, mean - 2 * std, mean + 2 * std, alpha=0.3, label=r"$\pm 2\sigma$")
        ax.plot(h, mean, lw=2, label="PIGP mean")
        if ex is not None:
            ax.plot(h, ex, "k--", lw=1.5, label=ref_label)
        ax.set_xlabel(xlabel)
        ax.set_title(f"$u_x$ at $x = {x_profile:.3g}$")
        ax.grid(alpha=0.3)
    axs[0, 0].set_ylabel("$u_x$")
    axs[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")