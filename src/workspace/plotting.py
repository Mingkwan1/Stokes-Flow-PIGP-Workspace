from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm

# (mean_key, symbol, diverging) - p uses a sequential map because its sign is
# not physically meaningful: only grad(p) and periodicity are constrained.
FIELDS = [("u_x", r"$u_x$", True),
          ("u_y", r"$u_y$", True),
          ("p",   r"$p$",   False)]

REL_VLIM = (1e-3, 1e0)     # shared across all fields, as in the paper
ABS_VLIM = (1e-4, 1e-1)

@dataclass
class PlotCtx:
    XX: np.ndarray          # (NX, NY) test-grid x
    YY: np.ndarray          # (NX, NY) test-grid y
    NX: int
    NY: int
    L: float
    a: float
    avg_width: float
    R_wall: np.ndarray      # (n, 2) no-slip points
    outdir: Path
    tag: str = ""           # fingerprint, appended to filenames
    subdir: str = "base"                        
    sparse_pts: np.ndarray = None          # (N,2) velocity training points

    def width(self, x):
        return self.avg_width + 2*self.a*np.sin(2*np.pi*np.asarray(x)/self.L)

    def grid(self, flat):
        arr = np.asarray(flat, dtype=np.float64) 
        return arr.reshape(self.NX, self.NY)

    def save(self, fig, name):
        path = Path(self.outdir) / "plots" / f"{name}_{self.tag}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[plot] saved {path}")

def _draw_map(ctx, fig, ax, F, symbol, title, diverging=True, cmap=None,
              mark_wall=False, log=False, vlim=None):
    F = np.asarray(F, dtype=np.float64)
    finite = np.isfinite(F)
    if not finite.any():
        print(f"[plot] WARNING: '{title}' is entirely non-finite")

    if diverging:
        vmax = float(np.max(np.abs(F[finite]))) if finite.any() else 0.0
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = cmap or "RdBu_r"
    elif log:
        if vlim is not None:
            norm, extend = LogNorm(vmin=vlim[0], vmax=vlim[1]), "both"
        else:
            pos = F[finite & (F > 0)]
            norm = LogNorm(vmin=max(float(np.percentile(pos, 1)),
                                    float(pos.max())*1e-6),
                           vmax=float(pos.max())) if pos.size else None
        cmap = cmap or "magma"
    else:
        norm, cmap = None, cmap or "viridis"

    pc = ax.pcolormesh(ctx.XX, ctx.YY, F, cmap=cmap, norm=norm,
                       shading="gouraud")
    fig.colorbar(pc, ax=ax, label=symbol)

    xw = np.linspace(0.0, ctx.L, 400)
    w = ctx.width(xw)
    ax.plot(xw, w/2, "k", lw=1.4)
    ax.plot(xw, -w/2, "k", lw=1.4)
    if mark_wall:
        ax.scatter(ctx.R_wall[:, 0], ctx.R_wall[:, 1], s=6, c="k",
                   alpha=0.5, zorder=4, label="no-slip points")
    if ctx.sparse_pts is not None:
        ax.scatter(ctx.sparse_pts[:, 0], ctx.sparse_pts[:, 1], s=34,
                   c="lime", marker="o", edgecolors="k", linewidths=0.6,
                   zorder=5, label="velocity training points")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

def _draw_profiles(ctx, ax, F, S, symbol, title):
    for frac, c in zip([0.0, 0.25, 0.5, 0.75],
                       ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
        i = int(frac*(ctx.NX - 1))
        ax.plot(F[i], ctx.YY[i], color=c, lw=1.8, label=f"$x={ctx.XX[i,0]:.2f}$")
        if S is not None:
            ax.fill_betweenx(ctx.YY[i], F[i] - 2*S[i], F[i] + 2*S[i],
                             color=c, alpha=0.15, lw=0)
    ax.axvline(0.0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(symbol)
    ax.set_ylabel("$y$")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)


def _row_pigp(ctx, fig, axes, mean, std, symbol, diverging, legend=False):
    F = ctx.grid(mean)
    S = ctx.grid(std) if std is not None else None
    _draw_map(ctx, fig, axes[0], F, symbol, f"posterior mean {symbol}",
              diverging, mark_wall=(ctx.sparse_pts is None))
    if legend:
        axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    _draw_profiles(ctx, axes[1], F, S, symbol,
                   f"profiles{r' ($\pm2\sigma$)' if S is not None else ''}")

def error_metrics(P, F, name):
    """Return (panels, stats).
    panels: list of (array, colorbar_label, title_suffix, cmap, log, vlim)
    stats:  dict printed by _row_compare
    """
    absE = np.abs(P - F)
    with np.errstate(divide="ignore", invalid="ignore"):
        relE = absE / np.abs(F)
    relE = np.where(np.isfinite(relE), relE, REL_VLIM[1]*10)

    panels = [
        (absE, r"$|\Delta|$",  "abs error", "magma", True, ABS_VLIM),
        (relE, "rel. error",   "rel error", "magma", True, REL_VLIM),
    ]
    stats = {
        "max_abs": np.nanmax(absE),
        "rms":     np.sqrt(np.nanmean((P - F)**2)),
        "med_rel": np.median(relE[np.isfinite(relE)]),
        "l2_rel":  np.linalg.norm(P - F) / np.linalg.norm(F),   # e.g. add your own
    }
    return panels, stats

def _row_compare(ctx, fig, axes, mean, fem, symbol, name, diverging):
    P = ctx.grid(mean)
    F = np.asarray(fem, dtype=np.float64).reshape(ctx.NX, ctx.NY)
    if name == "p":
        P = P - np.nanmean(P)
        F = F - np.nanmean(F)

    _draw_map(ctx, fig, axes[0], P, symbol, f"PIGP {symbol}", diverging)
    _draw_map(ctx, fig, axes[1], F, symbol, f"FEM {symbol}",  diverging)

    panels, stats = error_metrics(P, F, name)
    for ax, (A, lab, ttl, cmap, log, vlim) in zip(axes[2:], panels):
        _draw_map(ctx, fig, ax, A, lab, f"{ttl} {symbol}", False,
                  cmap=cmap, log=log, vlim=vlim)

    print(f"  {name:<4} " + "  ".join(f"{k}={v:9.4g}" for k, v in stats.items()))


def plot_all(ctx, pigp, fem=None):
    """pigp: {'u_x': (mean, std), 'u_y': (...), 'p': (...)}
       fem:  {'u_x': array, ...} or None.
    Saves each field individually, then one combined figure."""
    if fem is None:
        for name, symbol, div in FIELDS:
            f, ax = plt.subplots(1, 2, figsize=(14, 4.6),
                                 gridspec_kw={"width_ratios": [2.4, 1]})
            _row_pigp(ctx, f, ax, *pigp[name], symbol, div, legend=True)
            f.tight_layout()
            ctx.save(f, name)
            plt.close(f)

        fig, axes = plt.subplots(3, 2, figsize=(14, 12.5),
                                 gridspec_kw={"width_ratios": [2.4, 1]})
        for r, (name, symbol, div) in enumerate(FIELDS):
            _row_pigp(ctx, fig, axes[r], *pigp[name], symbol, div, legend=(r == 0))
        combined = "combined"
    else:
        print("\n[error] PIGP vs FEM:")
        for name, symbol, div in FIELDS:
            f, ax = plt.subplots(1, 4, figsize=(24, 4.2))
            _row_compare(ctx, f, ax, pigp[name][0], fem[name], symbol, name, div)
            f.tight_layout()
            ctx.save(f, f"{name}_vs_fem")
            plt.close(f)

        fig, axes = plt.subplots(3, 4, figsize=(24, 12.5))
        for r, (name, symbol, div) in enumerate(FIELDS):
            _row_compare(ctx, fig, axes[r], pigp[name][0], fem[name],
                         symbol, name, div)
        combined = "combined_fem"

    fig.tight_layout()
    ctx.save(fig, combined)
    return fig