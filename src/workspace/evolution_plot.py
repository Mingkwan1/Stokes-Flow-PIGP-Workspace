"""Velocity evolution vs UNSTEADY FEM: two separate figures, one for u_x and
one for u_y. Each has the PIGP snapshots across the top row and the FEM
solution AT THE SAME TIME STEP along the bottom row -- both march the
identical backward-Euler scheme (same dt, same u^0=0), so this is a genuine
per-time comparison, not a comparison against a steady target.

Two ways to use it.

1. Imported by the march (see --evolution / --fem in unsteady_stokes_flow.py):

       import evolution_plot
       evolution_plot.plot_evolution_vs_fem(
           snaps, OUTDIR, XX=XX, YY=YY, L=L, a=a, avg_width=avg_width,
           fem=fem_results)      # fem_results: {step: (u1, u2, p)}, from
                                  # fem_stokes.get_unsteady(), or None

   snaps      : list of (step, t, U1, U2) -- NOTE the step index is now
                required so it can be matched against fem_results' keys.
   fem_results: dict {step: (u1_flat, u2_flat, p_flat)} sampled at the same
                R_test as `snaps`, one entry per snapshot step. Comes from
                fem_stokes.get_unsteady(R_test, SNAP_STEPS, ..., dt=Δt).

   Writes evolution_ux_{tag}.png and evolution_uy_{tag}.png.

2. Standalone, rebuilt from saved state_t*.npz -- no re-solve needed:

       python evolution_plot.py
       python evolution_plot.py --fem --dt 0.01   # needs fem_stokes.py
       python evolution_plot.py --n-snap 8 --cmap viridis
"""
import numpy as np
import matplotlib.pyplot as plt


def _one_field(times, U_list, fem_list, fname, *, XX, YY, L, a, avg_width,
              symbol, cmap="RdBu_r", title=None,
              error_cmap="hot", abs_vlim=None, rel_vlim=(1e-1, 1e0)):
    """One figure for one field (u_x OR u_y).
 
    Row 0: PIGP at each time.
    Row 1 (only if fem_list is given): FEM AT THE SAME TIME, one per column.
    Row 2 (only if fem_list is given): pointwise ABSOLUTE error |U-F|, log scale.
    Row 3 (only if fem_list is given): pointwise RELATIVE error |U-F|/|F|, log scale.
 
    fem_list: list of arrays (same length as U_list) or None entries where
              no FEM result exists for that step -- those columns are left
              blank in ALL fem/error rows rather than silently reusing a
              wrong value.
 
    Pointwise vs. the caption's error: the row-0 caption is a single GLOBAL
    relative-L2 number per column, ||U-F||/||F||. Rows 2-3 are a DIFFERENT,
    spatial quantity -- error AT EACH POINT -- so they can show local hot
    spots even where the caption number looks small.
 
    abs_vlim, rel_vlim: (lo, hi) log-scale bounds for rows 2 and 3, shared
        across all columns within that row. None -> computed from this
        figure's own data. rel_vlim defaults to (1e-1, 1e0): tighter than
        an auto range would give, so the relative-error row shows contrast
        instead of saturating -- widen it if a run's error is genuinely
        outside that band (you'll see solid black or solid yellow if so).
 
    Colorbar placement: computed EXPLICITLY from each row's own axis
    position (ax.get_position()) rather than matplotlib's automatic
    multi-axes colorbar stacking (fig.colorbar(ax=[...])), which the
    previous version used and which places stacked colorbars inconsistently
    across matplotlib versions when constrained_layout is combined with
    several such calls -- that was the cause of the overlapping colorbars /
    blank gap seen with 4 rows. Explicit positions can't drift like that.
    """
    XX, YY = np.asarray(XX), np.asarray(YY)
    ns = len(times)
    has_fem = fem_list is not None
    nrows = 4 if has_fem else 1
 
    xw = np.linspace(0.0, float(L), 400)
    wall = float(avg_width) + 2*float(a)*np.sin(2*np.pi*xw/float(L))
 
    if has_fem:
        fem_list = [np.asarray(f).reshape(XX.shape) if f is not None else None
                   for f in fem_list]
 
    all_fields = [np.asarray(u) for u in U_list]
    if has_fem:
        all_fields += [f for f in fem_list if f is not None]
    vlim = max(float(np.nanmax(np.abs(np.concatenate(
        [f.ravel() for f in all_fields])))), 1e-12)
 
    # NOTE: constrained_layout is OFF on purpose -- see docstring above.
    # Spacing is set explicitly via subplots_adjust instead.
    fig, axes = plt.subplots(nrows, ns, figsize=(3.0*ns + 1.9, 3.0*nrows + 0.6),
                             squeeze=False, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.90, bottom=0.07,
                        hspace=0.32, wspace=0.10)
 
    def _draw(ax, A):
        pc = ax.pcolormesh(XX, YY, np.asarray(A), cmap=cmap, shading="gouraud",
                           vmin=-vlim, vmax=vlim)
        ax.plot(xw,  wall/2, "k", lw=1.0)
        ax.plot(xw, -wall/2, "k", lw=1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, float(L)/2, float(L)])
        return pc
 
    for c, (t, U) in enumerate(zip(times, U_list)):
        ax = axes[0][c]
        pc = _draw(ax, U)
        cap = f"$t={t:.3f}$"
        if has_fem and fem_list[c] is not None:
            good = np.isfinite(fem_list[c]) & np.isfinite(np.asarray(U))
            F = fem_list[c]
            rel = (np.linalg.norm(np.asarray(U)[good] - F[good])
                  / np.linalg.norm(F[good]))
            cap += f"\nrel $L_2$={rel:.1%}"
        ax.set_title(cap, fontsize=10)
        if c == 0:
            ax.set_ylabel("PIGP\n$y$", fontsize=11)
        if not has_fem:
            ax.set_xlabel("$x$")
 
    if has_fem:
        for c in range(ns):
            ax = axes[1][c]
            if fem_list[c] is not None:
                pc = _draw(ax, fem_list[c])
            else:
                ax.text(0.5, 0.5, "no FEM\nat this step", ha="center",
                       va="center", transform=ax.transAxes, fontsize=9,
                       color="0.5")
                ax.set_aspect("equal", adjustable="box")
            for spine in ax.spines.values():
                spine.set_linestyle((0, (4, 2)))
                spine.set_edgecolor("0.3")
            if c == 0:
                ax.set_ylabel("FEM\n$y$", fontsize=11)
 
    if has_fem:
        from matplotlib.colors import LogNorm
 
        abs_err_list, rel_err_list = [], []
        for c in range(ns):
            if fem_list[c] is None:
                abs_err_list.append(None)
                rel_err_list.append(None)
                continue
            U = np.asarray(U_list[c])
            F = fem_list[c]
            good = np.isfinite(U) & np.isfinite(F)
 
            abs_err = np.full_like(U, np.nan, dtype=float)
            abs_err[good] = np.abs(U[good] - F[good])
 
            denom = np.abs(F[good])
            rel_err = np.full_like(U, np.nan, dtype=float)
            rel_err[good] = np.where(denom == 0.0, 1.0, abs_err[good] / np.where(denom == 0.0, 1.0, denom))
 
            abs_err_list.append(abs_err)
            rel_err_list.append(rel_err)
 
        def _auto_bounds(arrs):
            vals = np.concatenate(
                [a[np.isfinite(a) & (a > 0)].ravel()
                 for a in arrs if a is not None])
            if vals.size == 0:
                return 1e-6, 1.0
            hi = float(np.nanmax(vals))
            lo = max(float(np.nanpercentile(vals, 1)), hi * 1e-4)
            return lo, hi
 
        abs_lo, abs_hi = abs_vlim if abs_vlim is not None else _auto_bounds(abs_err_list)
        rel_lo, rel_hi = rel_vlim if rel_vlim is not None else _auto_bounds(rel_err_list)
 
        def _draw_err(ax, A, lo, hi):
            if A is None or not np.isfinite(A).any():
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                       transform=ax.transAxes, fontsize=9, color="0.5")
                ax.set_aspect("equal", adjustable="box")
                return None
            Ac = np.clip(A, lo, hi)   # LogNorm needs strictly positive, finite
            pc = ax.pcolormesh(XX, YY, Ac, cmap=error_cmap,
                               norm=LogNorm(vmin=lo, vmax=hi), shading="gouraud")
            ax.plot(xw,  wall/2, "w", lw=0.8, alpha=0.6)
            ax.plot(xw, -wall/2, "w", lw=0.8, alpha=0.6)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([0, float(L)/2, float(L)])
            return pc
 
        pc_abs = pc_rel = None
        for c in range(ns):
            pc_abs = _draw_err(axes[2][c], abs_err_list[c], abs_lo, abs_hi)
            if c == 0:
                axes[2][c].set_ylabel(f"$|\\Delta|$\n$y$", fontsize=11)
 
            pc_rel = _draw_err(axes[3][c], rel_err_list[c], rel_lo, rel_hi)
            if c == 0:
                axes[3][c].set_ylabel("rel. err.\n$y$", fontsize=11)
 
    for c in range(ns):
        axes[-1][c].set_xlabel("$x$")
 
    # ---- explicit colorbar placement, one call per row-group, positions
    # taken directly from the ACTUAL rendered axes (ax.get_position()) so
    # nothing can silently overlap regardless of matplotlib version.
    cbar_x, cbar_w, gap = 0.885, 0.018, 0.10   # gap: fraction of each band
                                                # trimmed off top+bottom so
                                                # neighbouring colorbars
                                                # never touch
    def _row_band(row_indices):
        y0 = min(axes[r][0].get_position().y0 for r in row_indices)
        y1 = max(axes[r][0].get_position().y1 for r in row_indices)
        pad = gap * (y1 - y0)
        return y0 + pad, y1 - pad
 
    y0, y1 = _row_band([0, 1] if has_fem else [0])
    cax = fig.add_axes([cbar_x, y0, cbar_w, y1 - y0])
    fig.colorbar(pc, cax=cax, label=symbol)
 
    if has_fem:
        if pc_abs is not None:
            y0, y1 = _row_band([2])
            cax = fig.add_axes([cbar_x, y0, cbar_w, y1 - y0])
            fig.colorbar(pc_abs, cax=cax,
                        label=rf"$|\Delta {symbol[1:-1]}|$ (log)")
        if pc_rel is not None:
            y0, y1 = _row_band([3])
            cax = fig.add_axes([cbar_x, y0, cbar_w, y1 - y0])
            fig.colorbar(pc_rel, cax=cax, label="rel. error (log)")
 
    if title:
        fig.suptitle(title, fontsize=11, y=0.98)
    fig.savefig(fname, dpi=140)   # NOTE: no bbox_inches="tight" -- it can
                                  # re-crop stacked colorbars after the fact
                                  # and reintroduce exactly this kind of
                                  # overlap; explicit-position layout above
                                  # is already final.
    print(f"[evo] wrote {fname}  ({ns} snapshots"
          f"{', vs unsteady FEM' if has_fem else ''})")
    return fig


def plot_evolution_vs_fem(snaps, outdir, *, XX, YY, L, a, avg_width,
                          fem=None, cmap="RdBu_r", tag="", dt=None,
                          n_artificial=None, error_cmap="hot",
                          abs_vlim=None, rel_vlim=(1e-3, 1e0)):
    """Two figures: evolution_ux_{tag}.png and evolution_uy_{tag}.png.

    snaps : list of (step, t, U1, U2). `step` must match the keys used in
            `fem` so each PIGP column pairs with the FEM result at the SAME
            time step, not a steady reference.
    fem   : optional dict {step: (u1, u2, p)}, same grid as snaps, from
            fem_stokes.get_unsteady(). A step present in `snaps` but absent
            from `fem` renders as a blank "no FEM at this step" panel rather
            than silently falling back to something else.
    """
    from pathlib import Path
    outdir = Path(outdir)

    if not snaps:
        print("[evo] no snapshots given -- nothing plotted. "
              "Check that the march appends to its snapshot list.")
        return None, None

    steps  = [s[0] for s in snaps]
    times  = [s[1] for s in snaps]
    U1_list = [s[2] for s in snaps]
    U2_list = [s[3] for s in snaps]

    fem1 = fem2 = None
    if fem is not None:
        fem1 = [np.asarray(fem[st][0]) if st in fem else None for st in steps]
        fem2 = [np.asarray(fem[st][1]) if st in fem else None for st in steps]
        missing = [st for st in steps if st not in fem]
        if missing:
            print(f"[evo] WARNING: no FEM result for steps {missing} "
                  f"-- those panels will be blank, not steady-state filler")

    suffix = f"_{tag}" if tag else ""
    base_title = None
    if dt is not None:
        base_title = rf"$\Delta t={dt}$, backward Euler, matched FEM"
        if n_artificial is not None:
            base_title += rf", {n_artificial} artificial points/step"

    fig_ux = _one_field(
        times, U1_list, fem1, outdir / f"evolution_ux{suffix}.png",
        XX=XX, YY=YY, L=L, a=a, avg_width=avg_width, symbol=r"$u_x$",
        cmap=cmap,error_cmap=error_cmap, abs_vlim=abs_vlim, rel_vlim=rel_vlim,
        title=(f"$u_x$ evolution" + (f" -- {base_title}" if base_title else "")))

    fig_uy = _one_field(
        times, U2_list, fem2, outdir / f"evolution_uy{suffix}.png",
        XX=XX, YY=YY, L=L, a=a, avg_width=avg_width, symbol=r"$u_y$",
        cmap=cmap,error_cmap=error_cmap, abs_vlim=abs_vlim, rel_vlim=rel_vlim,
        title=(f"$u_y$ evolution" + (f" -- {base_title}" if base_title else "")))

    return fig_ux, fig_uy


def snapshots_from_states(directory, n_snap=5):
    """Read state_t*.npz written by the march, return (snaps, NX, NY).

    snaps is (step, t, U1, U2) -- step comes from the filename so it can be
    matched against an FEM results dict keyed the same way.
    """
    import re
    from pathlib import Path

    d = Path(directory)
    files = sorted(d.glob("state_t*.npz"),
                   key=lambda p: int(re.search(r"t(\d+)", p.name).group(1)))
    if not files:
        raise SystemExit(f"no state_t*.npz in {d.resolve()}")

    n = len(files)
    idx = sorted({max(0, int(round(k * n / n_snap)) - 1)
                  for k in range(1, n_snap + 1)})
    picked = [files[i] for i in idx]
    print(f"[evo] {n} states found, using {len(picked)}: "
          f"{[p.name for p in picked]}")

    snaps = []
    steps = []
    for p in picked:
        step = int(re.search(r"t(\d+)", p.name).group(1))
        z = np.load(p)
        snaps.append((step, float(z["t"]), z["u1"], z["u2"]))
        steps.append(step)
    return snaps, steps, snaps[0][2].shape


def _grid(NX, NY, L, a, avg_width, margin=0.999):
    x_line = np.linspace(0.0, L, NX)
    e_line = np.linspace(-margin, margin, NY) * 0.5
    XX, EE = np.meshgrid(x_line, e_line, indexing="ij")
    return XX, EE * (avg_width + 2*a*np.sin(2*np.pi*XX/L))


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")

    ap = argparse.ArgumentParser(description="u_x/u_y evolution vs unsteady FEM")
    ap.add_argument("--dir", default="outputs_unsteady_stokes_flow")
    ap.add_argument("--n-snap", type=int, default=5)
    ap.add_argument("--cmap", default="RdBu_r")
    ap.add_argument("--tag", default="")
    ap.add_argument("--L", type=float, default=2.5)
    ap.add_argument("--a", type=float, default=0.2)
    ap.add_argument("--avg-width", type=float, default=1.0)
    ap.add_argument("--test-margin", type=float, default=0.999)
    ap.add_argument("--fem", action="store_true",
                    help="march an UNSTEADY FEM reference with the same dt "
                         "and compare at the SAME steps (needs fem_stokes.py "
                         "and scikit-fem)")
    ap.add_argument("--fem-refit", action="store_true")
    ap.add_argument("--fem-nx", type=int, default=200)
    ap.add_argument("--fem-ny", type=int, default=80)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--dP", type=float, default=-30.0)
    ap.add_argument("--dt", type=float, required=False,
                    help="MUST match the dt the PIGP march used, or the FEM "
                         "steps won't line up with the same physical times")
    A = ap.parse_args()

    snaps, steps, (NX, NY) = snapshots_from_states(A.dir, A.n_snap)
    XX, YY = _grid(NX, NY, A.L, A.a, A.avg_width, A.test_margin)

    fem_results = None
    if A.fem:
        if A.dt is None:
            raise SystemExit("--fem requires --dt (must match the PIGP march's dt)")
        import fem_stokes
        margin = min(A.test_margin, 0.98)      # FEM mesh stops at the wall
        XXf, YYf = _grid(NX, NY, A.L, A.a, A.avg_width, margin)
        pts = np.stack([XXf.ravel(), YYf.ravel()], axis=-1)
        fem_results = fem_stokes.get_unsteady(
            pts, steps, Path(A.dir) / "fem_unsteady_reference.npz",
            refit=A.fem_refit, L=A.L, a=A.a, avg_width=A.avg_width,
            eta=A.eta, rho=A.rho, body_force_x=-A.dP/A.L, dt=A.dt,
            nx=A.fem_nx, ny=A.fem_ny)

    plot_evolution_vs_fem(snaps, Path(A.dir), XX=XX, YY=YY, L=A.L, a=A.a,
                          avg_width=A.avg_width, fem=fem_results, cmap=A.cmap,
                          tag=A.tag, dt=A.dt)