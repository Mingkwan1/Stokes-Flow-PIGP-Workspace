"""Run report (.txt + .csv) for the unsteady Stokes PIGP march.

Everything that used to be read from the main script's globals is now passed
in explicitly (n_active, n_theta, eps_jitter, propagate); dt is taken from
args.dt.
"""
import csv

import numpy as onp


def write_report(history, fp, args, outdir, *, n_active, n_theta, eps_jitter, propagate):
    """Single human-readable .txt + machine-readable .csv summarizing the
    whole march: per-step errors/uncertainty, calibration, error statistics."""
    txt_path = outdir / f"report_{fp}.txt"
    csv_path = outdir / f"report_{fp}.csv"

    # --- CSV: one row per step, union of all keys that ever appeared -------
    all_keys = []
    seen = set()
    for h in history:
        for k in h:
            if k not in seen:
                seen.add(k); all_keys.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, restval="", extrasaction="ignore")
        w.writeheader()
        for h in history:
            w.writerow(h)
    print(f"[report] wrote {csv_path}")

    # --- TXT: readable report ------------------------------------------------
    lines = []
    lines.append("=" * 78)
    lines.append("PIGP UNSTEADY STOKES FLOW -- RUN REPORT")
    lines.append("=" * 78)
    lines.append(f"fingerprint       : {fp}")
    lines.append(f"geometry          : {args.geometry}")
    lines.append(f"dt / n_loop       : {args.dt} / {args.n_loop}  (t_final = {args.dt*args.n_loop:.4f})")
    lines.append(f"n_artificial      : {args.n_artificial}  fresh_points={args.fresh_points}  fixed_points={args.fixed_points}")
    lines.append(f"nm_iter / tol     : {args.nm_iter} / {args.tol}")
    lines.append(f"lbfgs_iter / tol  : {args.lbfgs_iter} / {args.lbfgs_tol}")
    lines.append(f"refit_every_step  : {args.refit_every_step}")
    lines.append(f"zero_up / active   : {args.zero_up} / {n_active} of {n_theta}")
    lines.append(f"K (eq.12)          : G noise = diag(Sigma_prev), boundary noise 0, "
                 f"eps_jitter = {eps_jitter:g}, propagation term = {propagate}")
    lines.append(f"fem comparison    : {args.fem}")
    lines.append(f"plots stop at t   : {args.t_stop:g}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("PER-STEP DIAGNOSTICS")
    lines.append("-" * 78)
    has_fem = any("coverage_95_ux" in h for h in history)
    if has_fem:
        header = (f"{'step':>5}{'t':>9}{'ux_max':>11}{'|du|':>11}"
                   f"{'2sig_ux':>11}{'cov_ux':>9}{'mae_ux':>11}"
                   f"{'2sig_uy':>11}{'cov_uy':>9}{'mae_uy':>11}{'fem_L2':>9}")
    else:
        header = (f"{'step':>5}{'t':>9}{'ux_max':>11}{'|du|':>11}"
                   f"{'2sig_ux':>11}{'2sig_uy':>11}")
    lines.append(header)
    for h in history:
        row = f"{h['step']:>5}{h['t']:>9.4f}{h['ux_max']:>11.5f}{h['dmax']:>11.2e}"
        row += f"{h['twosig_max_ux']:>11.2e}"
        if has_fem:
            cov_x = h.get("coverage_95_ux")
            mae_x = h.get("mean_abs_err_ux")
            row += f"{(100*cov_x if cov_x is not None else float('nan')):>8.1f}%"
            row += f"{(mae_x if mae_x is not None else float('nan')):>11.2e}"
        row += f"{h['twosig_max_uy']:>11.2e}"
        if has_fem:
            cov_y = h.get("coverage_95_uy")
            mae_y = h.get("mean_abs_err_uy")
            row += f"{(100*cov_y if cov_y is not None else float('nan')):>8.1f}%"
            row += f"{(mae_y if mae_y is not None else float('nan')):>11.2e}"
            fl2 = h.get("fem_rel_l2")
            row += f"{(fl2 if fl2 is not None else float('nan')):>9.4f}" if fl2 is not None else f"{'':>9}"
        lines.append(row)
    lines.append("")

    if has_fem:
        lines.append("-" * 78)
        lines.append("CALIBRATION SUMMARY (final step)")
        lines.append("-" * 78)
        h_last_fem = [h for h in history if "coverage_95_ux" in h]
        if h_last_fem:
            hl = h_last_fem[-1]
            lines.append(f"  t = {hl['t']:.4f}")
            lines.append(f"  u_x : coverage_95 = {100*hl['coverage_95_ux']:.1f}%   "
                          f"mean|err| = {hl['mean_abs_err_ux']:.3e}   "
                          f"max 2sigma = {hl['twosig_max_ux']:.3e}")
            lines.append(f"  u_y : coverage_95 = {100*hl['coverage_95_uy']:.1f}%   "
                          f"mean|err| = {hl['mean_abs_err_uy']:.3e}   "
                          f"max 2sigma = {hl['twosig_max_uy']:.3e}")
        lines.append("")

    # --- min / max / mean of absolute and relative error vs unsteady FEM ----
    if has_fem and any("abs_mean_ux" in h for h in history):
        lines.append("-" * 78)
        lines.append("POINTWISE ERROR vs UNSTEADY FEM (PIGP mean - FEM)")
        lines.append("-" * 78)
        lines.append("  abs = |PIGP - FEM|")
        lines.append(f"  rel = |PIGP - FEM| / |FEM|, only where |FEM| >= "
                     f"{args.rel_floor:g} * max|FEM|")
        for comp, sym in (("ux", "u_x"), ("uy", "u_y")):
            hh = [h for h in history if f"abs_mean_{comp}" in h]
            if not hh:
                continue
            lines.append("")
            lines.append(f"  {sym}")
            lines.append(f"  {'step':>5}{'t':>9}{'abs_min':>11}{'abs_max':>11}"
                         f"{'abs_mean':>11}{'rel_min':>11}{'rel_max':>11}{'rel_mean':>11}")
            for h in hh:
                lines.append(
                    f"  {h['step']:>5}{h['t']:>9.4f}"
                    f"{h[f'abs_min_{comp}']:>11.2e}{h[f'abs_max_{comp}']:>11.2e}"
                    f"{h[f'abs_mean_{comp}']:>11.2e}{h[f'rel_min_{comp}']:>11.2e}"
                    f"{h[f'rel_max_{comp}']:>11.2e}{h[f'rel_mean_{comp}']:>11.2e}")
            lines.append("")
            for label, sub in (("all steps", hh),
                               (f"t <= {args.t_stop:g}",
                                [h for h in hh if h["t"] <= args.t_stop + 1e-9])):
                if not sub:
                    continue
                g = lambda k, fn: fn([h[f"{k}_{comp}"] for h in sub])
                lines.append(f"  {label:<10} abs: min {g('abs_min', min):.3e}   "
                             f"max {g('abs_max', max):.3e}   mean {g('abs_mean', onp.mean):.3e}")
                lines.append(f"  {'':<10} rel: min {g('rel_min', min):.3e}   "
                             f"max {g('rel_max', max):.3e}   mean {g('rel_mean', onp.mean):.3e}")
        lines.append("")

    lines.append("=" * 78)

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {txt_path}")