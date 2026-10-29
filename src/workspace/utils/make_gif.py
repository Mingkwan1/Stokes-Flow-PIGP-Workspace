"""Animated GIF of the PIGP march: three colour maps (u_x, u_y, p) per time step.

Every frame uses ONE fixed colour scale for the whole run, so colour changes
in the GIF are real changes in the flow, not a rescaling from frame to frame:

    u_x : -ux_max ... +ux_max   (ux_max = max |u_x| over all frames)
    u_y : -uy_max ... +uy_max   (uy_max = max |u_y| over all frames)
    p   : global min ... global max over all frames

No training points, no profile/uncertainty panels. Frames are read from the
state_t*.npz files the march writes at every step, so a GIF can also be rebuilt
for an old run without re-running it.

From the main script:
    from utils import make_gif
    make_gif.build_state_gif(OUTDIR, OUTDIR / "evolution.gif", XX=XX, YY=YY,
                             L=L, a=a, avg_width=avg_width)

As a script (run from src/workspace):
    python -m utils.make_gif <run_dir> [--out run.gif] [--fps 10] [--dpi 100]
        [--L 2.5 --a 0.2 --avg-width 1.0 --test-margin 0.98]
        [--max-step N] [--vmax-ux V] [--vmax-uy V] [--frames-dir DIR]
"""
import argparse
import io
import re
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image


def load_states(state_dir, max_step=None):
    """[(step, t, u1, u2, p), ...] sorted by step, from state_t*.npz files."""
    found = []
    for p in Path(state_dir).glob("state_t*.npz"):
        m = re.fullmatch(r"state_t(\d+)\.npz", p.name)
        if m and (max_step is None or int(m.group(1)) <= max_step):
            found.append((int(m.group(1)), p))
    found.sort()
    frames = []
    for step, p in found:
        z = np.load(p)
        frames.append((step, float(z["t"]), z["u1"], z["u2"], z["p"]))
    return frames


def _grid(NX, NY, L, a, avg_width, margin):
    """Same test grid as the march: x uniform, y = eta * local width."""
    x = np.linspace(0.0, L, NX)
    e = np.linspace(-margin, margin, NY) * 0.5
    XX, EE = np.meshgrid(x, e, indexing="ij")
    return XX, EE * (avg_width + 2.0 * a * np.sin(2.0 * np.pi * XX / L))


def _fixed_palette(cmap, p_cmap):
    """ONE 256-colour palette shared by every frame: 120 shades of the u map,
    90 of the p map, 46 greys (black/white/text). A per-frame adaptive palette
    would starve the colour bars on the nearly-white early frames and make
    colours jump between frames."""
    from matplotlib import colormaps
    cols = []
    for name, n in ((cmap, 120), (p_cmap, 90)):
        cm = colormaps[name]
        cols += [tuple(int(round(255 * c)) for c in cm(i / (n - 1))[:3])
                 for i in range(n)]
    cols += [(int(g), int(g), int(g)) for g in np.linspace(0, 255, 46).round()]
    pimg = Image.new("P", (1, 1))
    pimg.putpalette([v for c in cols for v in c])
    return pimg


def _render(frame, XX, YY, xw, wall, scales, cmap, p_cmap, dpi):
    step, t, U1, U2, P = frame
    vx, vy, p_lo, p_hi = scales
    fig = Figure(figsize=(7.5, 9.5), constrained_layout=True)
    axes = fig.subplots(3, 1, sharex=True)
    specs = [(U1, r"$u_x$", -vx, vx, cmap),
             (U2, r"$u_y$", -vy, vy, cmap),
             (P,  r"$p$",   p_lo, p_hi, p_cmap)]
    for ax, (F, label, lo, hi, cm) in zip(axes, specs):
        pc = ax.pcolormesh(XX, YY, np.asarray(F), cmap=cm, vmin=lo, vmax=hi,
                           shading="gouraud")
        ax.plot(xw, wall / 2, "k", lw=1.5)
        ax.plot(xw, -wall / 2, "k", lw=1.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylabel("$y$", fontsize=14)
        ax.tick_params(labelsize=11)
        cb = fig.colorbar(pc, ax=ax, pad=0.02)
        cb.set_label(label, fontsize=14)
        cb.ax.tick_params(labelsize=11)
    axes[-1].set_xlabel("$x$", fontsize=14)
    fig.suptitle(rf"$t = {t:.3f}$", fontsize=16)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_state_gif(state_dir, out_path, *, L, a, avg_width, XX=None, YY=None,
                    test_margin=0.98, fps=10, dpi=100, max_step=None,
                    vmax_ux=None, vmax_uy=None, frames_dir=None,
                    cmap="RdBu_r", p_cmap="viridis"):
    frames = load_states(state_dir, max_step)
    if not frames:
        print(f"[gif] no state_t*.npz files found in {state_dir}")
        return None

    NX, NY = frames[0][2].shape
    if XX is None or YY is None:
        XX, YY = _grid(NX, NY, L, a, avg_width, test_margin)
    XX, YY = np.asarray(XX), np.asarray(YY)
    xw = np.linspace(0.0, L, 400)
    wall = avg_width + 2.0 * a * np.sin(2.0 * np.pi * xw / L)

    # ---- ONE colour scale for the whole run ------------------------------
    vx = vmax_ux or max(float(np.nanmax(np.abs(f[2]))) for f in frames)
    vy = vmax_uy or max(float(np.nanmax(np.abs(f[3]))) for f in frames)
    p_lo = min(float(np.nanmin(f[4])) for f in frames)
    p_hi = max(float(np.nanmax(f[4])) for f in frames)
    vx, vy = max(vx, 1e-12), max(vy, 1e-12)
    if p_hi - p_lo < 1e-12:
        p_lo, p_hi = p_lo - 1e-6, p_hi + 1e-6
    scales = (vx, vy, p_lo, p_hi)
    print(f"[gif] fixed colour scale: u_x +/-{vx:.4f}  u_y +/-{vy:.4f}  "
          f"p [{p_lo:.3f}, {p_hi:.3f}]  ({len(frames)} frames)")

    if frames_dir is not None:
        frames_dir = Path(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    no_dither = getattr(Image, "Dither", Image).NONE   # dithering would flicker
    shared_palette = _fixed_palette(cmap, p_cmap)
    pal = []
    for fr in frames:
        im = _render(fr, XX, YY, xw, wall, scales, cmap, p_cmap, dpi)
        if frames_dir is not None:
            im.save(frames_dir / f"frame_t{fr[0]:04d}.png")
        pal.append(im.quantize(palette=shared_palette, dither=no_dither))

    dur = int(1000 / fps)
    durations = [dur] * (len(pal) - 1) + [1500]            # hold the last frame
    out_path = Path(out_path)
    pal[0].save(out_path, save_all=True, append_images=pal[1:],
                duration=durations, loop=0, disposal=2)
    print(f"[gif] {len(pal)} frames (steps {frames[0][0]}..{frames[-1][0]}) "
          f"-> {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    return out_path


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="folder containing state_t*.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--max-step", type=int, default=None)
    ap.add_argument("--L", type=float, default=2.5)
    ap.add_argument("--a", type=float, default=0.2, help="0 for flat plates")
    ap.add_argument("--avg-width", type=float, default=1.0)
    ap.add_argument("--test-margin", type=float, default=0.98,
                    help="0.98 when the run used --fem, else 0.999")
    ap.add_argument("--vmax-ux", type=float, default=None,
                    help="override the automatic +/-ux_max")
    ap.add_argument("--vmax-uy", type=float, default=None,
                    help="override the automatic +/-uy_max")
    ap.add_argument("--frames-dir", default=None,
                    help="also save every frame as a PNG here")
    A = ap.parse_args()
    out = A.out or (Path(A.run_dir) / "evolution.gif")
    build_state_gif(A.run_dir, out, L=A.L, a=A.a, avg_width=A.avg_width,
                    test_margin=A.test_margin, fps=A.fps, dpi=A.dpi,
                    max_step=A.max_step, vmax_ux=A.vmax_ux, vmax_uy=A.vmax_uy,
                    frames_dir=A.frames_dir)