"""Stitch the per-step PNGs written by the march into one animated GIF.

    python make_gif.py <plots_dir> [--prefix combined|u_x|u_y|p] [--fps 10]
                       [--width 900] [--max-step N] [--fp FINGERPRINT]
"""
import argparse
import re
from pathlib import Path

from PIL import Image


def build_gif(plot_dir, out_path, *, prefix="combined", fps=10, width=900,
              max_step=None, fp=None):
    plot_dir = Path(plot_dir)
    pat = re.compile(rf"^{re.escape(prefix)}_(.+)_t(\d+)\.png$")
    items = []
    for p in plot_dir.glob(f"{prefix}_*_t*.png"):
        m = pat.match(p.name)
        if m is None or (fp is not None and m.group(1) != fp):
            continue
        step = int(m.group(2))
        if max_step is not None and step > max_step:
            continue
        items.append((step, p))
    items.sort()
    if not items:
        print(f"[gif] no '{prefix}_*_t*.png' files found in {plot_dir}")
        return None

    resample = getattr(Image, "Resampling", Image).LANCZOS
    frames = []
    for step, p in items:
        im = Image.open(p).convert("RGB")
        if width and im.width > width:          # full-size frames make a huge GIF
            im = im.resize((width, round(im.height * width / im.width)), resample)
        frames.append(im.quantize(colors=256))  # per-frame palette, keeps memory low

    dur = int(1000 / fps)
    durations = [dur] * (len(frames) - 1) + [1500]      # hold the last frame
    out_path = Path(out_path)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=2)
    print(f"[gif] {len(frames)} frames (steps {items[0][0]}..{items[-1][0]}) "
          f"-> {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("plot_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default="combined")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--max-step", type=int, default=None)
    ap.add_argument("--fp", default=None,
                    help="only use frames from this fingerprint (if the folder "
                         "mixes runs)")
    A = ap.parse_args()
    out = A.out or (Path(A.plot_dir).parent / f"evolution_{A.prefix}.gif")
    build_gif(A.plot_dir, out, prefix=A.prefix, fps=A.fps, width=A.width,
              max_step=A.max_step, fp=A.fp)