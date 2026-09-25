#!/bin/sh
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g jh250081a
#PJM -j
#PJM -o logs_20260925/%n.%j.out

export PATH="/work/jh250081a/n15005/local/bin:$PATH"
export UV_CACHE_DIR="/work/jh250081a/n15005/.cache/uv"
export UV_PYTHON_INSTALL_DIR="/work/jh250081a/n15005/.local/share/uv/python"
export UV_TOOL_DIR="/work/jh250081a/n15005/.local/share/uv/tools"

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR"

cd /work/jh250081a/n15005/Stokes-Flow-PIGP-Workspace

REFIT_FLAG=""
if [ "${REFIT:-1}" = "1" ]; then
  REFIT_FLAG="--refit-every-step"
fi

GEOMETRY="${SHAPE:-sinusoidal}"

uv run python ./src/workspace/main_unsteady_stokes_flow.py \
  --profiles --evolution --fem --geometry "$GEOMETRY" --n-artificial "${ARTIFICIAL:-270}"\
  --n-candidate 680 --n-loop "${NLOOP:-50}" --dt "${DT:-0.01}" --fresh-points \
  --nm-iter "${NMITER:-50}" $REFIT_FLAG --n-snap "${NSNAP:-7}" --tol "${TOL:-1}" --profile-n-times 4