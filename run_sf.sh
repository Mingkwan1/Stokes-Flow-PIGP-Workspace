#!/bin/sh
#PJM -L rscgrp=share
#PJM -L gpu=2
#PJM -L elapse=24:00:00
#PJM -g jh250081a
#PJM -j
#PJM -o logs_20260927/%n.%j.out

export PATH="/work/jh250081a/n15005/local/bin:$PATH"
export UV_CACHE_DIR="/work/jh250081a/n15005/.cache/uv"
export UV_PYTHON_INSTALL_DIR="/work/jh250081a/n15005/.local/share/uv/python"
export UV_TOOL_DIR="/work/jh250081a/n15005/.local/share/uv/tools"

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR"

cd /work/jh250081a/n15005/Stokes-Flow-PIGP-Workspace

uv run python ./src/workspace/2D_Stokes_PIGP.py \
  --fem 