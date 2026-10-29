## Quickstart

Base
pjsub -x REFIT=1,DT=0.01,NLOOP=50,NSNAP=7,TOL=1 run.sh
uv run python ./src/workspace/unsteady_stokes_flow.py \
  --profiles --evolution --fem \
  --n-candidate 500 --n-loop 50 --dt 0.01 --fresh-points \
  --nm-iter 50 --refit-every-step --n-snap 7 --tol 1 --profile-n-times 5

small dt
pjsub -x REFIT=1,DT=0.005,NLOOP=100,NSNAP=7,TOL=1 run.sh
uv run python ./src/workspace/unsteady_stokes_flow.py \
  --profiles --evolution --fem \
  --n-candidate 500 --n-loop 50 --dt 0.01 --fresh-points \
  --nm-iter 50 --refit-every-step --n-snap 7 --tol 1 --profile-n-times 5

no refit every steo
pjsub -x REFIT=0,DT=0.01,NLOOP=50,NSNAP=7,TOL=1,NMITER=1000 run.sh

uv run python ./src/workspace/unsteady_stokes_flow.py \
  --profiles --evolution --fem \
  --n-candidate 500 --n-loop 50 --dt 0.01 --fresh-points \
  --nm-iter 50 --refit-every-step --n-snap 7 --tol 1 --profile-n-times 5
Base

uv run python ./src/workspace/unsteady_stokes_flow.py \
  --profiles --evolution --fem \
  --n-candidate 500 --n-loop 50 --dt 0.01 --fresh-points \
  --nm-iter 50 --refit-every-step --n-snap 7 --tol 1 --profile-n-times 5

# Running Code on Wisteria/BDEC-01

Quick reference for connecting to Wisteria and running the PIGP workspace code via batch jobs.

---

## 1. Connecting

### SSH config (`~/.ssh/config` on your local PC)

```
Host wisteria
    HostName wisteria.cc.u-tokyo.ac.jp
    User <your_username>
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

Connect with:
```bash
ssh wisteria
```

---

## 2. Checking system info

```bash
hostname                 # which node you're on (login node has no GPU)
lscpu                    # CPU info
nproc                    # core count
free -h                  # memory
nvidia-smi                # GPU info (only works on Aquarius compute nodes)
df -h                     # disk usage
```

Job-related:
```bash
pjstat --rsc              # available resource groups
pjstat --limit             # your project's job/GPU limits
pjstat                     # your current jobs
pjstat -H                  # recently finished jobs (last 3 days)
```

---

## 3. Resource groups (Aquarius, GPU)

| Group        | Max elapse | Nodes/job |
|--------------|-----------|-----------|
| `debug-a`     | 30 min    | 1 |
| `interactive-a` | 10 min  | 1 |
| `short-a`     | 2 hours   | 1–2 |
| `small-a`     | 48 hours  | 2–3 |
| `medium-a`    | 48 hours  | 3–4 |
| `large-a`     | 24 hours  | 5–8 |

Each node-occupancy Aquarius job uses 8 GPUs (1 node = 8 GPUs).

---

## 4. Directory rules — IMPORTANT

**Compute nodes do NOT mount `/home`.** Only `/work` is available inside a batch job.

- Keep your project, venv, and any tool installs (`uv`, caches) under `/work/jh250081a/n15005/...`
- Anything referencing `$HOME` (`~`) inside a job script will fail with `Permission denied` or `No such file or directory`.

Your project path:
```
/work/jh250081a/n15005/Stokes-Flow-PIGP-Workspace
```

---

## 5. Cloning the repo

```bash
cd /work/jh250081a/n15005
git clone git@github.com:Mingkwan1/Stokes-Flow-PIGP-Workspace.git
cd Stokes-Flow-PIGP-Workspace
git checkout develop        # or: git clone -b develop <url>
```

GitHub SSH auth on Wisteria needs its **own** key registered separately from your PC's key:
```bash
ssh-keygen -t ed25519 -C "wisteria-n15005"
cat ~/.ssh/id_ed25519.pub    # add this to GitHub → Settings → SSH keys
ssh -T git@github.com        # verify
```

---

## 6. Setting up `uv`

Install once (login node):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Copy the binary into `/work` so compute nodes can find it:
```bash
mkdir -p /work/jh250081a/n15005/local/bin
cp ~/.local/bin/uv /work/jh250081a/n15005/local/bin/
```

---

## 7. Batch job script (`run.sh`)

```bash
#!/bin/sh
#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -g jh250081a
#PJM -j

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

uv run python ./src/workspace/unsteady_stokes_flow.py \
  --profiles --evolution --fem \
  --n-candidate 500 --n-loop "${NLOOP:-50}" --dt "${DT:-0.01}" --fresh-points \
  --nm-iter 50 $REFIT_FLAG --n-snap "${NSNAP:-7}" --tol "${TOL:-1}" \
  --profile-n-times 5
```

> Note the `$REFIT_FLAG` is intentionally **unquoted** so an empty value doesn't pass as a blank argument.

---

## 8. Submitting jobs

**Single run (defaults):**
```bash
pjsub run.sh
```

**Parameter sweep (env vars via `-x`):**
```bash
pjsub -x NSNAP=5,TOL=1 run.sh
pjsub -x NSNAP=7,TOL=0.5 run.sh
pjsub -x REFIT=0,DT=0.005,NLOOP=100 run.sh
```

Rules for `-x`:
- One comma-separated list, **no spaces**, **no trailing comma**.
- `pjsub -x A=1,B=2 run.sh` ✅
- `pjsub -x A=1, B=2 run.sh` ❌ (space breaks parsing)

**Loop over many values:**
```bash
for tol in 0.5 1 2 5; do
  pjsub --name "tol${tol}" -x TOL=$tol run.sh
done
```

---

## 9. Checking jobs and output

```bash
pjstat              # running/queued jobs
pjstat -H            # finished jobs
pjdel <JOB_ID>        # cancel a job
```

Output file (stdout+stderr merged via `#PJM -j`):
```
run.sh.<JOB_ID>.out
```

```bash
tail -f run.sh.<JOB_ID>.out    # live view while running
```

---

## 10. Common errors & fixes

| Error | Cause | Fix |
|---|---|---|
| `uv: command not found` | `uv` not on `$PATH` in job | `export PATH="/work/.../local/bin:$PATH"` in `run.sh` |
| `Failed to initialize cache at /pjmhome/.cache/uv` | `$HOME` not mounted on compute node | Set `UV_CACHE_DIR` to a `/work` path |
| `Permission denied ... /pjmhome/.local/share/uv/python` | Same — uv's Python install dir defaults to `$HOME` | Set `UV_PYTHON_INSTALL_DIR` to `/work` |
| `Elapse limit is out of range` | Requested time exceeds group's max | Use a group with a longer limit (see table above), or reduce elapse |
| `git@github.com: Permission denied (publickey)` | No SSH key registered with GitHub *from Wisteria* | Generate a Wisteria-specific key or use agent forwarding (`ssh -A`) |

**Shortcut**: instead of setting each `UV_*` var individually, you can redirect `$HOME` itself for the job:
```bash
export HOME="/work/jh250081a/n15005"
```
This fixes any future `$HOME`-dependent tool at once (pip, matplotlib config, etc.).