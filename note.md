## Quickstart

Experiment 1.) Just running the base model, but with correcxt uncertainty propagation.
Objective: To study the error, uncertainty, steady state

Base
pjsub -x REFIT=1,DT=0.01,NLOOP=50,NSNAP=7,TOL=1 run_patch.sh

uv run python ./src/workspace/unsteady_stokes_flow.py \
  --profiles --evolution --fem \
  --n-candidate 500 --n-loop 50 --dt 0.01 --fresh-points \
  --nm-iter 50 --refit-every-step --n-snap 7 --tol 1 --profile-n-times 5

Experiment 2.) Size of time dt
Objective: To see if any dt size would affect the error, uncertainty.

- small dt
pjsub -x REFIT=1,DT=0.005,NLOOP=100,NSNAP=7,TOL=1 run_patch.sh
- big dt
pjsub -x REFIT=1,DT=0.02,NLOOP=25,NSNAP=7,TOL=1 run_patch.sh

Experiment 3.) Reoptimizing every step matter?
Objective: to study how reoptimizing every step affect the results tremendously or not. We run nelder mead at 1000 loop to figure out.

pjsub -x REFIT=0,DT=0.01,NLOOP=50,NSNAP=7,TOL=1,NMITER=1000 run_patch.sh

Experiment 4.) Plates poiseuille flow.

pjsub -x REFIT=1,DT=0.01,NLOOP=50,NSNAP=7,TOL=1,SHAPE="plates" run_patch.s\