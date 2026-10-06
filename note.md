## Quickstart

Experiment 1.) Just running the base model, but with correcxt uncertainty propagation.
Objective: To study the error, uncertainty, steady state

Base
pjsub -x REFIT=1,DT=0.01,NLOOP=50,NSNAP=7,TOL=0.02 run_main.sh

uv run python ./src/workspace/unsteady_stokes_flow.py \
  --profiles --evolution --fem \
  --n-candidate 500 --n-loop 50 --dt 0.01 --fresh-points \
  --nm-iter 50 --refit-every-step --n-snap 7 --tol 1 --profile-n-times 5

Experiment 2.) Size of time dt
Objective: To see if any dt size would affect the error, uncertainty.

- small dt
pjsub -x REFIT=1,DT=0.005,NLOOP=100,NSNAP=7,TOL=0.02 run_main.sh
- big dt
pjsub -x REFIT=1,DT=0.02,NLOOP=25,NSNAP=7,TOL=0.02 run_main.sh

Experiment 3.) Reoptimizing every step matter?
Objective: to study how reoptimizing every step affect the results tremendously or not. We run nelder mead at 1000 loop to figure out.

pjsub -x REFIT=0,DT=0.01,NLOOP=50,NSNAP=7,TOL=0.02,NMITER=1000 run_main.sh

Experiment 4.) Plates poiseuille flow.

pjsub -x REFIT=1,DT=0.01,NLOOP=50,NSNAP=7,TOL=0.02,SHAPE="plates" run_main.sh


All in the new folder
Experiment 5) fixed point
i want to know whether removing refresh every point will make the jitter less extreme and sommith curve or not

pjsub -x REFIT=1,DT=0.01,NLOOP=50,NSNAP=7,TOL=0.02,FRESH_POINTS=0 run_main.sh

Expoeriment 6. ) increasing the nm iters

pjsub -x REFIT=1,DT=0.01,NLOOP=200,NSNAP=7,TOL=0.02 run_main.sh
pjsub -x REFIT=1,DT=0.01,NLOOP=50,NSNAP=7,TOL=0.02,FRESH_POINTS=0 run_main.sh

# Experiment 2026/09/25
- Add a new steady time system: check whether it is working properly
- Trying to lower the uncertainty jitter. Doing this by adjusting the nelder mead iter, adjusting the artificial points.
- Parameter is dt

As the uncertainty of run after a certain time tends to jitter in a certain range, we found that such results display a sign of uncertainty incoherence in time propagation prediction.
We hypothesized that a simple adjustment of artificial training points, along with an increased in NM iteration would help improve the jitter of the uncertainty value.


## 1) Base run

```
pjsub -x REFIT=1,DT=0.01,NLOOP=50,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
```

## 2) Nelder Mead nm adjutment

```
pjsub -x REFIT=1,DT=0.01,NLOOP=50,NMITER=200,NSNAP=5,TOL=0.01 run.sh
```

```
pjsub -x REFIT=1,DT=0.01,NLOOP=50,NMITER=500,NSNAP=5,TOL=0.01 run.sh
```

```
pjsub -x REFIT=1,DT=0.01,NLOOP=50,NMITER=300,NSNAP=5,TOL=0.01 run.sh
```

## 3) Artificial points changes

```
pjsub -x REFIT=1,DT=0.01,NLOOP=50,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=380 run.sh
```

```
pjsub -x REFIT=1,DT=0.01,NLOOP=50,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=270 run.sh
```

## 4) Effect of Dt size

```
pjsub -x REFIT=1,DT=0.05,NLOOP=20,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
```

```
pjsub -x REFIT=1,DT=0.005,NLOOP=200,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
```

## Plates
pjsub -x REFIT=1,DT=0.01,NLOOP=50,NMITER=100,NSNAP=5,TOL=0.01,GEOMETRY=plates run.sh


# 3036/09/26

- Fixing the jitter so that it is the same as Raissia. Currently, the jitter is constant and applied diagonally to every K. It will now add at the correct postion.
```
pjsub -x REFIT=1,DT=0.01,NLOOP=40,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
```
```
pjsub -x REFIT=0,DT=0.01,NLOOP=50,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
```
```
pjsub -x REFIT=1,DT=0.01,NLOOP=50,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
```
```
pjsub -x REFIT=1,DT=0.01,NLOOP=50,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
```

No fresh points works!

```
pjsub -x REFIT=1,DT=0.01,NLOOP=40,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
```

```
pjsub -x REFIT=1,DT=0.05,NLOOP=100,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
```

```
pjsub -x REFIT=1,DT=0.005,NLOOP=200,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
```


```
pjsub -x REFIT=1,DT=0.01,NLOOP=40,NMITER=50,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
```


```
pjsub -x REFIT=1,DT=0.01,NLOOP=40,NMITER=200,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
```

```
pjsub -x REFIT=1,DT=0.01,NLOOP=40,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=270 run.sh
```

```
pjsub -x REFIT=1,DT=0.01,NLOOP=40,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=410 run.sh
```

### No args
pjsub -x REFIT=1,DT=0.01,NLOOP=40,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
pjsub -x REFIT=1,DT=0.01,NLOOP=100,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
pjsub -x REFIT=1,DT=0.05,NLOOP=20,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
pjsub -x REFIT=1,DT=0.005,NLOOP=80,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh

pjsub -x REFIT=1,DT=0.01,NLOOP=80,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=270 run.sh
pjsub -x REFIT=1,DT=0.01,NLOOP=40,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=410 run.sh

pjsub -x REFIT=1,DT=0.01,NLOOP=40,NMITER=50,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh
pjsub -x REFIT=1,DT=0.01,NLOOP=40,NMITER=200,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh

### freshpoints args

pjsub -x REFIT=1,DT=0.01,NLOOP=40,NMITER=100,NSNAP=5,TOL=0.01,ARTIFICIAL=340 run.sh

### Plates

pjsub -x REFIT=1,DT=0.01,NLOOP=100,NMITER=100,NSNAP=3,TOL=0.01,ARTIFICIAL=340,SHAPE=plates run.sh


uv run python ./src/workspace/replot_evolution.py 
  --outdir ./src/workspace/outputs/concurrent_points_0927/20260927_usf_0.01_40_340artificial_sinusoidal_nm_100_tol_0.01_fresh_points_False_refit_everystep_True_eps_0.001_zeroup_True 
  --dt 0.01 --n-loop 40 --n-snap 3 --n-artificial 340

# 2026/10/05

Trying new optimzation method: NM + L-BFGS
Also, new delta T calculated from CFL. t min = 0.0083

pjsub -x REFIT=1,DT=0.008,NLOOP=100,NMITER=500,NSNAP=4,TOL=0.001,ARTIFICIAL=340,LBFGS-ITER=100,LBFGS-TOL=1e-5 run.sh

pjsub -x REFIT=1,DT=0.008,NLOOP=100,NMITER=500,NSNAP=4,TOL=0.001,ARTIFICIAL=340,LBFGS-ITER=100,LBFGS-TOL=1e-5,SHAPE=plates run.sh

pjsub -x REFIT=0,DT=0.008,NLOOP=100,NMITER=500,NSNAP=4,TOL=0.001,ARTIFICIAL=340,LBFGS-ITER=100,LBFGS-TOL=1e-5 run.sh

pjsub -x REFIT=1,DT=0.008,NLOOP=100,NMITER=100,NSNAP=3,TOL=0.001,ARTIFICIAL=340,LBFGS-ITER=200,LBFGS-TOL=1e-5 run.sh

pjsub -x REFIT=1,DT=0.008,NLOOP=100,NMITER=100,NSNAP=3,TOL=0.001,ARTIFICIAL=340,LBFGS-ITER=100,LBFGS-TOL=1e-5 run.sh

pjsub -x REFIT=1,DT=0.005,NLOOP=160,NMITER=100,NSNAP=3,TOL=0.001,ARTIFICIAL=340,LBFGS-ITER=100,LBFGS-TOL=1e-5 run.sh

### Base 

pjsub -x REFIT=1,DT=0.008,NLOOP=100,NMITER=100,NSNAP=3,TOL=0.001,ARTIFICIAL=340,LBFGS-ITER=100,LBFGS-TOL=1e-5 run.sh

### Poiseuille Flow

pjsub -x REFIT=1,DT=0.008,NLOOP=100,NMITER=100,NSNAP=3,TOL=0.001,ARTIFICIAL=340,LBFGS-ITER=100,LBFGS-TOL=1e-5,SHAPE=plates run.sh

### Dt 

pjsub -x REFIT=1,DT=0.005,NLOOP=160,NMITER=100,NSNAP=3,TOL=0.001,ARTIFICIAL=340,LBFGS-ITER=100,LBFGS-TOL=1e-5 run.sh

pjsub -x REFIT=1,DT=0.002,NLOOP=400,NMITER=100,NSNAP=3,TOL=0.001,ARTIFICIAL=340,LBFGS-ITER=100,LBFGS-TOL=1e-5 run.sh

### L-BFGS iter

pjsub -x REFIT=1,DT=0.008,NLOOP=100,NMITER=100,NSNAP=3,TOL=0.001,ARTIFICIAL=340,LBFGS-ITER=200,LBFGS-TOL=1e-5 run.sh
pjsub -x REFIT=1,DT=0.008,NLOOP=100,NMITER=100,NSNAP=3,TOL=0.001,ARTIFICIAL=340,LBFGS-ITER=400,LBFGS-TOL=1e-5 run.sh

### Artificila points

pjsub -x REFIT=1,DT=0.008,NLOOP=100,NMITER=100,NSNAP=3,TOL=0.001,ARTIFICIAL=410,LBFGS-ITER=100,LBFGS-TOL=1e-5 run.sh
pjsub -x REFIT=1,DT=0.008,NLOOP=100,NMITER=100,NSNAP=3,TOL=0.001,ARTIFICIAL=270,LBFGS-ITER=100,LBFGS-TOL=1e-5 run.sh
