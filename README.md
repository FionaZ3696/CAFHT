# CAFHT (Conformalized Adaptive Forecasting of Heterogeneous Trajectories)
This repository implements *CAFHT*: a new conformal method for generating {\em simultaneous} forecasting bands guaranteed to cover the {\em entire path} of a new random trajectory with sufficiently high probability. Prompted by the need for dependable uncertainty estimates in motion planning applications where the behavior of diverse objects may be more or less unpredictable, we blend different techniques from online conformal prediction of single and multiple time series, as well as ideas for addressing heteroscedasticity in regression. This solution is both principled, providing precise finite-sample guarantees, and effective, often leading to more informative predictions than prior methods.

Accompanying paper: *Conformalized Adaptive Forecasting of Heterogeneous Trajectories*.


## Contents

 - `ConformalizedTS` Python package implementing our methods and some alternative benchmarks.
    - `ConformalizedTS/blackbox.py` Code to train and evaluate the blackbox model
    - `ConformalizedTS/evals.py` Code to evaluate the performance of conformal prediction bands
    - `ConformalizedTS/methods.py` Code of the conformal prediction methods for time series. 
    - `ConformalizedTS/networks.py` Example deep networks
 - `third_party/` Third-party Python packages imported by our package.
 - `experiments/` Codes to replicate the figures and tables for the experiments with real data discussed in the accompanying paper.
    - `experiments/ts_sim.py` Code to reproduce the numerical results on the synthetic data.
    - `experiments/ts_realdata.py` Code to reproduce the numerical results on the real data. 


    
## Prerequisites

Prerequisites for the CAFHT package:
 - numpy
 - scipy
 - sklearn
 - torch
 - random
 - pathlib
 - tqdm
 - math
 - pandas
 - matplotlib
 - statsmodels

Additional prerequisites to run the numerical experiments:
 - shutil
 - tempfile
 - pickle
 - sys
 - os
