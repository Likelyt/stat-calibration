# Statistical Calibration

This repository contains the implementation for the "statistical calibration" in python.

## File Overview

1. `main_sim.py`: This file contains the python code used in simulation studies.

* Configuration: We conducted 5 configurations from conf_1 to conf_5.

* Standard Deviation: We conducted 4 stds to demonstrate the effectiveness of our algorithms $\sigma^2$ = [0.1, 0.25, 0.5, 1].

2. `main_real.py`: This file constains the python code for the real data.

* The sample size is 19.

* The reponse variable is normalized current for maintaining a fixed membrane potential of -35mV

* The input variable is the logarithm of time.

3. `utils.py`: This file contains the data generation code and corresponding functions.


## Requirement

1. Create the conda env `rkhs_cal_env`

```bash
conda env create -f rkhs_cal_env.yaml
```

2. Activate the conda env

```bash
conda activate rkhs_cal_env
```

## Run the experiment

1. Run the simulation python script.

```bash
nohup python main_sim.py > logs/log_sim.txt &
```

2. Run the real data python script.

```bash
nohup python main_sim.py > logs/log_real.txt &
```

## Result

1. `\logs` folder: find the PMSE result and corresponding standard deviation over 100 replications.

2. `\figs` folder: plot the predicted fitting curve and true curve.
