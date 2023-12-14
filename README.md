# Statistical Calibration

This repository contains the implementation for the "statistical calibration" in python.

## File Overview

1. main.py: This file contains the python code used in simulation studies.

* Configuration: We conducted 5 configurations from conf_0 to conf_4 where in paper we use conf_1 to conf_5.

* Standard Deviation: We conducted 4 stds to demonstrate the effectiveness of our algorithms $\sigma^2$ = [0.1, 0.25, 0.5, 1].


2. main_real.py: This file constains the python code for the real data.

- The sample size is 19.

- The reponse variable is normalized current for maintaining a fixed membrane potential of -35mV

- The input variable is the logarithm of time.


3. utils.py: This file constains the data generation code and corresponding functions.

