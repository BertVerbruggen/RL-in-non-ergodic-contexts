# RL-in-non-ergodic-contexts
Reinforcement learning in path-dependent, non-ergodic contexts

# Content:
1. Data analysis files (Analysis_All_Outputs.ipynb) for all referenced training data in data repository (*zenodo*)
2. Data analysis for single training result (Analysis_RawData.ipynb), same data repository found in (*zenodo*)
3. Hyperparameter example model: HyperparameterModel.ipynb
4. Hyperparameter.csv (Example source file for hyperparameter selections on hyperparameter notebook.\

# Implementation
All data is synthetically generated using the original model as presented in each of the data-directories.
Each directory contains:
1. Some naive output results for sanity checks (*.pdf files)
2. The hyperparameters used to train the model (parameters_$id$.csv)
3. The model (MOL_ERL_LowData.py)
4. The log file of the training (logfile_$id$.log)
5. Training results stored as pickled object (DataDump_$id$.p)

# **To reproduce training results:
Rerun the model python file (MOL_ERL_LowData.py) -> Provide proper reference to the csv file for training (line 40)
Run the model from the script (MOL_ERL_PP.sh) -> Enter proper output directory and path to csv file with hyperparameters
Run the python file and use (Analysis_raw_data.ipynb) to create the figures as enclosed in the paper.

# **To reproduce figures from the paper:
Import the data directories from (*zenodo*) and run (Analysis_All_Outputs.ipynb)

Figures are generated either by the above procedure or from the Wolfram Mathematica notebook (GenericBinomial.nb) as enclosed in this repo and in (*zenodo*) data repo.

# Hyperparameter sweep:
Hyperparameter tuning and testing can be done using the HyperparameterModel.ipynb notebook. This is a standalone model to test hyperparameters. The same test can be performed on a larger scale using input parametres from hyperparameter files as shown by the example file (hyperparameters.csv).

# **Feedback
Enjoy playing around with these models. 
The training was performed on the HPC supercomputer from the Flemish government (https://www.vscentrum.be/hpc) and analysis on the local machine.

# Code and Data statement
All code and data are created by the authors.
