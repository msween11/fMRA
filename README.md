# fMRA
Functional Multi-Reference Alignment via Deconvolution


findz and center_BS are just technical files.

FUNC_singlerun and SCRIPT_singlerun are functional and script versions of the same file. They do the same things if the parameters are set the same. They perform a single run of the algorithm in the paper on a signal f1,...,f4 with either an aperiodic or uniform shift distribution.  

N_vs_error.m generates two tables that contain the data for error decay against sample size. 

sigma_vs_error.m does likewise for error decay against sigma.  
lambda_vs_error does likewise for error decay against lambda.

make_figures uses the data from the three above scripts and generates the plots in the figure.
