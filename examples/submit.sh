#!/bin/bash

#SBATCH -o optimizer.out-%A-%a
#SBATCH -c 4
#SBATCH -a 1-10

source /etc/profile

~/julia-1.7.3/bin/julia -t 4 run_experiment.jl $1 $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID

