#!/bin/bash -l

#SBATCH --job-name=ace-evaluator
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --account=gfdlhires
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=stdout/%x.%j.out
#SBATCH --signal=USR1@60
#SBATCH --open-mode=append

set -xe

ENVIRONMENT=2026-02-28-ace-ensemble-evaluator
CONFIG="config.yaml"
SCRATCH="/scratch4/GFDL/gfdlhires/Spencer.Clark/2026-02-28-output"

# Directory for saving output from evaluator job
FME_OUTPUT_DIR=${SCRATCH}/fme-output/${SLURM_JOB_ID}
mkdir -p $FME_OUTPUT_DIR

OVERRIDE="base_evaluator_config.experiment_dir=${FME_OUTPUT_DIR}"
WANDB_MODE=disabled srun -u conda run --no-capture-output --name $ENVIRONMENT \
     torchrun \
     -m fme.ace.batched_evaluator $CONFIG --override $OVERRIDE
