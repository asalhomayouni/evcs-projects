#!/bin/bash
#SBATCH --job-name=evcs_run
#SBATCH --account=def-tms
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --output=logs/evcs_%j.out
#SBATCH --error=logs/evcs_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=asalhmy7@yorku.ca

mkdir -p logs

module purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack
module load gurobi/10.0.3

source ~/EVCS-PROJECTS/venv/bin/activate

export GRB_LICENSE_FILE=/cvmfs/restricted.computecanada.ca/config/licenses/gurobi/clusters/narval.lic

echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Start     : $(date)"

cd ~/EVCS-PROJECTS
python scripts/run_evcs.py \
    --csv  ${CSV:-center_146_Verona_k250.csv} \
    --seed ${SEED:-11} \
    --D    ${D_KM:-2.0} \
    --T    ${T_VAL:-6} \
    --Q    ${Q_VAL:-20}

echo "End : $(date)"
