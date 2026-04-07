#!/bin/bash
#SBATCH --job-name=evcs_array
#SBATCH --account=def-tms
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --array=1-12
#SBATCH --time=06:00:00
#SBATCH --array=1-10
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=asalhmy7@yorku.ca

set -euo pipefail
trap 'echo "[$(date)] Job ${SLURM_ARRAY_TASK_ID} failed with exit code $?"; exit 1' ERR

mkdir -p logs

module purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack
module load gurobi/10.0.3

source ~/EVCS-PROJECTS/venv/bin/activate

export GRB_LICENSE_FILE=/cvmfs/restricted.computecanada.ca/config/licenses/gurobi/clusters/narval.lic

cd ~/EVCS-PROJECTS

LINE=$(awk -v N=$SLURM_ARRAY_TASK_ID 'NR==N' configs.tsv)

if [ -z "$LINE" ]; then
    echo "No config found for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

IFS=$'\t' read CSV SEED D T Q <<< "$LINE"

echo "==============================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Task    : $SLURM_ARRAY_TASK_ID"
echo "CSV     : $CSV"
echo "Seed    : $SEED"
echo "D       : $D"
echo "T       : $T"
echo "Q       : $Q"
echo "Start   : $(date)"
echo "==============================="

python scripts/run_evcs.py \
    --csv  $CSV  \
    --seed $SEED \
    --D    $D    \
    --T    $T    \
    --Q    $Q

echo "End : $(date)"