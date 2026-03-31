#!/bin/bash
#SBATCH --job-name=evcs_run
#SBATCH --account=def-tms
<<<<<<< HEAD
#SBATCH --time=03:00:00
=======
#SBATCH --time=03:30:00
>>>>>>> eb6485441a2810d973361936d1effee409ecb622
#SBATCH --cpus-per-task=4
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
python scripts/run_evcs.py

echo "End : $(date)"
