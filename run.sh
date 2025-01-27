#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=cpu
#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --qos=debug
###SBATCH --array=0

echo Python executable: $(which python)
echo
echo Job name: $SLURM_JOB_NAME
echo QOS: $SLURM_JOB_QOS
echo Account: $SLURM_JOB_ACCOUNT
echo Submit dir: $SLURM_SUBMIT_DIR
echo
echo Job array ID: $SLURM_ARRAY_JOB_ID
echo Job ID: $SLURM_JOBID
echo Job array task: $SLURM_ARRAY_TASK_ID
echo Job array task count: $SLURM_ARRAY_TASK_COUNT
echo
echo Nodes: $SLURM_NNODES
echo Head node: $SLURMD_NODENAME
echo hostname $(hostname)
echo Nodelist: $SLURM_NODELIST
echo Tasks per node: $SLURM_NTASKS_PER_NODE
echo CPUs per node: $SLURM_CPUS_PER_NODE

if [[ -n $SLURM_ARRAY_JOB_ID ]]; then
    export UNIQUE_IDENTIFIER=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
else
    export UNIQUE_IDENTIFIER=$SLURM_JOBID
fi
echo UNIQUE_IDENTIFIER: $UNIQUE_IDENTIFIER

JOB_DIR=/pscratch/sd/k/kevinsg/CBC_jobs/
mkdir --parents $JOB_DIR || exit
cd $JOB_DIR || exit
echo Job directory: $PWD

# Display the content of the Python script
PYTHON_SCRIPT=/global/cfs/cdirs/m3586/CBC_ROM/rom_for_gyrokinetics/parametric_global_basis.py
echo "Displaying the content of $PYTHON_SCRIPT"
echo "--------------------------------------------------"
cat $PYTHON_SCRIPT
echo "--------------------------------------------------"

START_TIME=$(date +%s)
srun python $PYTHON_SCRIPT
EXIT_CODE=$?
END_TIME=$(date +%s)
echo Slurm elapsed time $(( (END_TIME - START_TIME)/60 )) min $(( (END_TIME - START_TIME)%60 )) s

exit $EXIT_CODE