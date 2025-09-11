#!/bin/bash
#SBATCH --account=  # Ensure the job runs in the correct queue
#SBATCH --nodelist=forth03
#SBATCH --output=slurm_log.out
#SBATCH --error=slurm_log.err
#SBATCH --job-name=run_fsm_oggm
#SBATCH --time=24:00:00  # Set max run time (adjust as needed)
#SBATCH --mem=48GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # Adjust CPU allocation if needed
#SBATCH --mail-type=ALL  # Notifications for job begin, end, and failure
#SBATCH --mail-user=  # Email for notifications

# Load necessary modules (if required)
source ~/.bashrc
mamba activate oggm_fsm

SCRIPT_DIR="$(pwd)"
ROOT_DIR="$(realpath "$SCRIPT_DIR/..")"
# define your project root in one place
export FSM_OGGM_ROOT="../../FSM-OGGM"

# then use it wherever you need
export PYTHONPATH="$FSM_OGGM_ROOT:$PYTHONPATH"

# Path to your single config file
CONFIG="$SCRIPT_DIR/params.ini"

# Now call each Python script with that one argument:
python test_fsm_rofental.py "$CONFIG"
#python output_distributed_thickness_and_runoff.py "$CONFIG"
#python output_terminus_position_to_runoff_file.py "$CONFIG" Not completed!
