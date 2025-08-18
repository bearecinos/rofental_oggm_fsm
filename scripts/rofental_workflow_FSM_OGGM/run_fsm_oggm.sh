#!/bin/bash
#SBATCH --account=geos_extra # Ensure the job runs in the correct queue
#SBATCH --nodelist=forth03
#SBATCH --output=slurm_log.out
#SBATCH --error=slurm_log.err
#SBATCH --job-name=run_fsm_oggm
#SBATCH --time=24:00:00  # Set max run time (adjust as needed)
#SBATCH --mem=48GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # Adjust CPU allocation if needed
#SBATCH --mail-type=ALL  # Notifications for job begin, end, and failure
#SBATCH --mail-user=beatriz.recinos@ed.ac.uk # Email for notifications

# Load necessary modules (if required)
source ~/.bashrc
mamba activate oggm_fsm

SCRIPT_DIR="$(pwd)"
ROOT_DIR="$(realpath "$SCRIPT_DIR/..")"
export PYTHONPATH="/exports/csce/datastore/geos/users/brecinos/FSM-OGGM:$PYTHONPATH"

# Path to your single config file
CONFIG="$SCRIPT_DIR/params.ini"

# Now call each Python script with that one argument:
python test_fsm_rofental_new.py "$CONFIG"

#python test_fsm_rofental.py $(<params.txt)
#python output_distributed_thickness_and_runoff.py $(<params.txt)
#python output_terminus_position_to_runoff_file.py $(<params.txt)
#python output_area_change_shapefiles.py $(<params.txt)
