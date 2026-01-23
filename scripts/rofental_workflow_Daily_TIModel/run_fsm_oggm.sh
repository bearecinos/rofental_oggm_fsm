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

# Path to your single config file
CONFIG="$SCRIPT_DIR/params.ini"

# Now call each Python script with that one argument:
python test_oggm_rofental.py "$CONFIG"
python output_distributed_thickness.py "$CONFIG"
#python output_terminus_position_to_runoff_file.py "$CONFIG"
#python output_area_change_shapefiles.py "$CONFIG"
