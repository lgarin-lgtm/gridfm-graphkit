#!/bin/bash
#SBATCH --job-name=entrenamiento
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=vicomtech
#SBATCH --qos=qos_di13
#SBATCH --gres=gpu:1
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.err
#SBATCH --mail-user=lgarin@vicomtech.org
#SBATCH --mail-type=ALL

# ponytail: -e aborta si module load / activate fallan; sin -u por el func 'module' de Lmod
set -eo pipefail

PROJECT_DIR=/Users/carmine/Code/TFMLeire/gridfm-graphkit
CONFIG="$PROJECT_DIR/examples/config/mc_finetune.yaml"
DATA_PATH=/Users/carmine/Code/TFMLeire/gridfm-graphkit/examples/data
MODEL_PATH=/Users/carmine/Code/TFMLeire/gridfm-graphkit/examples/models/GridFM_v0_2.pth

# Entorno limpio + Python requerido
#module purge
#module load Python/3.11.5-GCCcore-13.2.0

cd "$PROJECT_DIR"

echo "Activando entorno virtual..."
source .venv/bin/activate
# Primera vez (crear venv + deps): python -m venv .venv && source .venv/bin/activate && pip install -e .

echo "Iniciando entrenamiento..."
gridfm_graphkit finetune --config "$CONFIG" --data_path "$DATA_PATH" --model_path "$MODEL_PATH"
echo "Entrenamiento finalizado."
