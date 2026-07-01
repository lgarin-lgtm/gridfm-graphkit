#!/bin/bash
#SBATCH --job-name=entrenamiento
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=vicomtech
#SBATCH --qos=qos_di13
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.err
#SBATCH --mail-user=lgarin@vicomtech.org
#SBATCH --mail-type=ALL

# ponytail: -e aborta si module load / activate fallan; sin -u por el func 'module' de Lmod
set -eo pipefail

PROJECT_DIR=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/Carmine/gridfm-graphkit
CONFIG="$PROJECT_DIR/examples/config/mc_finetune.yaml"
DATA_PATH=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/data_out
MODEL_PATH=/gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/Carmine/gridfm-graphkit/examples/models/GridFM_v0_2.pth

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
