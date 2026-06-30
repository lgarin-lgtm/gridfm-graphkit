#!/bin/bash
#SBATCH --job-name=entrenamiento
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10        
#SBATCH --mem 2GB                
#SBATCH --time=1:00:00  

#SBATCH --partition=vicomtech
#SBATCH --qos=qos_di13

#SBATCH --output=logs/output_%j.log   # Salva il flusso standard (stdout)
#SBATCH --error=logs/error_%j.err     # Salva il flusso di errore (stderr)

# --- CONFIGURAZIONE NOTIFICHE EMAIL ---
#SBATCH --mail-user=lgarin@vicomtech.org  # Indirizzo email del destinatario
#SBATCH --mail-type=ALL    


# Cargar entorno limpio y la versión de Python requerida
module purge
module load Python/3.11.5-GCCcore-13.2.0

echo "Activando entorno virtual..."
source ".venv/bin/activate"

echo "Actualizando pip..."
#pip install --upgrade pip
#pip install --default-timeout=1000 "urllib3>=2.6.0"
#pip install litlogger

#echo "Instalando GridFM-GraphKit en modo editable..."
#pip uninstall -y torch torchvision torchaudio
#pip cache purge
#pip install -r requirements.txt --force-reinstall --no-cache-dir



# 2. Navigate to your locally modified folder
cd /gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/gridfm-graphkit

# 3. Create and activate the virtual environment
# We check if 'venv' exists so it only installs the first time you run it.
#if [ ! -d ".venv" ]; then
    #python -m venv .venv
    #source .venv/bin/activate
    #pip install -e .  # Installs your local, modified code
#else
    source .venv/bin/activate
#fi

# 4. Run your training command

echo "Iniciando entrenamiento..."

gridfm_graphkit train --config /gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/gridfm-graphkit/examples/config/gridFMv0.2_pretraining.yaml --data_path /gpfs/VICOMTECH/proiektuak/DI13/SYSTEMICO/data_out

echo "Entrenamiento finalizado."