#!/bin/bash
#SBATCH --job-name=generate_data
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20        
#SBATCH --mem 32G                
#SBATCH --time=10:00:00  

#SBATCH --partition=vicomtech
#SBATCH --qos=qos_di13

#SBATCH --output=logs/output_%j.log   # Salva il flusso standard (stdout)
#SBATCH --error=logs/error_%j.err     # Salva il flusso di errore (stderr)

# --- CONFIGURAZIONE NOTIFICHE EMAIL ---
#SBATCH --mail-user=lgarin@vicomtech.org  # Indirizzo email del destinatario
#SBATCH --mail-type=ALL


echo "Iniciando generación de datos..."

apptainer run --writable-tmpfs --bind config.yaml:/app/config.yaml,./data_out:/data_out image.sif

echo "Proceso finalizado."