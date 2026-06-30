#!/bin/bash
#SBATCH --job-name=generate_data
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10        
#SBATCH --mem 16G                
#SBATCH --time=5:00:00  

#SBATCH --partition=vicomtech
#SBATCH --qos=qos_di13

#SBATCH --output=logs/output_%j.log   # Salva il flusso standard (stdout)
#SBATCH --error=logs/error_%j.err     # Salva il flusso di errore (stderr)

# --- CONFIGURAZIONE NOTIFICHE EMAIL ---
#SBATCH --mail-user=lgarin@vicomtech.org  # Indirizzo email del destinatario
#SBATCH --mail-type=ALL          

# --- Convert the .parquet files to CSV ---
source ~/venvs/parquet_env/bin/activate
python Convert_Data.py


# --- To do the training of the new model Montecarlo dropout ---
