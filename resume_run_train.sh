#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00

dataset="LJSpeech"
SCRATCHDIR="/scratch/anandaswarup/"

mkdir -p ${SCRATCHDIR}
mkdir -p ${SCRATCHDIR}/datasets/
mkdir -p ${SCRATCHDIR}/experiments/

# Copy dataset tarfile from /share1/anandaswarup/TTS/datasets to /scratch/anandaswarup/datasets
echo "Begin dataset copy"
rsync -aP anandaswarup@ada:/share1/anandaswarup/TTS/datasets/${dataset}.tar.gz ${SCRATCHDIR}/datasets/
echo "Dataset copied"

# Untaring the dataset
echo "Begin dataset untar"
tar -xzvf ${SCRATCHDIR}/datasets/${dataset}.tar.gz -C ${SCRATCHDIR}/datasets/
echo "Untar complete"

rm ${SCRATCHDIR}/datasets/${dataset}.tar.gz
echo "Removed tar file"

# Copying the checkpoints and logs from /share1/anandaswarup/TTS/experiments to /scratch/anandaswarup
echo "Begin checkpoints and logs copy"
rsync -aP anandaswarup@ada:/share1/anandaswarup/TTS/experiments/${dataset}_deepvoice3 ${SCRATCHDIR}/experiments/
echo "Copy complete"

# Model training
echo "Resuming training from specified checkpoint"
python train.py --data_dir ${SCRATCHDIR}/datasets/${dataset}/ --checkpoint_dir ${SCRATCHDIR}/experiments/${dataset}_deepvoice3/checkpoints/ --logs_dir ${SCRATCHDIR}/experiments/${dataset}_deepvoice3/logs/ --checkpoint_path ${SCRATCHDIR}/experiments/${dataset}_deepvoice3/checkpoints/checkpoint_step000710000.pth

# Backup checkpoints and logs to /share1/anandaswarup/TTS/experiments
echo "Begin checkpoints and logs copy"
rsync -aP ${SCRATCHDIR}/experiments/${dataset}_deepvoice3 anandaswarup@ada:/share1/anandaswarup/TTS/experiments/
echo "Copy complete"

# Cleanup /scratch dir
rm -r ${SCRATCHDIR}
