#!/usr/bin/env bash
#SBATCH --job-name=a2_worker
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/logs/worker_%j.out
# Holds one GPU and runs every shell script dropped into jobs/queue/ in name order. Touch jobs/STOP to exit.
set -u
cd /nas/Selim/selim_sarowar/AML_Assignment_World-Model
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TORCH_HOME=/nas/Selim/selim_sarowar/hf_cache/torch HF_HOME=/nas/Selim/selim_sarowar/hf_cache
export PYTHONPATH=$PWD OMP_NUM_THREADS=8
while [ ! -e jobs/STOP ]; do
  next=$(ls jobs/queue/*.sh 2>/dev/null | head -1)
  if [ -z "$next" ]; then sleep 15; continue; fi
  name=$(basename "$next" .sh)
  mv "$next" "jobs/running/$name.sh"
  echo "[$(date +%F' '%T)] start $name"
  bash "jobs/running/$name.sh" > "jobs/logs/$name.log" 2>&1
  echo "[$(date +%F' '%T)] end $name exit=$?"
  mv "jobs/running/$name.sh" "jobs/done/$name.sh"
done
