#!/usr/bin/env bash
#SBATCH --job-name=a2_worker
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/logs/worker_%j.out
# Holds one GPU and runs every shell script dropped into jobs/queue/ in name order. Touch jobs/STOP to exit.
# Submit from the repository root (mkdir -p jobs/{queue,running,done,logs}). The job inherits the submitting
# shell's environment, so export LIBERO_ROOT, and TORCH_HOME / HF_HOME if home is not mounted on the GPU nodes.
set -u
cd "$SLURM_SUBMIT_DIR"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
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
