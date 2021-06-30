LOG_INT=1000

# For Mujoco headless rendering to recognize Slurm GPUs
# For `srun`:
CUDA_VISIBLE_DEVICES=$SLURM_STEP_GPUS
# For `sbatch` (?):
# GPUS=$SLURM_STEP_GPUS

python -u train.py --log_interval $LOG_INT --resume --purge |& tee print_logs.txt