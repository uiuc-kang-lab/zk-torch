#!/bin/sh
#SBATCH --partition=ddkang
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --nodelist=ccc0419
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="zktorch_gh_action"
#SBATCH --output=zktorch_gh_action.out
#SBATCH --account=ddkang-cs-eng
#SBATCH --gres=gpu:1

module load cmake/3.26.3
module load cuda/12.4
module load python/3.9.16

srun python test_gpu_scripts/notify_enqueue.py "$1" "$2" "$3"
srun rustup override set nightly

export LIBCLANG_PATH=/projects/illinois/eng/cs/ddkang/bjchen4/gpu/llvm-project/build/lib
srun --export=ALL cargo run --bin zk_torch --features gpu -- config.yaml
srun python test_gpu_scripts/notify_result.py "$1" "$2" "$3"

srun --export=ALL cargo test --features gpu
srun python test_gpu_scripts/notify_test_result.py "$1" "$2" "$3"
