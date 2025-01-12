Command for fitting random 2D models on cluster:
`PYTHONUNBUFFERED="TRUE" sbatch --array 0-99 -c 4 --mem-per-cpu=2gb --time 00:30:00 -p burst,miller -A miller --output slurm/fit/%A_%a.out niarb fit fit.toml -N 1 -o fits --linfo --ignore-errors`
Command for fitting random 2D models with numerical validation on cluster:
`PYTHONUNBUFFERED="TRUE" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-19 -c 8 --mem-per-cpu=4gb --time 01:00:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/fit/%A_%a.out niarb fit fit.toml -o fits --linfo --ignore-errors`
Command for running random 2D models on cluster:
`PYTHONUNBUFFERED="TRUE" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" sbatch --array 0-49 -c 8 --mem-per-cpu=4gb --time 00:05:00 -p burst,miller -A miller --gres=gpu:a40:1 --output slurm/run/resp/%A_%a.out niarb run run.toml --linfo`
