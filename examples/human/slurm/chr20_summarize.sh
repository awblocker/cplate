#!/bin/bash

#SBATCH -J chr20_summarize
#SBATCH --array=1-1721
#SBATCH -n 12
#SBATCH -p airoldi
#SBATCH -t 0
#SBATCH --mem-per-cpu=4096
#SBATCH --exclusive
#SBATCH -o logs/%j.STDOUT
#SBATCH -e logs/%j.STDERR
#SBATCH --mail-user=ablocker@gmail.com
#SBATCH --mail-type=ALL

readonly CONFIG="config/gaffney_chr20_genes.json"

readonly GENE_ID=${SLURM_ARRAY_TASK_ID}

run_mcmc() {
  # Args:
  #   config: Path to YAML config file.
  #   chrom: Chromosome index.

  local config="$1"
  local chrom="$2"

  # Create a directory for logs if needed.
  local experiment_name
  experiment_name="$(basename $config)"
  experiment_name="${experiment_name%.*}"
  local log_dir=logs/$experiment_name 
  stat $log_dir || mkdir -p $log_dir

  # Run base pair level summaries
  cplate_summarise_mcmc --chrom $chrom --mmap $config
  # Run hyperparameter summaries
  cplate_summarise_params_mcmc --chrom $chrom $config
  # Run cluster level summaries
  cplate_summarise_clusters_mcmc --chrom $chrom $config
}

run_mcmc $CONFIG $GENE_ID
