#!/bin/bash

#SBATCH -J chr20_detect
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

  # Run base pair level summaries
  cplate_detect_mcmc --chrom $chrom --pm 1 --threshold 0.68 $config
  cplate_detect_mcmc --chrom $chrom --pm 3 --threshold 0.77 $config
}

run_mcmc $CONFIG $GENE_ID
