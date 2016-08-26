#!/bin/bash

#SBATCH -J chr20_mcmc_null
#SBATCH --array=1-1721
#SBATCH -n 6
#SBATCH -p airoldi
#SBATCH -t 0
#SBATCH --mem-per-cpu=4096
#SBATCH -o logs/%j.STDOUT
#SBATCH -e logs/%j.STDERR
#SBATCH --mail-user=ablocker@gmail.com
#SBATCH --mail-type=ALL

readonly CONFIG="config/gaffney_chr20_genes.json"
readonly DRAW_PATTERN="mcmcOutput/mcmc_draws_control_gaffney-chr20_gene%02d.tar"

readonly GENE_ID=${SLURM_ARRAY_TASK_ID}

run_mcmc() {
  # Args:
  #   config: Path to YAML config file.
  #   draw_pattern: Printf pattern for archives of MCMC draws.
  #   chrom: Chromosome index.

  local config="$1"
  local draw_pattern="$2"
  local chrom="$3"

  local draws
  draws="$(printf "${draw_pattern}" $chrom)"

  # Halt if draws already exist.
  if [ -e ${draws} ]
  then
    continue
  fi

  # Create a directory for logs if needed.
  local experiment_name
  experiment_name="$(basename $config)"
  experiment_name="${experiment_name%.*}/null"
  local log_dir=logs/$experiment_name 
  stat $log_dir || mkdir -p $log_dir

  # Run the MCMC.
  mpirun -np 6 cplate_deconvolve_mcmc \
    -c $chrom \
    --null \
    $config \
    1>${log_dir}/${chrom}.STDOUT \
    2>${log_dir}/${chrom}.STDERR
}

run_mcmc $CONFIG $DRAW_PATTERN $GENE_ID
