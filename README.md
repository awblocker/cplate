# Analyzing nucleosome positioning with genome-wide Bayesian deconvolution

This README covers the details of carrying out an analysis of nucleosome
positing from high-throughput sequencing data using the methods of Blocker and
Airoldi (2016). Analyzing a single experiment separates into 3 broad phases:

1. Data management: Aligning, parsing, and reducing the raw sequencing data
(typically FASTQ files) into the form required in for statistical analysis.

2. Estimation: Estimating the segmentation of the genome and digestion-error
templates from the reduced sequencing data. Running Bayesian deconvolution
(via distributed HMC). Processing MCMC draws into posterior summaries.

3. Analysis: Subsequent biological analyses, using the estimates from
deconvolution as inputs (features). Clustering, selecting regions of
interest, assessing reproducibility, and so on.

Two examples are provided, one toy based on a single chromosome of fake data in
`examples/toy` and one based on the gene-by-gene analysis of H. sapiens
chromosome 21 from the Gaffney et al. (2012) data in `examples/human`. The
former provides examples of running on an LSF-managed cluster, whereas the
latter provides examples of running on a SLURM-managed cluster. `cplate` can be
run on MPI-compatible cloud clusters such as
[StarCluster](http://star.mit.edu/cluster/).

## Architecture

Every script in `cplate` uses YAML/JSON configuration files. Each file
describes all of the data, parameters, and outputs for a single experiment's
dataset. These files must be created during the data management phase of the
analysis. There are a lot of fields, paths, and patterns to configure in each
file, but they are entirely machine- and human-readable for each configuration.
The `config` folder contains `example.yml`, a fully commented example of this
configuration file for a small dataset. This file is the canonical reference
for the YAML configuration structure and requirements.

## Tools

Each phase uses a distinct set of tools. The data management phase uses:

* [bowtie](http://bowtie-bio.sourceforge.net/index.shtml):
Aligns fragments obtained from high-throughput sequencing. Takes raw FASTQ
files and a reference genome as input, output a SAM file containing
alignments. This would be better replaced by bwa, a more modern aligner that can
handle the ALT contigs of hg38.

* [samtools](http://www.htslib.org/) and
[pysam](https://github.com/pysam-developers/pysam):
Tools for manipulating SAM and BAM files, which are standards for the storage of
alignments from high-throughput sequencing. The entire SAM specification is
available at
[http://samtools.sourceforge.net/SAM1.pdf](http://samtools.sourceforge.net/SAM1.pdf).

* [`pipeline`](https://github.com/awblocker/paired-end-pipeline):
A custom Python package that (with `samtools` and `pysam`) does the bulk of the
heavy lifting for parsing the SAM files and extracting the relevant information.

* `bash` scripts:
Assorted `bash` scripts to coordinate everything. Simple, easily maintained
glue.

All of the above components can be substituted with any workflow that provides
fragment center counts and length distributions in the formats required by
`cplate`.

The estimation phase requires fewer tools, but they are more specialized:

* [`cplate`](https://github.com/awblocker/cplate):
The grand kahuna. The big one. Where all of the deconvolution action happens.
This is a custom Python package that handles all of the segmentation, template
estimation, deconvolution, and posterior summaries. It's big, it's complex, but
it's also modular.

* `bash` scripts:
More glue. These are primarily `bsub` scripts that interface the distributed HMC
with the Harvard Odyssey cluster.

* R and Python scripts:
Specialized R and Python scripts for the particular phases. The most important
of these is `R/analyze_fdr.R`, which runs the FDR calibration specified in
Blocker and Airoldi (2016). Another very useful pair is `detections_to_bed.py`
and `clusters_to_bed.py`, which convert detection and cluster output from
`cplate` into the [BED](https://genome.ucsc.edu/FAQ/FAQformat.html#format1)
format, which is a standard in the field and can be viewed in
[IGV](https://www.broadinstitute.org/igv/) and similar tools. BED files can
also be used with the [Galaxy](https://usegalaxy.org/) platform,
[bedtools](http://bedtools.readthedocs.org/en/latest/), and other standard
bioinformatics packages.


The analysis phase takes a more diverse set of tools, and their selection is
almost entirely up to the analyst.

## Requirements

For the example data management phase, `bowtie`, `samtools`, `pysam`, and
`pipeline` must be installed. `pipeline` requires Python 2.7 or newer and
`numpy` in addition to `pysam`. Using the GUI version of `pipeline` requires
the `wx` Python package and a working `wx` installation.

For estimation, `cplate` requires the Python packages `mpi4py`, `yaml`, `numpy`,
and `scipy`.

For `analyze_fdr.R`, the `plyr`, `stringr`, `reshape2`, `yaml`, `qvalue`, and
`ascii` packages are required from CRAN. The `qvalue` package from Bioconductor
is also required.

To use `lib.R` and `lib.cpp` in subsequent analyses, the only requirement is the
`Rcpp` package.

## Usage - Case study

To illustrate how the entire analysis works, we're going to go through an
example from the raw FASTQ files to the final posterior summaries. The
estimation portion corresponds to the `example.yml` file. However, the data
management portion does not have a corresponding data set for practical reasons.

Throughout this example, we will be working with the following directory
structure:
    
    [user@system work]$ ls .
    config data results logs

### Data management

Suppose we receive a set of FASTQ files from our collaborators as
`experiment.fastq.tar.gz`. We extract this into the `data` directory and
find two files:

    [user@system data]$ ls .
    experiment.R1.fastq experiment.R2.fastq

These correspond to the forward and reversed ends from paired-end sequencing.
The first step in our data processing is to align these reads to a reference
genome. Assuming that we're working with S. Cerevisiae data, we can use the
files provided at the [bowtie
website](http://bowtie-bio.sourceforge.net/index.shtml).
Their reference genomes consist of a compressed set of `.ebwt` files, all of
which should be extracted into the `data` directory.
As an aside, the `.ebwt` file extension refers to the Burrows-Wheeler
transformation used by `bowtie` for efficient alignment.
After this extraction, we have:

    [user@system data]$ ls .
    experiment.R1.fastq experiment.R2.fastq s_cerevisiae.1.ebwt
    s_cerevisiae.2.ebwt s_cerevisiae.3.ebwt s_cerevisiae.4.ebwt
    s_cerevisiae.rev.1.ebwt s_cerevisiae.rev.2.ebwt

We're now ready to run `bowtie` and align our reads to the reference genome. A
typical command for this (run in the `data` directory) will be:

    [user@system data]$ N_THREADS=$LSB_DJOB_NUMPROC

    [user@system data]$ time bowtie --phred33-quals -q \
        -n 1 --best -M 1 \
        -I 100 -X 300 \
        --threads $N_THREADS \
        -S \
        s_cerevisiae \
        -1 experiment.R1.fastq -2 experiment.R2.fastq \
        1>experiment.sam \
        2>align_experiment.log

This takes a bit of explanation and some quality time with the
[`bowtie` manual](http://bowtie-bio.sourceforge.net/manual.shtml) to parse.
There are only a few classes of options and arguments to `bowtie` used above,
though:

* `--phred33-quals -q` tells `bowtie` that the inputs are FASTQ files and how to
interpret the quality strings from these files.

* `-n 1 --best -M 1 -I 100 -X 300` sets the
alignment policy. `-n 1` tells `bowtie` to discard any alignment with more than
1 nucleotide mismatched. `--best -M 1` tells `bowtie` to report a single
randomly-sampled alignment among all valid alignments with the best quality.
`-I 100 -X 300` tells `bowtie` to consider only those alignments with lengths
between 100 and 300 base pairs; this only makes sense because we are looking for
nucleosomal DNA, which is about 150bp in length.

* `--threads $N_THREADS` tells `bowtie` to use multiple threads for its
alignment. This can speed up alignment immensely; I've found 12 cores on a
single Odyssey node to be very effective for S. cerevisiae datasets. If this is
running inside of an LSF job, the `$LSB_DJOB_NUMPROC` environment variable will
contain the number of processor allocated to the job.

* `-S` tells `bowtie` to provide SAM-formatted output. Without this, it's in a `bowtie`-specific format.

* `s_cerevisiae` is the name of the reference genome (note the lack of file extension). It's also the only positional argument in this whole command.

* `-1 experiment.R1.fastq -2 experiment.R2.fastq` specifies that the given files each contain one end from paired end sequencing.

* `1>experiment.sam` sends the output from `bowtie`, which is printed to `stdout` by default, to `experiment.sam`

* `2>align_experiment.log` sends the `stderr` output of `bowtie` to the given log file. This contains useful statistics about the proportions of reads that aligned at different levels of specificity.

Once this alignment is complete, there are three steps left:

    # Parse SAM output into read list
    [user@system data]$ time parseSAMOutput.py \
        experiment.sam \
        1>reads_experiment.txt \
        2>parseSAM_experiment.log

    # Extract length distribution from read list
    [user@system data]$ time buildReadLengthDist.py \
        reads_experiment.txt \
        1>lengthDist_experiment.txt \
        2>buildLengthDist_experiment.log

    # Convert reads to counts of centers per base pair
    [user@system data]$ time readsToChromCounts.py \
        --randomize --seed=20130530 \
        reads_experiment.txt \
        1>y_experiment.txt \
        2>count_experiment.log

All three scripts here are part of the `pipeline` package. The first,
`parseSAMOutput.py`, converts the SAM output from `bowtie` to a condensed list
of aligned reads. This simplifies subsequent processing and any read-level
analyses that you choose to do later. In the process of doing this conversion,
`parseSAMOutput.py` converts the SAM input to BAM (binary SAM), then to a sorted
BAM file. These conversions make later processing much more efficient. The BAM
file is also much smaller than the SAM file. The latter can be removed after
this step, as `parseSAMOutput.py` can also (intelligently) use BAM and sorted
BAM inputs if you need to rerun it.

The second script, `buildReadLengthDist.py`, extracts the distribution of
aligned read lengths from the condensed list of reads. Its output consists of a
space-separated file with two columns. The first column is the read length in
base pairs, and the second is the number of aligned reads with that length. This
is needed to estimate the distribution of digestion errors.

The third scripts, `readsToChromCounts.py`, reduces the list of aligned reads to
the number of aligned reads centered at each base pair of the genome. It outputs
a ragged, comma-separated array to `stdout`. Each row of this array contains the
counts for each base pair of a single chromosome. The
`--randomize --seed=20130530` options tell the script to randomly round fragment
centers to integers, using the given RNG seed. I recommend setting the seed
explicitly to ensure that your processing is reproducible.

### Estimation

With this data (and our `config/example.yml` file) in hand, we can move on to
estimation. This needs to be done in a particular sequence, but everything uses
the `cplate` package. The sequence is

1. `cplate_estimate_template` : Estimate template

2. `cplate_segment_genome` : Segment genome

3. `cplate_simulate_null` : Simulate from permutation null

4. `cplate_deconvolve_mcmc` : Run Bayesian deconvolution via distributed HMC on observed and null data.

5. `cplate_summarise_mcmc`, `cplate_summarise_clusers_mcmc`, `cplate_summarise_clusters_mcmc` : Extract posterior summaries from MCMC draws

Each of these scripts takes one (or more, for some of them) YAML configuration
files as inputs. So long as these configuration files are properly configured,
everything gets pulled from and put to the right place without further tweaking
and settings. Each script also has `-h` and `--help` options that provide
descriptions of all options that may be needed.

To start, we estimate the template using

    [user@system work]$ cplate_estimate_template config/example.yml

Then, we turn to the segmentation. This requires an additional file specifying
the location of each ORF in the genome, as specified in the script's `--help`:

    [user@system work]$ cplate_segment_genome --help

    Usage: cplate_segment_genome [options] GENEINDEX CONFIG [CONFIG ...]

    Options:
      -l/--minlength=       Minimum length of segments. Defaults to 800.
      -s/--sep=             Separator for GENEINDEX input. Defaults to \t.
      -v/--verbose=         Set verbosity of output. Defaults to 0.
      -h, --help            Show this help message and exit

    Segments a genome using the hierarchical merging algorithm of Blocker and
    Airoldi 2016.

    GENEINDEX must be a path to a SEP-delimited file containing at least the
    following fields with appropriate column labels:
      - chromosome : Integer (starting from 1) chromosome number
      - start : Beginning of each ORF (TSS, preferably)
      - stop : End of each TSS
    Can have start > stop or stop > start depending on the orientation of each
    gene along the chromosome.

    Details of the required format for the YAML CONFIG files can be found it
    further documentation.

So, to segment the genome for our example, we run:

    [user@system work]$ cplate_segment_genome --minlength=800 \
        data/gene_index.txt config/example.yml

We can then simulate reads according to our permutation null with:

    [user@system work]$ cplate_simulate_null config/example.yml

With those steps complete we can run the MCMC-based deconvolution algorithm on
the observed and simulated data. This uses the `cplate_deconvolve_mcmc` script,
which has the following `--help`:


    Usage: cplate_deconvolve_mcmc [options] CONFIG [CONFIG ...]

    Options:
      -h, --help            Show this help message and exit
      -c CHROM, --chrom=CHROM
                            Comma-separated indices of chromosomes to analyze;
                            defaults to 1
      --null                Run using null input from CONFIG
      --both                Run using both actual and null input from CONFIG
      --all                 Run all chromosomes

    Details of the required format for the YAML CONFIG files can be found it
    further documentation.

These options are used for `cplate_deconvolve_mcmc` and all of the
`cplate_summarise*` scripts. By default, each script will run on chromosome 1 of
a single file. `cplate_deconvolve_mcmc` is unique in that it must be called via
`mpirun` or its specialized equivalent on a given cluster. So, to run MCMC-based
deconvolution on our example, we could run the following from the command line:

    [user@system work]$ mpirun -np 4 cplate_deconvolve_mcmc --all --both \
        config/example.yml

This would run the distributed MCMC-based deconvolution `all` chromosomes in our
example (there's only 1) for `both` the observed and null datasets. The `-np 4`
option to `mpirun` tells MPI to use 4 processors.

For actual datasets, we often want far more than 4 processors. An example of 
doing so through a LSF cluster can be found in the `scripts` folder as
`mcmc_example.bsub`:

    #BSUB -J mcmcExample
    #BSUB -q airoldi
    #BSUB -n 120
    #BSUB -a openmpi
    #BSUB -oo logs/mcmc_example.log
    #BSUB -eo logs/mcmc_example.err

    # Run MCMC on observed data
    mpirun.lsf -np $LSB_DJOB_NUMPROC cplate_deconvolve_mcmc \
        --all config/example.yml \
        1>logs/mcmc_example_obs.log \
        2>logs/mcmc_example_obs.err

    # Run MCMC on simulated null data
    mpirun.lsf -np $LSB_DJOB_NUMPROC cplate_deconvolve_mcmc \
        --all --null config/example.yml \
        1>logs/mcmc_example_null.log \
        2>logs/mcmc_example_null.err

This script requests 120 cores (not necessarily contiguous) from the `airoldi`
queue, then uses all of these to run the distributed Bayesian deconvolution
(HMC) algorithm. The `stdout` and `stderr` output from each MCMC run and the
overall job are piped to appropriate files in the `logs/` directory.

In some cases, it can be useful to call `cplate_deconvolve_mcmc` separately for
each chromosome via a `bash` loop. This looks odd, but it may be needed due to
inefficient or incomplete garbage collection in Python between chromosomes when
multiple chromosomes are used in a single call to `cplate_deconvolve_mcmc`. An
example for 16 chromosomes would be:

    NCHROM=16

    for (( CHROM=1; CHROM <= $NCHROM; CHROM++))
    do
        # Run MCMC on observed data
        mpirun.lsf -np $LSB_DJOB_NUMPROC cplate_deconvolve_mcmc \
            --chrom=$CHROM config/example.yml \
            1>logs/mcmc_example_obs_chrom`printf '%02d' $CHROM`.log \
            2>logs/mcmc_example_obs_chrom`printf '%02d' $CHROM`.err

        # Run MCMC on simulated null data
        mpirun.lsf -np $LSB_DJOB_NUMPROC cplate_deconvolve_mcmc \
            --chrom=$CHROM --null config/example.yml \
            1>logs/mcmc_example_null_chrom`printf '%02d' $CHROM`.log \
            2>logs/mcmc_example_null_chrom`printf '%02d' $CHROM`.err
    done

Once these MCMC runs are complete, the posterior summaries are straightforward
to run. However, they do require a substantial amount of memory.
An example `bsub` script for running the summaries is provided as
`scripts/summaries_example.bsub` and reproduced below:

    #BSUB -J summariesExample
    #BSUB -q airoldi
    #BSUB -n 12
    #BSUB -R "span[ptile=12]"
    #BSUB -a openmpi
    #BSUB -oo logs/summaries_example.log
    #BSUB -eo logs/summaries_example.err

    # Iterate over null and nonnull cases
    for NULL in "" --null
    do
        # Run base pair level summaries
        cplate_summarise_mcmc --all --mmap $NULL config/example.yml

        # Run hyperparameter summaries
        cplate_summarise_params_mcmc --all $NULL config/example.yml

        # Run cluster level summaries
        cplate_summarise_clusters_mcmc --all $NULL config/example.yml
    done

The `--mmap` option for `cplate_summarise_mcmc` is very important. It tells the
script to access the MCMC draws iteratively without loading everything into
memory at once. It's not very fast, but it has a huge effect on memory usage.

Once all of these summaries are complete, we need to calibrate our detections to
a given FDR. This requires some manual input, but it's quite straightforward.
First, we run the `analyze_fdr.R` script to obtain the thresholds corresponding
to various FDRs:

    [user@system work] Rscript analyze_fdr.R \
        config/example.yml results/fdr_example.txt

For further details on `analyze_fdr.R`, it has a thorough `-h/--help` message.
Once this has been run, we inspect the `results/fdr_example.txt` file and select
the appropriate threshold for the `pm` level and FDR of interest. Then, we edit
the `config/example.yml` file to reflect this selection.

Then, we can run the detection algorithm again to reflect this selection with:

    [user@system work] cplate_detect_mcmc --all config/example.yml

This is not the most elegant part of the workflow, and it can certainly be
refined. In particular, it would be better for the configuration files to
specify an FDR with the resulting threshold detection threshold extracted and
stored automatically.


