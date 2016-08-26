# Load libraries
library(plyr)
library(stringr)
library(reshape2)
library(yaml)
library(qvalue)
library(ascii)
library(argparse)
library(data.table)

kProg <- 'Rscript analyze_fdr.R'
kDescription <- 'Run FDR analysis based on null and nonnull posterior summaries.'
kEpilogue <- 'Requires the argparse R package.\\n\\n'

kParser <- ArgumentParser(description = kDescription, epilog = kEpilogue,
                          prog = kProg)

kParser$add_argument(
  'config', metavar = 'CONFIG', type = 'character', nargs = 1,
  help = 'A YAML configuration file.')
kParser$add_argument(
  'outpath', metavar = 'OUTFILE', type = 'character', nargs = 1,
  help = 'Path for text output.')
kParser$add_argument(
  '--chrom', default = '', type = 'character',
  help = 'Comma-separated list of chromosomes to analyze. Defaults to all.')
kParser$add_argument(
  '--regexp', type = 'character', default = 'p_local_concentration.*',
  help = 'Regular expression (PCRE) for test statistics [default %(default)s]')
kParser$add_argument(
  '--fdr', default = '0.2,0.1,0.05,0.01,0.005,0.001',
  help = 'Comma-separated list of FDRs to analyze [default %(default)s].')
kParser$add_argument(
  '--sample', default = 0, type = 'integer',
  help = 'Optional sample size to use for test statistics [default %(default)s].')
kParser$add_argument(
  '--tags', default = '', type = 'character',
  help = 'Comma-separated list of tags to parse in CONFIG.')


# Function definitions

ReadColumnViaPipe <- function(path, column, ...) {
  # Pattern for awk-based field extraction
  kPipePattern <- "awk '{print $%d}' %s" 
  conn <- pipe(sprintf(kPipePattern, column, path))
  
  dat <- NULL
  tryCatch(dat <- scan(conn, ...), finally=close(conn))
  return(dat)
}

LoadColumnsFromRegexp <- function(path, regexp.var, sep = perl('\\s')) {
  conn <- file(path)
  var.names <- readLines(conn, n = 1)
  close(conn)

  var.names <- str_split(var.names, sep)[[1]]
  columns <- which(str_detect(var.names, regexp.var))
  var.names <- var.names[columns]

  column.list <- lapply(columns, function(col)
    ReadColumnViaPipe(path, col, skip = 1))
  names(column.list) <- var.names
  return(as.data.frame(column.list))
}

LoadViaList <- function(paths, ids, id.var = 'id', load.fun = read.table, ...) {
  data.list <- lapply(paths, load.fun, ...)

  for (i in 1:length(data.list)) {
    data.list[[i]][[id.var]] <- ids[i]
  }

  data.df <- rbindlist(data.list)
  rm(data.list)
  gc()

  return(data.df)
}

GetFdrInformation <- function(obs.stat, null.stat, fdrs, ...) {
  # Compute empirical CDF of null statistics
  null.p.ecdf <- ecdf(null.stat)
  # Compute p-values of observed statistics
  obs.p.values <- 1 - null.p.ecdf(obs.stat)
  stats <- list(
    Unadjusted = obs.p.values,
    qvalue = qvalue(obs.p.values)$qvalue,
    BH = p.adjust(obs.p.values, method = 'BH'),
    BY = p.adjust(obs.p.values, method = 'BY'))
  results <- list()
  # Iterate over FDRs of interest
  for (fdr in fdrs) {
    for (m in names(stats)) {
      info <- data.table(
        fdr = fdr, method = m,
        threshold.p.value = max(obs.p.values[stats[[m]] <= fdr]),
        threshold.stat = min(obs.stat[stats[[m]] <= fdr]))
      results[[length(results) + 1]] <- info
    }
  }
  return(rbindlist(results))
}

ParseConfig <- function(config, tags='id', value.tag=NULL) {
  # Parses simple Python-style string substitutions in config list. Runs
  # recursively to parse entire nested list structure.
  #
  # Args:
  #   config: A list generated from YAML config file
  #   tags: A character vector of substitutions to parse
  #   value.tag: Used in recursions. Leave as NULL.
  #
  # Returns:
  #   A list with the same structure as config with simple string
  #   substitutions parsed
  #
  # Run sequentially, once per tag
  for (tag in tags) {
    if (is.null(value.tag))
      value.tag <- config[[tag]]
    
    config <- lapply(config, function(entry) {
      if (is.list(entry))
        ParseConfig(entry, tags=tags, value.tag=value.tag)
      else
        str_replace(entry, fixed(str_c('{', tag, '}')),
                    value.tag)
    })
  }
  return(config)
}

LoadConfig <- function(optargs) {
  cfg <- yaml.load_file(optargs$config)
  if (str_length(optargs$tags[1]) > 0) {
    cfg <- ParseConfig(config = cfg, tags = optargs$tags)
  }
  if (length(optargs$chrom) == 0) {
    optargs$chrom <- 1:cfg$data$n_chrom
  }
  return(cfg)
}

Main <- function(argv) {
  # Parse options
  optargs <- kParser$parse_args(argv)

  optargs$fdr <- as.numeric(strsplit(optargs$fdr, ',', fixed = TRUE)[[1]])
  optargs$chrom <- as.integer(strsplit(optargs$chrom, ',', fixed = TRUE)[[1]])

  # Load configuration file
  cfg <- LoadConfig(optargs)

  # Load test statistics
  obs <- LoadViaList(
    paths = sprintf(cfg$mcmc_output$summary_pattern, optargs$chrom),
    ids = optargs$chrom, id.var = 'chrom', load.fun = fread)

  null <- LoadViaList(
    paths = sprintf(cfg$mcmc_output$null_summary_pattern, optargs$chrom),
    ids = optargs$chrom, id.var = 'chrom', load.fun = fread)

  # Get names of test statistics
  names.test.stats <- intersect(
    names(obs)[str_detect(names(obs), perl(optargs$regexp))],
    names(null)[str_detect(names(obs), perl(optargs$regexp))])

  # Run analysis on each test statistic
  fdr.output <- ldply(
    names.test.stats,
    function(name.stat, obs, null, fdrs, sample.size, ...) {
      if (sample.size > 0) {
        data.frame(test.stat = name.stat, GetFdrInformation(
          obs.stat = sample(obs[[name.stat]], sample.size),
          null.stat = null[[name.stat]],
          fdrs = fdrs, ...))
      } else {
        data.frame(test.stat = name.stat, GetFdrInformation(
          obs.stat = obs[[name.stat]],
          null.stat = null[[name.stat]],
          fdrs = fdrs, ...))
      }
    },
    obs = obs, null = null, fdrs = optargs$fdr,
    sample.size = optargs$sample, .progress = 'time')

  # Save output
  print(ascii(fdr.output, include.rownames = FALSE, header = FALSE,
              format = 'g'),
        format = 'rest',
        file = optargs$outpath)
}


ToLong <- function(df) {
  df$position <- seq(nrow(df))
  melt(df, id.vars = c("chrom", "position"))
}

if (sys.nframe() == 0) {
  argv <- commandArgs(TRUE)
  Main(argv)
}
