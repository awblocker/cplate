
import sys
import getopt

import numpy as np

from cplate.libio import *


def simulate_permutation_null(cfg):
  # Extract paths
  y_path = cfg['data']['chrom_path'].strip().format(**cfg)
  regions_path = cfg['data']['regions_path'].strip().format(**cfg)
  null_path = cfg['data']['null_path'].strip().format(**cfg)
  
  # Load reads data
  Y = []
  with open(y_path, "rb") as f:
    for line in f:
      Y.append(np.fromstring(line.strip(), sep=','))
  
  # Load region type information
  regionTypes = []
  with open(regions_path, 'rb') as f:
    for line in f:
      regionTypes.append(np.fromstring(line.strip(), sep=' ', dtype=int))
  
  for chrom in xrange(len(regionTypes)):
    # Normalize region types
    regionTypes[chrom] -= regionTypes[chrom].min()

    # Iterate over unique regions
    regionIDs = np.unique(regionTypes[chrom])

    for ID in regionIDs:
      region = np.where(regionTypes[chrom]==ID)[0]
      region = slice(np.min(region), np.max(region))
  
      n = np.ceil(np.sum(Y[chrom][region]))
      nullRegion = np.random.multinomial(n, np.ones(region.stop - region.start)/
                                         (region.stop - region.start + 0.0))
      Y[chrom][region] = nullRegion
    
  # Write simulated reads to null file
  with open(null_path, 'wb') as f:
    for y_null in Y:
      np.savetxt(f, y_null[np.newaxis,:], fmt="%d", delimiter=',')
  
  return 0

