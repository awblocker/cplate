# Load packages
import sys
import getopt
import csv
import numpy as np

# Set constants
CHROM_LENGTHS = (230208,813178,316617,1531919,576869,270148,1090947,562643,
                 439885,745741,666454,1078175,924429,784334,1091289,948062)
NCHROM = len(CHROM_LENGTHS)
#
ORF_START = 1
ORF_STRIDE = 2
INTERGENIC_START = 0
INTERGENIC_STRIDE = 2

def calcRegionLengths(regions, regionIds=None):
  '''
  Calculate region lengths; can use existing regionIds or calculate as-needed
  '''
  if regionIds is None:
    regionIds = np.unique(regions)

  # Compute region lengths; efficient list comprehension
  regionLengths = np.array([np.sum(regions==regionId) for regionId in
                            regionIds])
  
  return regionLengths

def calcCoverage(regions, reads, regionIds=None):
  '''
  Calculate coverage by region
  '''
  if regionIds is None:
    regionIds = np.unique(regions)
  
  coverage = np.zeros(np.max(regionIds)+1)
  
  for id in regionIds:
    region = np.where(regions==id)[0]
    coverage[id] = np.mean(reads[region])
  
  return coverage

def mergeRegions(regions, coverage, regionIds=None, minLength=0, normalize=True,
         verbose=False):
  '''
  Iteratively merge regions until minimum length constraint is met.
  
  Greedy algorithm; proceeds starting with best merge and stops when
  constraint is uniformly met
  
  Operates on region definitions for a single chromosome
  '''
  
  # Get initial region IDs if needed
  if regionIds is None:
    regionIds = np.unique(regions)
  
  # Convert IDs and coverage to lists; enables more efficient manipulation
  # In particular, need efficient element deletion
  if type(regionIds) is np.ndarray:
    regionIds = regionIds.tolist()
  if type(coverage) is np.ndarray:
    coverage = coverage.tolist()
    
  # Find initial region lengths
  regionLengths = calcRegionLengths(regions, regionIds)
  regionLengths = regionLengths.tolist()
  
  # Iterate until length constraint is satisfied
  while np.min(regionLengths) < minLength:
    # Setup arrays
    regionLengthsArray = np.array(regionLengths)
    regionIdsArray = np.array(regionIds)
    coverageArray = np.array(coverage)
    
    # Find candidate regions
    candidateInd = np.where(regionLengthsArray < minLength)[0]
    
    candidateQuality = np.zeros(candidateInd.size, dtype=np.float)
    candidateNeighbors = np.zeros(candidateInd.size, dtype=np.int)
    
    # Evaluate candidate merge qualities
    for ii in xrange(candidateInd.size):
      ind = candidateInd[ii]
      # Standard case
      if 0 < ind < regionIdsArray.size-1:
        neighbors = ind + np.array([-1,1], dtype=int)
        quality = -np.abs(coverageArray[neighbors] - coverageArray[ind])
        candidateNeighbors[ii] = neighbors[np.argmax(quality)]
        candidateQuality[ii] = np.max(quality)
      # Boundary cases
      elif ind == 0:
        candidateNeighbors[ii] = 1
        candidateQuality[ii] = -np.abs( coverageArray[0] - coverageArray[1] )
      else:
        candidateNeighbors[ii] = regionIdsArray.size-2
        candidateQuality[ii] = -np.abs( np.diff(coverageArray[-2:]) )
    
    # Select best candidate for merge
    ind = np.argmax( candidateQuality )
    neighbor = candidateNeighbors[ind]
    ind = candidateInd[ind]
    shortId = regionIds[ind]
    
    # Execute merge
    
    # Update regions
    regions[regions==shortId] = regionIds[neighbor]
    
    # Update region IDs
    del regionIds[ind]
    
    # Update coverage
    w = regionLengths[ind]/(regionLengths[neighbor]+regionLengths[ind]+0.0)
    coverage[neighbor] = w*coverage[ind] + (1-w)*coverage[neighbor]
    del coverage[ind]
    
    # Update region lengths
    regionLengths[neighbor] += regionLengths[ind]
    del regionLengths[ind]
  
  # Normalize regions if requested; resets to 0-based sequential ids
  if normalize:
    regionIds, regions = np.unique(regions, return_inverse=True)
  
  return regions

def joinOverlappingOrfs(geneInfoList, geneInfoFields, verbose=False):
  '''
  Combine overlapping ORFs into ORF regions; simplifies subsequent analysis
  '''
  # Initialize ORF region list
  orfRegionList = []
  for l in CHROM_LENGTHS:
    orfRegionList.append(-np.ones(l, dtype=np.int))

  # Initialize region codes
  maxRegionCodes = np.zeros( len(CHROM_LENGTHS), dtype=np.int )
  
  # Iterate through genes
  for gene in geneInfoList:
    # Extract ORF information
    chrom = int(gene['chromosome'])
    tStart = int(gene['start'])
    tStop = int(gene['stop'])
    
    # Check for validity (chrom > len(CHROM_LENGTHS) -> mitochondrial)
    if chrom > len(CHROM_LENGTHS):
      continue
    
    # Get ORF & promoter slices in 0-based indexing
    if tStart < tStop:
      orfStart, orfEnd = tStart-1, tStop
    else:
      orfStart, orfEnd = tStop-1, tStart
      
    orf = slice(orfStart, orfEnd)

    # Check for overlap
    if np.any( orfRegionList[chrom-1][orf] > 0 ):
      orfRegionList[chrom-1][orf] = orfRegionList[chrom-1][orf].max()
    else:
      # Otherwise, define a new region code
      maxRegionCodes[chrom-1] += 1
      orfRegionList[chrom-1][orf] = maxRegionCodes[chrom-1]
  
  # Rescan through regions to assign identifiers to intergenic regions
  # Keeping odd for ORFs, even for intergenic, monotone increasing 3' to 5'
  regionList = []
  for chrom in xrange(len(CHROM_LENGTHS)):
    # Diagnostic info
    if verbose:
      print >> sys.stderr, "Chromosome\t=\t%d" % chrom
    # Initialize region vector
    regionList.append(np.zeros(CHROM_LENGTHS[chrom], dtype=np.int))    
    
    # Initialize tags
    activeOrfTag = ORF_START
    activeIntergenicTag = INTERGENIC_START
    activeTag = 0
    lastTage = 0
    
    # Iterate through regions until end of chromosome
    lastPos = 0
    while lastPos < CHROM_LENGTHS[chrom]-1:
      # Get current region ID
      regionId = orfRegionList[chrom][lastPos]
      
      # Find current region's location
      if regionId > 0:
        # For numbered regions, it's easy
        posInRegion = np.where(orfRegionList[chrom] == regionId)[0]
      else:
        # For zero regions, it's more subtle
        posInRegion = np.where(orfRegionList[chrom] == regionId)[0]
        diff = np.diff(posInRegion)
        if np.any(diff>1):
          posInRegion = posInRegion[:np.min(np.where(diff>1)[0])+1]
        del diff
      
      # Set appropriate tag for current location
      if regionId > 0:
        regionList[chrom][posInRegion] = activeOrfTag
        lastTag = activeOrfTag
        activeOrfTag += ORF_STRIDE
      else:
        regionList[chrom][posInRegion] = activeIntergenicTag
        lastTag = activeIntergenicTag
        activeIntergenicTag += INTERGENIC_STRIDE
      
      # Consistency check for next tags
      if activeOrfTag < lastTag:
        activeOrfTag += ORF_STRIDE
      if activeIntergenicTag < lastTag:
        activeIntergenicTag += INTERGENIC_STRIDE
      
      # Update lastPos
      lastPos = np.max(posInRegion) + 1
      
      # Clear region in orfRegionList
      orfRegionList[chrom][posInRegion] = 0
      
      # Print information if requested
      if verbose:
        print >> sys.stderr, "Original region ID\t=\t%d" % regionId
        print >> sys.stderr, ("New region ID\t=\t%d" %
                              regionList[chrom][posInRegion][0] )
        print >> sys.stderr, "Next position\t=\t%d" % lastPos

  return regionList

def segmentGenome(infoFile, readsFile, outFile, minLength, sep='\t',
                  normalize=True, verbose=False):
  # Read data from infoFile in nice format
  infoReader = csv.DictReader(infoFile, delimiter=sep)
  
  geneInfoFields = infoReader.fieldnames
  geneInfoList = []
  for line in infoReader:
    geneInfoList.append(line)
    
  # Combine overlapping ORFs into ORF regions
  initialRegionList = joinOverlappingOrfs(geneInfoList, geneInfoFields)
  initialRegionIds = [np.unique(x) for x in initialRegionList]
  
  if verbose:
    print >> sys.stderr, "Initial regions done"  
  
  # Setup uniform coverage list; retain if coverage information not available
  coverageList = []
  for l in CHROM_LENGTHS:
    coverageList.append(np.ones(l))
    
  # Get reads information if available
  if readsFile is not None:
    # Get the reads information
    readsList = []
    for line in readsFile:
      readsList.append( np.fromstring( line, sep=',' ) )
    
    # Calculate coverage for each identified region
    for chrom in xrange(len(CHROM_LENGTHS)):
      coverageList[chrom] = calcCoverage(initialRegionList[chrom],
                         readsList[chrom],
                         initialRegionIds[chrom])
  
  # Merge short regions until length constraint is met
  mergedRegionList = []
  for chrom in xrange(len(CHROM_LENGTHS)):
    mergedRegionList.append(mergeRegions(initialRegionList[chrom],
                                         coverageList[chrom],
                                         regionIds=initialRegionIds[chrom],
                                         minLength=minLength,
                                         normalize=normalize, verbose=verbose))
    if verbose:
      print >> sys.stderr, "Chromosome %02d merge done" % chrom 
  
  # Output region definitions
  for regions in mergedRegionList:
    np.savetxt(outFile, regions[np.newaxis,:], "%d", ' ')
  
  return 0


