import sys
import multiprocessing
import itertools

import numpy as np
from scipy import optimize
from scipy import stats

def find_maxima(x, boundary=True):
    '''
    Finds local maxima in sequence x.
    Defining local maxima simply by low-high-low triplets.
    Retains boundary cases as well (high-low at start & low-high at end).
    Returns array of size # of maxima.
    '''
    # Intialization
    up, down    = np.ones((2, x.size))
    # Central cases
    up[1:-1]    = (x[1:-1]>x[:-2])
    down[1:-1]  = (x[2:]<x[1:-1])
    
    if boundary:
        # Boundary cases
        down[0]     = (x[1]<x[0])
        up[-1]      = (x[-1]>x[-2])
    # Logical and
    maxima  = np.where(up*down)[0]
    return maxima

def estimate_fdr(t, alpha, null, nonnull):
    '''
    Crude estimate of FDR based on empirical probability of exceeding a
    threshold.
    '''
    fdr = np.mean(null>=t)/np.mean(nonnull>=t)
    if not np.isfinite(fdr): fdr = 0
    return fdr - alpha

def estimate_fdr_direct(t, alpha, p_nonnull, lam=0.8):
    '''
    Based on Storey 2002, "A Direct Approach to False Discovery Rates".
    Less conservative than BH, but it is also less robust to dependence.
    '''
    m       = p_nonnull.size
    pi0Hat  = np.minimum(1.0, np.mean(p_nonnull > lam)/(1-lam))
    p_hat    = np.maximum( np.mean(p_nonnull<t), 1.0/m )
    fdr_hat  = pi0Hat * t / p_hat # / ( 1 - (1-t)**m )
    
    return fdr_hat - alpha

def estimate_threshold_bh(alpha, p_nonnull):
    '''
    Estimate of detection threshold based on reversing standard BH procedure.
    '''
    # So p-values
    p_sorted = np.sort(p_nonnull)
    
    # Detect until p_(k) > k/n * alpha
    n = np.size(p_sorted)
    k = np.arange(1, n+1)

    n_detected = np.max(np.where(p_sorted * k/n <= alpha)[0])

    return np.mean(p_sorted[n_detected:n_detected+2])
    
def integrate_interval(arg_tuple):
    '''
    Utility function to integrate KDE over a single interval
    Needed to use multiprocessing.Pool.map; that provides the odd argument
    syntax
    '''
    kde, lower, upper = arg_tuple
    
    return kde.integrate_box( lower, upper )

def estimate_pvalues( null, nonnull, pool=None, n_proc=None ):
    '''
    Obtain p-values for nonnull sample based on null sample.
    Using kernel density estimator to obtain these.
    Gaussian kernel, so data should be transformed before passing to this
    function
    '''
    # Setup KDE object
    kde     = stats.gaussian_kde( null )

    # Get bounds for integration
    upper   = np.maximum( nonnull.max(), null.max() ) + 4*np.std(null)

    # Iteratively numerically integrate over sorted nonnull values
    # Integrating over successive small areas between observations
    ccdf    = np.zeros(nonnull.size)
    ind     = np.argsort(nonnull)[::-1]

    ccdf[0] = kde.integrate_box( nonnull[ind[0]], upper )
    if pool is None:
        # Process intervals sequentially
        for ii in xrange(1,ccdf.size):
            ccdf[ii]    = kde.integrate_box( nonnull[ind[ii]],
                                             nonnull[ind[ii-1]] )
    else:
        # Use multiple processes
        parallel_iterator = itertools.izip(itertools.repeat(kde),
                                          nonnull[ind[1:]],
                                          nonnull[ind[:-1]])
        p_list       = pool.map(integrate_interval, parallel_iterator,
                               chunksize=nonnull.size/n_proc)
        ccdf[1:]    = p_list

    ccdf    = np.cumsum(ccdf)

    # Rearrange p-values into original ordering
    p       = np.empty(nonnull.size)
    p[ind]  = ccdf

    return p
    
def process_region(arg_tuple):
    '''
    Utility function to process a single region given region, null, and nonnull
    Needed to use multiprocessing.Pool.map; that provides the odd argument
    syntax
    '''
    # Unpack argument tuple
    region, null, nonnull, alpha = arg_tuple    
    
    null_region      = null[region]
    nonnull_region   = nonnull[region]
    
    # Transform data
    log_null     = np.log(null_region)
    log_nonnull  = np.log(nonnull_region)
    
    # Calculate p-values
    p           = estimate_pvalues( log_null, log_nonnull )
    
    # Get threshold
    thresh_p     = optimize.bisect(estimate_fdr_direct, 0, 1,
                                  args=(alpha, p))
    thresh_ind   = np.searchsorted( np.sort(p), thresh_p )
    if thresh_ind > 0:
        thresh_coef  = np.sort(log_nonnull)[::-1][thresh_ind-1:thresh_ind+1]
        thresh_coef  = thresh_coef.mean()
    else:
        thresh_coef  = log_nonnull.max() + np.log(2)
    thresh      = np.exp(thresh_coef)
    
    return thresh

def get_fdr_threshold_estimate(null, nonnull, region_list, alpha, maxima=False,
                               n_proc=None, verbose=False):
    '''
    Calculate FDR-based detection threshold from given data for FDR=alpha.

    Operates across regions and returns numpy array with threshold for each
    region.

    Uses direct estimation technique for FDR as in Storey (2002).

    Estimates p-values using Gaussian KDE on null sample.
    '''
    # Setup pool if needed
    if n_proc is not None:
        pool    = multiprocessing.Pool(processes=n_proc)
    else:
        pool    = None
    
    thresh_list = []
    for region in region_list:
        if verbose: print >> sys.stderr, "Limits\t=\t%d\t%d" % (region[0],
                                                                region[-1])
        
        null_region = null[region]
        if maxima:
            subset = find_maxima(null_region)
            if subset.size > 1:
                null_region = null_region[subset]
            else:
                if verbose:
                    err_msg = "Region %d had only %d local maxima" % (region,
                                                                    subset.size)
                    print >> sys.stderr, err_msg
        null_region.sort()
        
        nonnull_region = nonnull[region]
        nonnull_max    = nonnull.max()
        if maxima:
            nonnull_region = nonnull_region[find_maxima(nonnull_region)]
        nonnull_region.sort()
        
        # Handle case of no local maxima
        if nonnull_region.size < 1:
            thresh_list.append(nonnull_max*2)
            continue

        # Transform data
        log_null     = np.log(null_region)
        log_nonnull  = np.log(nonnull_region)
        
        # Calculate p-values
        try:
            p   = estimate_pvalues( log_null, log_nonnull, pool, n_proc )
        except:
            # Empirical CDF fall-back for singular cases
            p   = np.searchsorted( log_null, log_nonnull ).astype(np.float)
            p   /= log_null.size
        
        # Check for pathological case
        # Both cases have same sign -> give up
        if estimate_fdr_direct(0.0,alpha,p)*estimate_fdr_direct(1.0,alpha,p) > 0:
            thresh_p     = p.min()*(1-np.sqrt(np.spacing(1)))
            if verbose:
                print >> sys.stderr, "Pathological case"
        else:
            # Otherwise, get threshold via bisection
            thresh_p     = optimize.bisect(estimate_fdr_direct, 0.0, 1.0,
                                          args=(alpha, p))
        thresh_ind   = np.searchsorted( np.sort(p), thresh_p )
        if thresh_ind > 0:
            thresh_coef  = np.sort(log_nonnull)[::-1][thresh_ind-1:thresh_ind+1]
            thresh_coef  = thresh_coef.mean()
        else:
            thresh_coef  = log_nonnull.max() + np.log(2)
        thresh      = np.exp(thresh_coef)
        
        thresh_list.append(thresh)
        
    return np.array(thresh_list)

                           
def get_fdr_threshold(null, nonnull, region_list, alpha, maxima=False):
    '''
    Calculate FDR-based detection threshold from given data for FDR=alpha.

    Uses crude empirical-distribution based estimate of FDR.
    
    Operates across regions and returns numpy array with threshold for each
    region.
    '''
    thresh_list = []
    for region in region_list:
        null_region = null[region]
        if maxima:
            null_region = null_region[find_maxima(null_region)]
        null_region.sort()
        
        nonnull_region = nonnull[region]
        if maxima:
            nonnull_region = nonnull_region[find_maxima(nonnull_region)]
        nonnull_region.sort()
        
        thresh = optimize.bisect(estimate_fdr,
                                 min(null_region.min(), nonnull_region.min()),
                                 max(null_region.max(), nonnull_region.max()),
                                 args=(alpha, null_region, nonnull_region))
        thresh_list.append(thresh)
        
    return np.array(thresh_list)

def get_fdr_threshold_bh(null, nonnull, region_list, alpha, maxima=False,
                         n_proc=None, verbose=False):
    '''
    Calculate FDR-based detection threshold from given data for FDR=alpha.

    Operates across regions and returns numpy array with threshold for each
    region.

    Uses Benjamini-Hochberg technique.

    Estimates p-values using Gaussian KDE on null sample.
    '''
    # Setup pool if needed
    if n_proc is not None:
        pool    = multiprocessing.Pool(processes=n_proc)
    else:
        pool    = None
    
    thresh_list = []
    for region in region_list:
        if verbose: print >> sys.stderr, "Limits\t=\t%d\t%d" % (region[0],
                                                                region[-1])
        
        null_region = null[region]
        if maxima:
            subset = find_maxima(null_region)
            if subset.size > 1:
                null_region = null_region[subset]
            else:
                if verbose:
                    err_msg = ("Region %d had only %d local maxima" %
                               (region, subset.size))
                    print >> sys.stderr, err_msg
        null_region.sort()
        
        nonnull_region = nonnull[region]
        nonnull_max    = nonnull.max()
        if maxima:
            nonnull_region = nonnull_region[find_maxima(nonnull_region)]
        nonnull_region.sort()
        
        # Handle case of no local maxima
        if nonnull_region.size < 1:
            thresh_list.append(nonnull_max*2)
            continue

        # Transform data
        log_null     = np.log(null_region)
        log_nonnull  = np.log(nonnull_region)
        
        # Calculate p-values
        try:
            p   = estimate_pvalues( log_null, log_nonnull, pool, n_proc )
        except:
            # Empirical CDF fall-back for singular cases
            p   = np.searchsorted( log_null, log_nonnull ).astype(np.float)
            p   /= log_null.size
        
        # Get threshold based upon BH procedure
        thresh_p     = estimate_threshold_bh(p_nonnull=p, alpha=alpha)

        thresh_ind   = np.searchsorted( np.sort(p), thresh_p )
        if thresh_ind > 0:
            thresh_coef  = np.sort(log_nonnull)[::-1][thresh_ind-1:thresh_ind+1]
            thresh_coef  = thresh_coef.mean()
        else:
            thresh_coef  = log_nonnull.max() + np.log(2)
        thresh      = np.exp(thresh_coef)
        
        thresh_list.append(thresh)
        
    return np.array(thresh_list)
