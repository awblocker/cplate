import os
import sys

import numpy as np

import io

#==============================================================================
# General-purpose MCMC diagnostic and summarization functions
#==============================================================================

def effective_sample_sizes(**kwargs):
    '''
    Estimate effective sample size for each input using AR(1) approximation.
    Each input should be a 1- or 2-dimensional ndarray. 2-dimensional inputs
    should have one variable per column, one iteration per row.
    
    Parameters
    ----------
    
    Returns
    -------
        - If only one array of draws is provided, a single array containing the
          effective sample size(s) for those variables.
        - If multiple arrays are provided, a dictionary with keys identical to
          those provided as parameters and one array per input containing
          effective sample size(s).
    
    '''
    # Ensure that at least one input was provided
    if len(kwargs) < 1:
        return ValueError('Must provide at least one array of draws.')
    
    # Allocate empty dictionary for results
    ess = {}
    
    # Iterate over arrays of draws
    for var, draws in kwargs.iteritems():
        # Add dimension to 1d arrays
        if len(np.shape(draws)) < 2:
            draws = draws[:,np.newaxis]
        
        # Demean the draws
        draws = draws - draws.mean(axis=0)
        
        # Compute lag-1 autocorrelation by column
        acf = np.mean(draws[1:]*draws[:-1], axis=0) / np.var(draws, axis=0)
    
        # Compute ess from ACF
        ess[var] = np.shape(draws)[0]*(1.-acf)/(1.+acf)
    
    if len(kwargs) > 1:
        return ess
    else:
        return ess[kwargs.keys()[0]]

def posterior_means(**kwargs):
    '''
    Estimate posterior means from inputs.
    Each input should be a 1- or 2-dimensional ndarray. 2-dimensional inputs
    should have one variable per column, one iteration per row.
    
    Parameters
    ----------
        - **kwargs
            Names and arrays of MCMC draws.
    
    Returns
    -------
        - If only one array of draws is provided, a single array containing the
          posterior mean estimate(s) for those variables.
        - If multiple arrays are provided, a dictionary with keys identical to
          those provided as parameters and one array per input containing
          posterior mean estimate(s).
    
    '''
    # Ensure that at least one input was provided
    if len(kwargs) < 1:
        return ValueError('Must provide at least one array of draws.')
    
    # Allocate empty dictionary for results
    means = {}
    
    # Iterate over arrays of draws
    for var, draws in kwargs.iteritems():
        # Add dimension to 1d arrays
        if len(np.shape(draws)) < 2:
            draws = draws[:,np.newaxis]
        
        # Estimate posterior means
        means[var] = np.mean(draws, 0)
    
    if len(kwargs) > 1:
        return means
    else:
        return means[kwargs.keys()[0]]

def posterior_variances(**kwargs):
    '''
    Estimate posterior variances from inputs.
    Each input should be a 1- or 2-dimensional ndarray. 2-dimensional inputs
    should have one variable per column, one iteration per row.
    
    Parameters
    ----------
        - **kwargs
            Names and arrays of MCMC draws.
    
    Returns
    -------
        - If only one array of draws is provided, a single array containing the
          posterior variance estimate(s) for those variables.
        - If multiple arrays are provided, a dictionary with keys identical to
          those provided as parameters and one array per input containing
          posterior variance estimate(s).
    
    '''
    # Ensure that at least one input was provided
    if len(kwargs) < 1:
        return ValueError('Must provide at least one array of draws.')
    
    # Allocate empty dictionary for results
    variances = {}
    
    # Iterate over arrays of draws
    for var, draws in kwargs.iteritems():
        # Add dimension to 1d arrays
        if len(np.shape(draws)) < 2:
            draws = draws[:,np.newaxis]
        
        # Estimate posterior means
        variances[var] = np.var(draws, 0)
    
    if len(kwargs) > 1:
        return variances
    else:
        return variances[kwargs.keys()[0]]

def posterior_stderrors(**kwargs):
    '''
    Estimate posterior standard errors from inputs.
    Each input should be a 1- or 2-dimensional ndarray. 2-dimensional inputs
    should have one variable per column, one iteration per row.
    
    Parameters
    ----------
        - **kwargs
            Names and arrays of MCMC draws.
    
    Returns
    -------
        - If only one array of draws is provided, a single array containing the
          posterior standard error estimate(s) for those variables.
        - If multiple arrays are provided, a dictionary with keys identical to
          those provided as parameters and one array per input containing
          posterior standard error estimate(s).
    
    '''
    # Ensure that at least one input was provided
    if len(kwargs) < 1:
        return ValueError('Must provide at least one array of draws.')
    
    # Allocate empty dictionary for results
    stderrors = {}
    
    # Iterate over arrays of draws
    for var, draws in kwargs.iteritems():
        # Add dimension to 1d arrays
        if len(np.shape(draws)) < 2:
            draws = draws[:,np.newaxis]
        
        # Estimate posterior means
        stderrors[var] = np.std(draws, 0)
    
    if len(kwargs) > 1:
        return stderrors
    else:
        return stderrors[kwargs.keys()[0]]


def find_maxima(x, boundary=True):
    '''
    Finds local maxima in sequence x.
    Defining local maxima simply by low-high-low triplets.
    Retains boundary cases as well (high-low at start & low-high at end).
    Returns a boolean array of the same size as x.
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
    maxima  = up*down
    return maxima

def local_relative_occupancy(theta_t, window_small, window_local):
    b_t = np.exp(theta_t)
    return (np.convolve(b_t, window_small, 'same') /
            np.convolve(b_t, window_local, 'same'))

def condense_detections(detections):
    x = detections.copy() + 0.
    n = np.ones_like(x)
    
    while np.any(np.diff(x) < 2):
        first = np.min(np.where(np.diff(x)<2)[0])
        x *= n
        x = np.r_[ x[:first], (x[first] + x[first+1]), x[first+2:]]
        n = np.r_[ n[:first], (n[first] + n[first+1]), n[first+2:]]
        x /= n

    return x, n

def spacing(theta_t, window_local, local_concentration=10.):
    local_rel_occupancy = np.exp(theta_t) / np.convolve(np.exp(theta_t),
                                                        window_local,
                                                        mode='same')
    calls = np.where(local_rel_occupancy >
                     1./np.sum(window_local)*local_concentration)[0]
    return np.diff(calls)

def summarise(cfg, chrom=1, null=False):
    '''
    Coordinate summarisation of MCMC results.
    
    Parameters
    ----------
        - cfg : dictionary
            Dictionary of parameters containing at least those relevant MCMC
            draw and summary output paths and parameters for summarization.
        - chrom : int
            Index of chromosome to analyze
        - null : bool
            Summarise null results?
    '''
    # Reference useful information in local namespace
    n_burnin    = cfg['mcmc_params']['n_burnin']
    scratch     = cfg['mcmc_summaries']['path_scratch']
    width_local = cfg['mcmc_summaries']['width_local']
    concentration_pm = cfg['mcmc_summaries']['concentration_pm']

    # Check for existence and writeability of scratch directory
    if os.access(scratch, os.F_OK):
        # It exists, check for read-write
        if not os.access(scratch, os.R_OK | os.W_OK):
            print >> sys.stderr, ("Error --- Cannot read and write to %s" %
                                  scratch)
            return 1
    else:
        # Otherwise, try to make the directory
        os.makedirs(scratch)

    # Extract results to scratch directory
    if null:
        pattern_results = cfg['mcmc_output']['out_pattern']
    else:
        pattern_results = cfg['mcmc_output']['null_out_pattern']
    pattern_results = pattern_results.strip()
    path_results = pattern_results.format(**cfg) % chrom

    with np.load(path_results) as f:
        f.extractall(scratch)
        names_npy = f.zip.namelist()

    # Load results of interest
    theta   = np.load(scratch + '/theta.npy')
    mu      = np.load(scratch + '/mu.npy')

    # Remove burnin
    if n_burnin > 0:
        mu = mu[n_burnin:]
        theta = theta[n_burnin:]

    # Compute effective sample sizes
    n_eff = effective_sample_sizes(theta=theta)

    # Estimate P(theta_i > mu)
    p_theta_gt_mu = np.mean( theta - mu.flatten() > 0, 1)

    # Compute local relative occupancy
    window_pm    = np.ones(1 + 2*concentration_pm)
    window_local = np.ones(width_local)
    local_occupancy_draws = np.apply_along_axis(local_relative_occupancy,
                                                1,
                                                theta,
                                                np.ones(1), window_local)
    baseline = (1 / np.convolve(np.ones_like(theta[0]), window_local, 'same'))
    p_local_concentration_exact = np.mean(local_occupancy_draws > baseline, 0)

    is_local_concentration_pm = np.apply_along_axis(np.convolve, 1,
                                                     local_occupancy_draws >
                                                     baseline,
                                                     window_pm, 'same')
    is_local_concentration_pm = np.minimum(1, is_local_concentration_pm)
    p_local_concentration_pm = np.mean(is_local_concentration_pm, 0)

    # Compute posterior means
    theta_postmean = np.mean(theta, 0)
    b_postmean = np.mean(np.exp(theta), 0)

    # Compute standard errors
    theta_se = np.std(theta, 0)
    b_se = np.std(np.exp(theta), 0)

    # Compute posterior medians
    theta_postmed = np.median(theta, 0)
    b_postmed = np.exp(theta_postmed)

    # Provide nicely-formatted table of output for analyses and plotting
    if null:
        pattern_summaries = cfg['mcmc_output']['summary_pattern']
    else:
        pattern_summaries = cfg['mcmc_output']['null_summary_pattern']
    pattern_summaries = pattern_summaries.strip()
    path_summaries = pattern_summaries.format(**cfg) % chrom

    summaries = np.rec.fromarrays([theta_postmean, theta_postmed, theta_se,
                                   b_postmean, b_postmed, b_se, n_eff,
                                   p_theta_gt_mu, p_local_concentration_exact,
                                   p_local_concentration_pm],
                                  names=('theta', 'theta_med', 'se_theta', 'b',
                                         'b_med', 'se_b', 'n_eff',
                                         'p_theta_gt_mu',
                                         'p_local_concentration_pm0',
                                         'p_local_concentration_pm%d' %
                                         concentration_pm))
    io.write_recarray_to_file(fname=path_summaries, data=summaries,
                              header=True, sep=' ')
    # Clean-up scratch directory
    for name in names_npy:
        os.remove(scratch + '/' + name)
 
    return 0

