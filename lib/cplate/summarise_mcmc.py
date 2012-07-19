import gc
import os
import sys

import numpy as np
from scipy import stats

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

def find_maxima(x, boundary=False):
    '''
    Finds local maxima in sequence x, defining local maxima simply by
    low-high-low triplets.

    Parameters
    ----------
    - x : ndarray
        Sequence of values to search for local maxima
    - boundary : bool
        If True, include boundaries as possible maxima

    Returns
    -------
    - maxima : ndarray
        Boolean array of the same size as x with local maxima True

    '''
    # Intialization
    up, down    = np.ones((2, x.size), dtype=int)
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

def local_relative_occupancy(b_t, window_small, window_local):
    '''
    Compute local relative occupancy from vector of coefficients.

    Parameters
    ----------
    - b_t : ndarray
        Array of coefficients from a single draw
    - window_small : ndarray
        Array containing small window for local relative occupancy
    - window_local : ndarray
        Array containing larger window for local relative occupancy

    Returns
    -------
    - l : ndarray
        Array of same size as b_t containing local relative occupancies

    '''
    return (np.convolve(b_t, window_small, 'same') /
            np.convolve(b_t, window_local, 'same'))

def condense_detections(detections):
    '''
    Condense adjacent detections (from smoothed local occupancy) into centers
    and number of adjacent detections.

    Parameters
    ----------
    - detections : ndarray
        1d array of detected positions

    Returns
    -------
    - detections : ndarray
        1d array of detected centers
    - n : integer ndarray
        Number of detections per center
    '''
    x = detections.copy() + 0.
    n = np.ones_like(x)

    while np.any(np.diff(x) < 2):
        first = np.min(np.where(np.diff(x)<2)[0])
        x *= n
        x = np.r_[ x[:first], (x[first] + x[first+1]), x[first+2:]]
        n = np.r_[ n[:first], (n[first] + n[first+1]), n[first+2:]]
        x /= n

    return x, n

def greedy_maxima_search(x, min_spacing=100, remove_boundary=1, verbose=0):
    '''
    Greedily search for local maxima in sequence subject to minimum spacing
    constraint.

    Parameters
    ----------
    - x : ndarray
        1d sequence of values to search for local maxima
    - min_spacing : int
        Minimum spacing of positions. Greedy search continues until this
        constraint is met.
    - remove_boundary : int
        Length of region to exclude at each end of the sequence. 
    - verbose : int
        Level of verbosity in output

    Returns
    -------
    - out : ndarray
        Integer array of same shape as x containing ones at positions found in
        greedy search and zeros everywhere else.
    '''
    # Find local maxima in sequence; need indices of maxima, not binary
    # indicators
    positions       = np.where(find_maxima(x))[0]

    if remove_boundary > 0:
        # Exclude boundary positions
        positions = positions[positions>=remove_boundary]
        positions = positions[positions<x.size-remove_boundary]

    # Get spacing
    spacing         = np.diff(positions)

    # Check for bad overlaps
    while spacing.size > 0 and spacing.min() < min_spacing:
        # Save positions from previous iterations
        positions_last = positions.copy()

        # Find bad positions
        bad = np.where(spacing < min_spacing)[0]

        # Find first bad position
        first_bad    = np.min(bad)

        # Find which positions overlap with given position

        # First, get where overlaps below threshold are located
        good    = np.where(spacing >= min_spacing)[0]

        # Get number of positions from top bad one to good ones
        dist    = first_bad - good

        # Find limits of bad cluster
        if np.any(dist<0):
            last_in_cluster   = good[dist<0][np.argmax(dist[dist<0])]
            last_in_cluster   = min(last_in_cluster+1, spacing.size+1)
        else:
            last_in_cluster   = spacing.size+1

        if np.any(dist>0):
            first_in_cluster  = good[dist>0][np.argmin(dist[dist>0])]
            first_in_cluster  = max(0,first_in_cluster+1)
        else:
            first_in_cluster  = 0

        # Check coefficients of positions in cluster for maximum
        top_in_cluster    = np.argmax(x[positions[first_in_cluster:
                                                  last_in_cluster]])
        top_in_cluster    = first_in_cluster + top_in_cluster

        # Handle non-uniqueness
        top_in_cluster    = np.min(top_in_cluster)

        # Eliminate bad neighbors from positions
        keep    = np.ones(positions.size, dtype=bool)

        if top_in_cluster > 0:
            space = (positions[top_in_cluster] - positions[top_in_cluster-1])
            if space < min_spacing:
                keep[top_in_cluster-1] = False

        if top_in_cluster < positions.size-1:
            space = (positions[top_in_cluster+1] - positions[top_in_cluster])
            if space < min_spacing:
                keep[top_in_cluster+1] = False

        positions       = positions[keep]

        if positions.size == positions_last.size:
            print >> sys.stderr, 'Error --- greedy search is stuck'
            print >> sys.stderr, positions, spacing
            break

        if verbose:
            print >> sys.stderr, positions, spacing

        # Update spacing
        spacing         = np.diff(positions)

    out = np.zeros(np.size(x), dtype=np.int)
    out[positions] = 1
    return out

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

    Returns
    -------
    - status : int
        Integer status for summarisation. 0 for success, > 0 for failure.
    '''
    # Reference useful information in local namespace
    n_burnin    = cfg['mcmc_params']['n_burnin']
    scratch     = cfg['mcmc_summaries']['path_scratch']
    width_local = cfg['mcmc_summaries']['width_local']
    concentration_pm = cfg['mcmc_summaries']['concentration_pm']
    p_detect    = cfg['mcmc_summaries']['p_detect']
    bp_per_nucleosome = cfg['mcmc_summaries']['bp_per_nucleosome']
    

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
        pattern_results = cfg['mcmc_output']['null_out_pattern']
    else:
        pattern_results = cfg['mcmc_output']['out_pattern']
    pattern_results = pattern_results.strip()
    path_results = pattern_results.format(**cfg) % chrom

    f = np.load(path_results)
    f.zip.extractall(scratch)
    names_npy = f.zip.namelist()
    f.close()

    # Load results of interest
    theta   = np.load(scratch + '/theta.npy', mmap_mode='r')
    mu      = np.load(scratch + '/mu.npy')

    # Load region type information
    with open(cfg['data']['regions_path'].format(**cfg), 'rb') as f:
        lines_read = 0
        for line in f:
            lines_read += 1
            if lines_read == chrom:
                region_types = np.fromstring(line.strip(), sep=' ', dtype=int)
                break

    # Remove burnin
    if n_burnin > 0:
        mu = mu[n_burnin:]
        theta = theta[n_burnin:]

    # Compute effective sample sizes
    n_eff = effective_sample_sizes(theta=theta)

    # Estimate P(theta_i > mu)
    p_theta_gt_mu = np.mean(theta - mu[:,region_types] > 0, 0)

    # Compute probability of single-basepair local concentrations
    window_local = np.ones(width_local)
    baseline = (1. / np.convolve(np.ones_like(theta[0]), window_local, 'same'))
    #
    p_local_concentration_exact = np.zeros(theta.shape[1], dtype=np.float)
    #
    for t in xrange(theta.shape[0]):
        local_occupancy_draw = local_relative_occupancy(np.exp(theta[t]),
                                                        np.ones(1),
                                                        window_local)
        p_local_concentration_exact *= t/(t+1.)
        p_local_concentration_exact += (local_occupancy_draw > baseline)/(t+1.)

    # Clean-up
    gc.collect()

    # Posterior probability of +/-(concentration_pm) concentrations
    window_pm    = np.ones(1 + 2*concentration_pm)
    baseline_smoothed = (np.convolve(np.ones_like(theta[0]), window_pm, 'same')
                         / np.convolve(np.ones_like(theta[0]), window_local,
                                       'same'))
    #
    p_local_concentration_pm = np.zeros(theta.shape[1], dtype=np.float)
    #
    for t in xrange(theta.shape[0]):
        local_occupancy_smoothed = local_relative_occupancy(np.exp(theta[t]),
                                                            window_pm,
                                                            window_local)
        p_local_concentration_pm *= t/(t+1.)
        p_local_concentration_pm += ((local_occupancy_smoothed >
                                      baseline_smoothed)/(t+1.))

    # Clean-up
    gc.collect()
    
    # Posterior quantiles for global concentrations
    baseline_global = (np.sum(np.exp(theta), 1) / theta.shape[1]
                        * bp_per_nucleosome)
    #
    q_global_concentration_exact = np.zeros(theta.shape[1], dtype=np.float)
    mean_global_concentration_exact = np.zeros(theta.shape[1], dtype=np.float)
    q_global_concentration_pm = np.zeros(theta.shape[1], dtype=np.float)
    mean_global_concentration_pm = np.zeros(theta.shape[1], dtype=np.float)
    for bp in xrange(theta.shape[1]):
        # Single-basepair first; it's the easiest
        prop = np.exp(theta[:,bp])/baseline_global
        mean_global_concentration_exact[bp] = np.mean(prop)
        q_global_concentration_exact[bp] = stats.mstats.mquantiles(prop,
                                                                   1.-p_detect)

        # Now, +/-(concentration_pm) basepairs
        w = slice(max(0,bp-concentration_pm), min(bp+concentration_pm,
                                                  theta.shape[1]))
        prop = np.sum(np.exp(theta[:,w]), 1)/baseline_global/(w.stop-w.start)
        mean_global_concentration_pm[bp] = np.mean(prop)
        q_global_concentration_pm[bp] =  stats.mstats.mquantiles(prop,
                                                                 1.-p_detect)
    
    # Compute posterior means
    theta_postmean = np.mean(theta, 0)
    b_postmean = np.mean(np.exp(theta), 0)

    # Compute standard errors
    theta_se = np.std(theta, 0)
    b_se = np.std(np.exp(theta), 0)

    # Compute posterior medians
    theta_postmed = np.median(theta, 0)
    b_postmed = np.exp(theta_postmed)

    # Provide nicely-formatted delimited output for analyses and plotting
    if null:
        pattern_summaries = cfg['mcmc_output']['null_summary_pattern']
    else:
        pattern_summaries = cfg['mcmc_output']['summary_pattern']
    pattern_summaries = pattern_summaries.strip()
    path_summaries = pattern_summaries.format(**cfg) % chrom

    summaries = np.rec.fromarrays([theta_postmean, theta_postmed, theta_se,
                                   b_postmean, b_postmed, b_se, n_eff,
                                   p_theta_gt_mu, p_local_concentration_exact,
                                   p_local_concentration_pm,
                                   q_global_concentration_exact,
                                   mean_global_concentration_exact,
                                   q_global_concentration_pm,
                                   mean_global_concentration_pm],
                                  names=('theta', 'theta_med', 'se_theta', 'b',
                                         'b_med', 'se_b', 'n_eff',
                                         'p_theta_gt_mu',
                                         'p_local_concentration_pm0',
                                         'p_local_concentration_pm%d' %
                                         concentration_pm,
                                         'q_global_concentration_pm0',
                                         'mean_global_concentration_pm0',
                                         'q_global_concentration_pm%d' %
                                         concentration_pm,
                                         'mean_global_concentration_pm%d' %
                                         concentration_pm,))
    io.write_recarray_to_file(fname=path_summaries, data=summaries,
                              header=True, sep=' ')

    # Run detection, if requested
    if p_detect is not None and not null:
        # Find detected positions
        detected = np.where(p_local_concentration_pm > p_detect)[0]

        # Condense regions
        detected, n = condense_detections(detected)

        # Write detections to text file
        pattern_detections = cfg['mcmc_output']['detections_pattern']
        pattern_detections = pattern_detections.strip()
        path_detections = pattern_detections.format(**cfg) % chrom

        detections = np.rec.fromarrays([detected, n],
                                       names=('pos', 'n'))
        io.write_recarray_to_file(fname=path_detections,
                                  data=detections, header=True,
                                  sep=' ')

    # Clean-up scratch directory
    for name in names_npy:
        os.remove(scratch + '/' + name)

    return 0

