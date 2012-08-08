import sys
import time
import bz2
import contextlib
import cPickle

import numpy as np
from scipy import sparse
from scikits.sparse import cholmod
from mpi4py import MPI

import lib_deconvolve_em as lib
import libio

# Set constants

# MPI constants
MPIROOT     = 0
# Tags for worker states
STOPTAG     = 0
SYNCTAG     = 1
WORKTAG     = 2

# Interval between M-steps
INTERVAL    = 4

def load_data(chrom, cfg, null=False):
    '''
    Load and setup all data for runs.

    Parameters
    ----------
        - chrom : int
            Index (starting from 1) of chromosome to extract.
        - cfg : dictionary
            Dictionary containing (at least) data section with paths
            to template, null, read data, and regions.
        - null : bool
            If null, read null data instead of actual reads.

    Returns
    -------
        Dictionary containing
        - chrom : int
            Index of chromosome extracted.
        - y : integer ndarray
            Counts of read centers per base pair.
        - region_types : integer ndarray
            Vector of region types by base pair.
        - region_list : list of integer ndarrays
            List of index vectors by region id.
        - region_sizes : integer ndarray
            Vector of region sizes by region id.
        - region_ids : integer ndarray
            Vector of distinct region ids.
    '''
    # Load template data
    template = np.loadtxt(cfg['data']['template_path'].format(**cfg))

    # Load chromosome-level read center counts
    if null:
        chrom_path = cfg['data']['null_path'].format(**cfg)
    else:
        chrom_path = cfg['data']['chrom_path'].format(**cfg)

    with open(chrom_path, 'r') as f:
        lines_read = 0
        for line in f:
            lines_read += 1
            if lines_read == chrom:
                reads = np.fromstring(line.strip(), sep=',')
                break

    # Load region type information
    with open(cfg['data']['regions_path'].format(**cfg), 'rb') as f:
        lines_read = 0
        for line in f:
            lines_read += 1
            if lines_read == chrom:
                region_types = np.fromstring(line.strip(), sep=' ', dtype=int)
                break

    # Get length of chromosome; important if regions and reads disagree
    chrom_length = min(region_types.size, reads.size)

    # Truncate region types to chromosome length
    region_types = region_types[:chrom_length]

    # Set region types to start at 0 for consistent array indexing
    region_types -= region_types.min()

    # Get unique region identifiers
    n_regions = region_types.max() + 1
    region_ids = np.unique(region_types)

    # Build map of regions by r
    region_list = [None]*n_regions
    region_sizes = np.ones(n_regions, dtype=np.int)
    for r in region_ids:
        region = np.where(region_types==r)[0]
        region_list[r] = slice(region.min(), region.max()+1)
        region_sizes[r] = region.size

    # Setup y variable
    y = reads[:chrom_length]

    # Build dictionary of data to return
    data = {'chrom' : chrom,
            'y' : y,
            'template' : template,
            'region_types' : region_types,
            'region_list' : region_list,
            'region_sizes' : region_sizes,
            'region_ids' : region_ids
            }
    return data

def initialize(data, cfg, rank=None, null=False):
    '''
    Initialize parameters across all nodes.

    Parameters
    ----------
        - data : dictionary
            Data as output from load_data.
        - cfg : dictionary
            Dictionary containing (at least) prior and estimation_params
            sections with appropriate entries.
        - rank : int
            If not None, rank of node to print in diagnostic output.

    Returns
    -------
        Dictionary of initial parameters containing
        - theta : ndarray
            Starting values for base-pair specific nucleosome occupancies.
        - mu : ndarray
            Starting values for log-mean (mu) parameters.
        - sigmasq : ndarray
            Starting values for log-variance (sigmasq) parameters.
    '''
    # Create references to frequently-accessed config information
    # Prior on 1 / sigmasq
    a0  = cfg['prior']['a0']
    b0  = cfg['prior']['b0']
    # Verbosity
    verbose = cfg['estimation_params']['verbose']

    # Create references to relevant data entries in local namespace
    y            = data['y']
    region_list  = data['region_list']
    region_ids   = data['region_ids']
    region_sizes = data['region_sizes']

    # Compute needed data properties
    n_regions = region_ids.size

    # Initialize nucleotide-level occupancies
    if cfg['mcmc_params']['initialize_theta_from_em']:
        # Load estimates from EM iterations
        if null:
            coef_pattern = cfg['estimation_output']['null_coef_pattern']
        else:
            coef_pattern = cfg['estimation_output']['coef_pattern']

        coef_pattern = coef_pattern.strip()
        coef_path = coef_pattern.format(**cfg) % data['chrom']

        theta = np.log(np.loadtxt(coef_path))
    else:
        theta = np.log(y+1.0)

    # Initialize
    if cfg['mcmc_params']['initialize_params_from_em']:
        # Load parameters from EM iterations
        if null:
            param_pattern = cfg['estimation_output']['null_param_pattern']
        else:
            param_pattern = cfg['estimation_output']['param_pattern']

        param_pattern = param_pattern.strip()
        param_path = param_pattern.format(**cfg) % data['chrom']

        param_dtype = [('mu', np.float),
                       ('sigmasq', np.float)]
        mu, sigmasq = np.loadtxt(param_path, skiprows=1, dtype=param_dtype,
                                 usecols=(1,2), unpack=True, ndmin=1)
    else:
        mu = np.zeros(n_regions)
        sigmasq = np.ones(n_regions)
        for r in region_ids:
            # Initialize mu and sigmasq with correct posterior draw
            region = region_list[r]

            # Draw sigmasq from marginal distribution
            shape_sigmasq   = region_sizes[r]/.2 + a0
            rate_sigmasq    = np.var(theta[region])*region_sizes[r]/2. + b0
            sigmasq[r]      = 1./np.random.gamma(shape=shape_sigmasq,
                                                 scale=1./rate_sigmasq)

            # Draw mu | sigmasq
            mean_mu = np.mean(theta[region])
            var_mu  = sigmasq[r] / region_sizes[r]
            mu[r]   = mean_mu + np.sqrt(var_mu)*np.random.randn(1)

    if verbose:
        print "Node %d initialization complete" % rank
        if verbose > 2: print mu, sigmasq

    # Build dictionary of initial params to return
    init = {'theta' : theta,
            'mu' : mu,
            'sigmasq' : sigmasq}
    return init

def master(comm, n_proc, data, init, cfg):
    '''
    Master node process for parallel MCMC. Coordinates draws, handles all
    region-level parameter draws, and collects results.

    Parameters
    ----------
        - comm : mpi4py.MPI.COMM
            Initialized MPI communicator.
        - n_proc : int
            Number of processes in communicator.
        - data : dictionary
            Data as output from load_data.
        - init : dictionary
            Initial parameter values as output from initialize.
        - cfg : dictionary
            Dictionary containing (at least) prior and estimation_params
            sections with appropriate entries.

    Returns
    -------
        Dictionary of results containing:
        - theta : ndarray
            Estimated values of base-pair specific nucleosome occupancies
        - vartheta : ndarray
            Approximate variance of log-occupancies conditional on (mu, sigmasq)
            by base-pair.
        - mu : ndarray
            MAP estimates of log-mean (mu) parameters.
        - sigmasq : ndarray
            MAP estimates of log-variance (sigmasq) parameters.
        - region_ids : integer ndarray
            Vector of distinct region ids.
    '''
    # Create references to frequently-accessed config information
    # Prior on mu - sigmasq / 2
    mu0 = cfg['prior']['mu0']
    k0  = cfg['prior']['k0']
    # Prior on 1 / sigmasq
    a0  = cfg['prior']['a0']
    b0  = cfg['prior']['b0']
    # Iteration limits
    max_iter = cfg['mcmc_params']['mcmc_iterations']
    # Verbosity
    verbose = cfg['estimation_params']['verbose']
    timing = cfg['estimation_params']['timing']

    # Compute derived quantities from config information
    sigmasq0 = b0 / a0
    adapt_prior = (mu0 is None)

    # Create references to relevant data entries in local scope
    y           = data['y']
    region_list  = data['region_list']
    region_sizes = data['region_sizes']
    region_ids   = data['region_ids']
    # Template and derived properties
    template = data['template']
    w = template.size/2 + 1
    
    # Compute needed data properties
    chrom_length = y.size
    n_regions = region_ids.size

    # Initialize data structures for draws
    theta       = np.empty((max_iter, chrom_length))
    theta[0]    = init['theta']
    #
    mu          = np.empty((max_iter, n_regions))
    mu[0]       = init['mu']
    #
    sigmasq     = np.empty((max_iter, n_regions))
    sigmasq[0]  = init['sigmasq']

    # Compute block width for parallel theta draws
    n_workers = n_proc - 1
    if cfg['estimation_params']['block_width'] is None:
        block_width = chrom_length / n_workers
    else:
        block_width = cfg['estimation_params']['block_width']

    # Compute maximum size of theta slices to send
    theta_buf_size = block_width + 2*w

    # Setup prior means
    prior_mean = np.zeros(n_regions)
    if adapt_prior:
        # Adapt prior means if requested
        # Get coverage by region
        coverage = np.zeros(n_regions)
        for i in region_ids:
            coverage[i] = np.mean(y[region_list[i]])

        # Translate to prior means
        prior_mean[coverage>0] = np.log(coverage[coverage>0]) - sigmasq0 / 2.0
    else:
        prior_mean += mu0

    # Initialize information for MCMC sampler
    ret_val = np.empty(block_width)
    status = MPI.Status()

    # Start timing, if requested
    if timing:
        tme = time.clock()

    assigned = np.zeros(n_workers, dtype=np.int)

    # Setup blocks for worker nodes
    # This is the scan algorithm with a 2-iteration cycle.
    # It is designed to ensure consistent sampling coverage of the chromosome.
    start_vec = [np.arange(0, chrom_length, block_width, dtype=np.int),
                 np.arange(block_width/2, chrom_length, block_width,
                           dtype=np.int)]
    start_vec = np.concatenate(start_vec)
    theta_send_buf = np.empty(theta_buf_size, dtype=np.float)

    # Initialize acceptance statistics
    n_prop_per_iteration = np.zeros(chrom_length, dtype=np.int)
    for start in start_vec:
        end = min(start + block_width, chrom_length)
        n_prop_per_iteration[start:end] += 1

    n_accepted = np.zeros(chrom_length, dtype=np.int)
    n_accepted_tm1 = np.zeros_like(n_accepted)

    if verbose > 1:
        # Print starting values for parameters
        print mu[0], sigmasq[0]
        # Initialize rough block identifiers
        block_ids = np.arange(chrom_length, dtype=np.int) / block_width

    for t in xrange(1, max_iter):
        # (1) Distributed draw of theta | mu, sigmasq, y on workers.

        # First, synchronize parameters across all workers

        # Coordinate the workers into the synchronization state
        for k in range(1, n_workers+1):
            comm.Send([np.array(0, dtype=np.int), MPI.INT], dest=k, tag=SYNCTAG)

        # Broadcast theta and parameter values to all workers
        comm.Bcast(mu[t-1], root=MPIROOT)
        comm.Bcast(sigmasq[t-1], root=MPIROOT)

        # Initialize local theta for current iteration
        theta[t] = theta[t-1]

        # Dispatch jobs to workers until completed
        n_jobs       = start_vec.size
        n_started    = 0
        n_completed  = 0

        # Randomize block ordering
        np.random.shuffle(start_vec)

        # Send first batch of jobs
        for worker in range(1,min(n_workers, start_vec.size)+1):
            # Setup block to send
            end = min(chrom_length, start_vec[n_started] + block_width)
            block = slice(max(start_vec[n_started] - w, 0),
                          min(end+w, chrom_length))
            theta_send_buf[:block.stop-block.start] = theta[t,block]
            
            # Tell worker to update slice of theta and execute theta draw
            comm.Send(np.array(start_vec[n_started], dtype=np.int),
                      dest=worker, tag=WORKTAG)
            
            # Update theta slice on worker node
            comm.Send(theta_send_buf, dest=worker, tag=MPIROOT)

            # Update job counter and assignment array
            assigned[worker-1] = start_vec[n_started]
            n_started += 1

        # Collect results from workers and dispatch additional jobs until
        # complete
        while n_completed < n_jobs:
            # Collect any completed results
            comm.Recv(ret_val, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                      status=status)
            n_completed += 1
            
            # Slot updated slice of theta into proper place
            worker = status.Get_source()
            start = assigned[worker-1]
            end = min(start+block_width, chrom_length)
            n_accepted[start:end] += status.Get_tag()
            theta[t,start:end] = ret_val[:end-start]

            # If all jobs are not complete, update theta on the just-finished
            # worker and send another job.
            if n_started < n_jobs:
                # Setup block to send
                end = min(chrom_length, start_vec[n_started] + block_width)
                block = slice(max(start_vec[n_started] - w, 0),
                              min(end+w, chrom_length))
                theta_send_buf[:block.stop-block.start] = theta[t,block]

                # Tell worker to update slice of theta and execute theta draw
                comm.Send(np.array(start_vec[n_started], dtype=np.int),
                          dest=worker, tag=WORKTAG)
                
                # Update theta slice on worker node
                comm.Send(theta_send_buf, dest=worker, tag=MPIROOT)

                # Update job counter and assignment array
                assigned[worker-1] = start_vec[n_started]
                n_started += 1

        # (2) Draw region-level parameters given occupancies

        for r in region_ids:
            region = region_list[r]

            # Draw sigmasq from marginal distribution
            shape_sigmasq   = region_sizes[r]/2. + a0
            rate_sigmasq    = (np.var(theta[t,region])*region_sizes[r]/2. + b0
                               + k0*region_sizes[r]/2./(1.+k0)*
                               (np.mean(theta[t,region]) - prior_mean[r])**2)
            sigmasq[t,r]    = rate_sigmasq/np.random.gamma(shape=shape_sigmasq,
                                                           scale=1.)

            # Draw mu | sigmasq
            mean_mu = (np.mean(theta[t,region]) + prior_mean[r]*k0)/(1.0 + k0)
            var_mu  = sigmasq[t,r] / (1. + k0) / region_sizes[r]
            mu[t,r] = mean_mu + np.sqrt(var_mu)*np.random.randn(1)

        if verbose:
            if timing: print >> sys.stderr, ( "%d:\tIteration time: %s" %
                                              (t, time.clock() - tme) )
            if verbose > 1:
                prop_accepted = (n_accepted-n_accepted_tm1)/n_prop_per_iteration
                print np.mean(prop_accepted)
                print (np.bincount(block_ids, weights=prop_accepted) /
                       np.bincount(block_ids))
                print mu[t]
                print sigmasq[t]
                n_accepted_tm1 = n_accepted.copy()

        if timing:
            tme = time.clock()


    # Halt all workers
    for k in range(1,n_proc):
        comm.send((None,None), dest=k, tag=STOPTAG)

    # Return results
    out = {'theta' : theta,
           'mu' : mu,
           'sigmasq' : sigmasq,
           'region_ids' : region_ids,
           'prop_accepted' : n_accepted/(max_iter - 1.)/n_prop_per_iteration}
    return out

def rmh_worker_theta(comm, block_width, start, y, template, theta, mu, sigmasq,
                     region_types, prop_df=5.):
    # Compute needed data properties
    chrom_length = y.size
    w = template.size/2 + 1

    # Calculate subset of data to work on
    end = min(chrom_length, start + block_width)
    block = slice(max(start-w, 0), min(end+w, chrom_length))
    size_block  = block.stop - block.start

    subset = slice(w*(start!=0)+start-block.start,
                   size_block-w*(end!=chrom_length) - (block.stop-end))
    size_subset = subset.stop - subset.start

    original = slice(start-block.start, size_block - (block.stop-end))

    theta_block     = theta[block]
    theta_subset    = theta_block[subset]

    # Setup initial return value
    ret_val = np.empty(block_width)

    # Run optimization to obtain conditional posterior mode
    theta_hat = lib.deconvolve(lib.loglik_convolve,
                               lib.dloglik_convolve,
                               y[block], region_types[block], template,
                               mu, sigmasq,
                               subset=subset, theta0=theta_block,
                               log=True,
                               messages=0)[0]

    # Compute (sparse) conditional observed information
    X = sparse.spdiags((np.ones((template.size,size_block)).T *
                        template).T, diags=range(-w+1, w),
                        m=size_block, n=size_block, format='csr')

    info = lib.ddloglik(theta=theta_hat,
                        theta0=theta_block,
                        X=X, Xt=X, y=y[block],
                        mu=mu, sigmasq=sigmasq,
                        region_types=region_types[block],
                        subset=subset, log=True)
    info = info[subset,:]
    info = info.tocsc()
    info = info[:,subset]

    # Propose from multivariate t distribution
    try:
        info_factor = cholmod.cholesky(info)
    except:
        # Always reject for these cases
        accept = 0
        ret_val[:end-start] = theta_block[original]

        # Transmit result
        comm.Send(ret_val, dest=MPIROOT, tag=accept)

        return

    L, D = info_factor.L_D()
    D = D.diagonal()
    #
    z = np.random.standard_t(df=prop_df, size=size_subset)
    #
    theta_draw = info_factor.solve_Lt(z / np.sqrt(D))
    theta_draw = info_factor.solve_Pt(theta_draw)
    theta_draw = theta_draw.flatten()
    theta_draw += theta_hat
    #
    theta_prop = theta_block.copy()
    theta_prop[subset] = theta_draw

    # Check for overflow issues
    if np.max(theta_prop) >= np.log(np.finfo(np.float).max)/2.:
        # Always reject for these cases
        accept = 0
        ret_val[:end-start] = theta_block[original]

        # Transmit result
        comm.Send(ret_val, dest=MPIROOT, tag=accept)

        return

    # Demean and decorrelate previous draw
    z_prev =  L.T * info_factor.solve_P(theta_subset-theta_hat)
    z_prev = z_prev.flatten()
    z_prev *= np.sqrt(D)

    # Compute log target and proposal ratios
    log_target_ratio = -lib.loglik_convolve(theta=theta_prop,
                               y=y[block],
                               region_types=region_types[block],
                               template=template, mu=mu,
                               sigmasq=sigmasq, subset=None,
                               theta0=theta_prop, log=True)
    log_target_ratio -= -lib.loglik_convolve(theta=theta_block,
                               y=y[block],
                               region_types=region_types[block],
                               template=template, mu=mu,
                               sigmasq=sigmasq, subset=None,
                               theta0=theta_block, log=True)

    log_prop_ratio = -0.5*(prop_df+1)*np.sum(np.log(1. + z**2/prop_df)-
                                         np.log(1. + z_prev**2/prop_df))

    # Execute MH step
    log_accept_prob = log_target_ratio - log_prop_ratio
    #print block, log_target_ratio, log_prop_ratio, log_accept_prob
    if np.log(np.random.uniform()) < log_accept_prob:
        accept = 1
        ret_val[:end-start] = theta_prop[original]
    else:
        accept = 0
        ret_val[:end-start] = theta_block[original]

    # Transmit result
    comm.Send(ret_val, dest=MPIROOT, tag=accept)

def rhmc_worker_theta(comm, block_width, start, y, template, theta, mu, sigmasq,
                      region_types, prop_df=5., eps=0.1, n_steps=100):
    # Compute needed data properties
    chrom_length = y.size
    w = template.size/2 + 1

    # Calculate subset of data to work on
    end = min(chrom_length, start + block_width)
    block = slice(max(start-w, 0), min(end+w, chrom_length))
    size_block  = block.stop - block.start

    subset = slice(w*(start!=0)+start-block.start,
                   size_block-w*(end!=chrom_length) - (block.stop-end))
    size_subset = subset.stop - subset.start

    original = slice(start-block.start, size_block - (block.stop-end))
    
    theta_block     = theta[:size_block]
    theta_subset    = theta_block[subset]

    # Setup initial return value
    ret_val = np.empty(block_width)

#    # Run optimization to obtain conditional posterior mode
#    theta_hat = lib.deconvolve(lib.loglik_convolve,
#                               lib.dloglik_convolve,
#                               y[block], region_types[block], template,
#                               mu, sigmasq,
#                               subset=subset, theta0=theta_block,
#                               log=True,
#                               messages=0)[0]
#
#    # Compute (sparse) conditional observed information
#    X = sparse.spdiags((np.ones((template.size,size_block)).T *
#                        template).T, diags=range(-w+1, w),
#                        m=size_block, n=size_block, format='csr')
#
#    info = lib.ddloglik(theta=theta_hat,
#                        theta0=theta_block,
#                        X=X, Xt=X, y=y[block],
#                        mu=mu, sigmasq=sigmasq,
#                        region_types=region_types[block],
#                        subset=subset, log=True)
#    info = info[subset,:]
#    info = info.tocsc()
#    info = info[:,subset]

    # Draw momentum variables
    p = np.random.randn(size_subset)
    p_0 = p.copy()

    # Initialize new draw of theta
    theta_draw = theta_subset.copy()

    # Run leapfrog iterations
    grad = lib.dloglik_convolve(theta=theta_draw, y=y[block],
                                 region_types=region_types[block],
                                 template=template, mu=mu, sigmasq=sigmasq,
                                 theta0=theta_block, subset=subset, log=True)

    # Start with half step for momentum
    p -= eps*grad / 2.

    # Alternate full steps for position and momentum
    for i in xrange(n_steps):
        # Full step for position
        theta_draw += eps*p
        # Update gradient
        grad = lib.dloglik_convolve(theta=theta_draw, y=y[block],
                                     region_types=region_types[block],
                                     template=template, mu=mu,
                                     sigmasq=sigmasq, theta0=theta_block,
                                     subset=subset, log=True)
        # Full step for momentum, execept at the end of the trajectory
        if i<(n_steps - 1): p -= eps*grad

    # Half step for momentum at the end
    p -= eps*grad/2.

    # Reverse momentum at end of trajectory to make the proposal symmetric.
    p = -p

    # Construct complete proposal for theta
    theta_prop = theta_block.copy()
    theta_prop[subset] = theta_draw

    # Compute log target and kinetic energy differences
    log_target_ratio = -lib.loglik_convolve(theta=theta_prop, y=y[block],
                                            region_types=region_types[block],
                                            template=template, mu=mu,
                                            sigmasq=sigmasq, subset=None,
                                            theta0=theta_prop, log=True)
    log_target_ratio -= -lib.loglik_convolve(theta=theta_block, y=y[block],
                                             region_types=region_types[block],
                                             template=template, mu=mu,
                                             sigmasq=sigmasq, subset=None,
                                             theta0=theta_block, log=True)

    log_kinetic_diff = 0.5*np.sum(p**2 - p_0**2)

    # Execute MH step
    log_accept_prob = log_target_ratio - log_kinetic_diff
    #print block, log_target_ratio, log_kinetic_diff, log_accept_prob
    if np.log(np.random.uniform()) < log_accept_prob:
        accept = 1
        ret_val[:end-start] = theta_prop[original]
    else:
        accept = 0
        ret_val[:end-start] = theta_block[original]

    # Transmit result
    comm.Send(ret_val, dest=MPIROOT, tag=accept)

def worker(comm, rank, n_proc, data, init, cfg):
    '''
    Worker-node process for parallel MCMC sampler.
    Receives parameters and commands from master node, sends draws of theta.

    Parameters
    ----------
        - comm : mpi4py.MPI.COMM
            Initialized MPI communicator.
        - rank : int
            Rank (>= MPIROOT) of worker.
        - n_proc : int
            Number of processes in communicator.
        - data : dictionary
            Data as output from load_data.
        - init : dictionary
            Initial parameter values as output from initialize.
        - cfg : dictionary
            Dictionary containing (at least) prior and estimation_params
            sections with appropriate entries.

    Returns
    -------
        None.
    '''
    # Create references to relevant data entries in local namespace
    y           = data['y']
    region_types = data['region_types']
    # Template and derived properties
    template = data['template']
    w = template.size/2 + 1

    # Compute needed data properties
    chrom_length = y.size

    # Extract needed initializations for parameters
    mu      = init['mu']
    sigmasq = init['sigmasq']

    # Compute block width for parallel MH step
    n_workers = n_proc - 1
    if cfg['estimation_params']['block_width'] is None:
        block_width = chrom_length / n_workers
    else:
        block_width = cfg['estimation_params']['block_width']

    # Compute maximum size of theta slices to send
    theta_buf_size = block_width + 2*w

    # Restrict theta to needed size
    theta = np.empty(theta_buf_size, dtype=np.float)

    # Prepare to receive tasks
    working = True
    status = MPI.Status()
    start = np.array(0)

    while working:
        # Receive task information
        comm.Recv([start, MPI.INT], source=MPIROOT, tag=MPI.ANY_TAG,
                  status=status)

        if status.Get_tag() == STOPTAG:
            working = False
        elif status.Get_tag() == SYNCTAG:
            # Synchronize parameters (conditioning information)
            comm.Bcast(mu, root=MPIROOT)
            comm.Bcast(sigmasq, root=MPIROOT)
        elif status.Get_tag() == WORKTAG:
            # Update value of theta for next job within given outer loop
            comm.Recv(theta, source=MPIROOT, tag=MPI.ANY_TAG)

            # Execute HMC step, including sending result
            rhmc_worker_theta(comm=comm, block_width=block_width, start=start,
                              y=y, template=template, theta=theta, mu=mu,
                              sigmasq=sigmasq, region_types=region_types)

def run(cfg, comm=None, chrom=1, null=False):
    '''
    Coordinate parallel estimation based upon process rank.

    Parameters
    ----------
        - cfg : dictionary
            Dictionary containing (at least) prior and estimation_params
            sections with appropriate entries.
        - comm : mpi4py.MPI.COMM
            Initialized MPI communicator. If None, it will be set to
            MPI.COMM_WORLD.
        - chrom : int
            Index (starting from 1) of chromosome to extract.
        - null : bool
            If null, use null reads instead of actual.

    Returns
    -------
        For master process, dictionary from master() function. Else, None.
    '''
    if comm is None:
        # Start MPI communications if no comm provided
        comm = MPI.COMM_WORLD

    # Get process information
    rank = comm.Get_rank()
    n_proc = comm.Get_size()

    # Load data
    data = load_data(chrom=chrom, cfg=cfg, null=null)

    # Run global initialization
    init = initialize(data=data, cfg=cfg, rank=rank, null=null)

    if rank == MPIROOT:
        # Run estimation
        results = master(comm=comm, n_proc=n_proc, data=data, init=init,
                         cfg=cfg)
        return results
    else:
        worker(comm=comm, rank=rank, n_proc=n_proc, data=data, init=init,
               cfg=cfg)
        return

def write_results(results, cfg, chrom=1, null=False):
    '''
    Write results from estimation to appropriate files.

    Parameters
    ----------
        - results : dictionary
            Estimation results as output from master() function.
        - cfg : dictionary
            Dictionary containing (at least) prior and estimation_params
            sections with appropriate entries.
        - chrom : int
            Index (starting from 1) of chromosome to extract.
        - null : bool
            If null, write to null paths instead of defaults.

    Returns
    -------
        None
    '''
    # Save coefficients
    if null:
        coef_pattern = cfg['estimation_output']['null_coef_pattern']
    else:
        coef_pattern = cfg['estimation_output']['coef_pattern']
    coef_pattern = coef_pattern.strip()

    coef_path = coef_pattern.format(**cfg) % chrom
    np.savetxt(coef_path, results['theta'], '%.10g', '\t')

    # Save (lower bounds on) standard errors
    if null:
        se_pattern = cfg['estimation_output']['null_se_pattern']
    else:
        se_pattern = cfg['estimation_output']['se_pattern']
    se_pattern = se_pattern.strip()

    se_path = se_pattern.format(**cfg) % chrom
    np.savetxt(se_path, np.sqrt(results['var_theta']), '%.10g', '\t')

    # Save parameters
    if null:
        param_pattern = cfg['estimation_output']['null_param_pattern']
    else:
        param_pattern = cfg['estimation_output']['param_pattern']
    param_pattern = param_pattern.strip()

    param_path = param_pattern.format(**cfg) % chrom

    header = '\t'.join(("region_type", "mu", "sigmasq")) + '\n'

    param_file = open(param_path , "wb")
    param_file.write(header)
    for region_type in results['region_ids']:
        line = [ str(x) for x in (region_type,
                                  results['mu'][int(region_type)],
                                  results['sigmasq'][int(region_type)]) ]
        param_file.write('\t'.join(line) + '\n')
    param_file.close()

def pickle_results(results, cfg, chrom=1, null=False):
    if null:
        out_pattern = cfg['mcmc_output']['null_out_pattern']
    else:
        out_pattern = cfg['mcmc_output']['out_pattern']
    out_pattern = out_pattern.strip()

    out_path = out_pattern.format(**cfg) % chrom

    with contextlib.closing(bz2.BZ2File(out_path, mode='wb')) as f:
        cPickle.dump(results, f, protocol=-1)

def save_results(results, cfg, chrom=1, null=False):
    if null:
        out_pattern = cfg['mcmc_output']['null_out_pattern']
    else:
        out_pattern = cfg['mcmc_output']['out_pattern']
    out_pattern = out_pattern.strip()

    out_path = out_pattern.format(**cfg) % chrom

    libio.write_arrays_to_tarball(fname=out_path, compress='',
                                  scratch=cfg['mcmc_params']['path_scratch'],
                                  **results)

