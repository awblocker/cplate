import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
try:
    from scikits.sparse import cholmod
except:
    print >> sys.stderr, "Failed to load cholmod"
from mpi4py import MPI

import lib_deconvolve_em as lib

# Set constants

# MPI constants
MPIROOT     = 0
# Tags for worker states
STOPTAG     = 0
SYNCTAG     = 1
WORKTAG     = 2
UPDATETAG   = 3

# Interval between M-steps
INTERVAL    = 1

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

def initialize(data, cfg, rank=None):
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
    
    # Compute needed data properties
    n_regions = region_ids.size
    
    # Initialize nucleotide-level occupancies
    theta = (y+1.0)
    
    # Initialize mu using method-of-moments estimator based on prior variance
    sigmasq0 = b0 / a0
    mu = np.ones(n_regions)
    mu[region_ids] = np.array([np.log(theta[region_list[r]].mean()) -
                              sigmasq0 / 2.0 for r in region_ids])
    
    # Initialize sigmasq based upon prior mean
    sigmasq = np.ones(n_regions)*sigmasq0
    
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
    Master node process for parallel approximate EM. Coordinates estimation and
    collects results.

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
    # Tolerance for convergence
    tol = cfg['estimation_params']['tol']
    # Iteration limits
    min_iter = cfg['estimation_params']['min_iter']
    max_iter = cfg['estimation_params']['max_iter']    
    # Memory limits
    max_dense_mem = cfg['estimation_params']['max_mem'] * 2.**20
    # Verbosity
    verbose = cfg['estimation_params']['verbose']
    timing = cfg['estimation_params']['timing']
    # Use diagonal approximation when inverting Hessian?
    diag_approx = cfg['estimation_params']['diag_approx']
    # Debugging flags to fix hyperparameters
    fix_mu = cfg['estimation_params']['fix_mu']
    fix_sigmasq = cfg['estimation_params']['fix_sigmasq']
    
    # Compute derived quantities from config information
    sigmasq0 = b0 / a0
    adapt_prior = (mu0 is None)
    
    # Create references to relevant data entries in local scope
    y           = data['y']
    template    = data['template']
    region_types = data['region_types']
    region_list  = data['region_list']
    region_sizes = data['region_sizes']
    region_ids   = data['region_ids']
    
    # Compute needed data properties
    chrom_length = y.size
    n_regions = region_ids.size
    
    # Reference initialized quantities in local scope
    theta   = init['theta']
    mu      = init['mu']
    sigmasq = init['sigmasq']
    
    # Compute block width for parallel approximate E-step
    n_workers = n_proc - 1
    if cfg['estimation_params']['block_width'] is None:
        block_width = chrom_length / n_workers
    else:
        block_width = cfg['estimation_params']['block_width']
    
    # Compute block width and limits for bounded-memory inversion of Hessian
    var_block_width = max_dense_mem / (8*chrom_length)
    if (chrom_length / var_block_width) * var_block_width < chrom_length:
        var_max = chrom_length
    else:
        var_max = chrom_length - var_block_width
    
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
    
    # Build sparse basis
    basis = sparse.spdiags(
            (np.ones((template.size,chrom_length)).T * template).T,
            np.arange(-(template.size/2), template.size/2 + 1),
            chrom_length, chrom_length )
    
    # Setup basis matrix
    basis = basis.tocsr()
    basist = basis.T
    basist = basist.tocsr()
    
    # Initialize information for optimization
    iter = 0
    ret_val = np.empty(block_width)
    status = MPI.Status()
    
    # Start with optimization on unlogged scale
    last_switch = -1
    log = False
    
    # Setup initial values of parameters and var(theta | params)
    var_theta = sigmasq[region_types]
    params = np.array([mu, sigmasq])
    
    # Compute initial value of Q-function
    q_vec = np.empty(max_iter+1, dtype='d')
    q_vec[iter] = -lib.loglik(theta, y, region_types,
                         basis, basist,
                         slice(None), theta,
                         mu, sigmasq,
                         log=log)
    q_vec[iter] += -np.sum( var_theta / 2.0 / sigmasq[region_types] )
    q_vec[iter] += -np.sum(0.5/sigmasq*k0*mu**2)
    q_vec[iter] += -np.sum( np.log(sigmasq) )
    converged = False
    
    if log: b_previous_interval = np.exp(theta.copy())
    else:   b_previous_interval = theta.copy()
    
    # Setup blocks for worker nodes
    # This is the scan algorithm with a 2-iteration cycle.
    # It is designed to ensure consistent sampling coverage of the chromosome.
    start_vec = [np.arange(0, chrom_length, block_width, dtype=np.int),
                 np.arange(block_width/2, chrom_length, block_width, 
                           dtype=np.int)]
    start_vec = np.concatenate(start_vec)
    
    while iter < max_iter and (not converged or iter < min_iter):
        # Store estimates from last iteration for convergence check
        if log: b_previous_iteration = np.exp(theta.copy())
        else:   b_previous_iteration = theta.copy()
            
        # First, synchronize parameters across all workers
        # Coordinate the workers into the synchronization state
        for k in range(1, n_workers+1):
            comm.send((0,log), dest=k, tag=SYNCTAG)
            
        # Broadcast theta and parameter values to all workers
        comm.Bcast(theta, root=MPIROOT)
        params[0], params[1] = (mu, sigmasq)
        comm.Bcast(params, root=MPIROOT)
        
        # Dispatch jobs to workers until completed
        n_jobs       = start_vec.size
        n_started    = 0
        n_completed  = 0
        
        # Randomize block ordering
        np.random.shuffle(start_vec)
        
        # Send first batch of jobs
        for k in range(1,min(n_workers, start_vec.size)+1):
            comm.send((start_vec[n_started],log), dest=k, tag=WORKTAG)
            n_started += 1
        
        # Collect results from workers and dispatch additional jobs until
        # complete
        while n_completed < n_jobs:
            # Collect any complete results
            comm.Recv(ret_val, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                      status=status)
            n_completed += 1
            start = status.Get_tag()
            end = min(start+block_width, chrom_length)
            theta[start:end] = ret_val[:end-start]
            
            # If all jobs are not complete, update theta on the just-finished
            # worker and send another job.
            if n_started < n_jobs:
                # Update theta on given worker
                worker = status.Get_source()
                comm.send((0,log), dest=worker, tag=UPDATETAG)
                comm.Send(theta, dest=worker, tag=MPIROOT)
                
                # Start next job on worker
                comm.send((start_vec[n_started],log), dest=worker, tag=WORKTAG)
                n_started += 1
        
        # Exponentiate resulting theta if needed
        if log:
            logb = theta
        else:
            logb = np.log(theta)
        
        # Run M-step at appropriate intervals
        if iter % INTERVAL == 0 and iter > 0:
            if verbose and timing: tme = time.clock()
            if not fix_sigmasq:
                if diag_approx:
                    Hdiag = lib.ddloglik_diag(logb, y, region_types, basis,
                                              basist, slice(None),
                                              logb, mu, sigmasq, log=True)
                    var_theta = 1.0/Hdiag
                else:
                    H = lib.ddloglik(logb, y, region_types, basis, basist,
                                     slice(None), logb,
                                     mu, sigmasq, log=True)
                    try:
                        Hfactor = cholmod.cholesky(H)
                        for start in xrange(0, var_max, var_block_width):
                            stop = min(chrom_length, start+var_block_width)
                            var_theta[start:stop] = Hfactor.solve_A(
                                    np.eye(chrom_length, stop - start, -start)
                                    ).diagonal()
                    except:
                        if verbose: print 'Cholesky fail'
                        Hfactor = splinalg.splu(H)
                        for start in xrange(0, var_max, var_block_width):
                            stop = min(chrom_length, start+var_block_width)
                            print (start, stop)
                            var_theta[start:stop] = Hfactor.solve(
                                    np.eye(chrom_length, stop - start, -start)
                                    ).diagonal()
                if verbose and timing:
                    print >> sys.stderr, ( "var_theta time: %s" %
                                           (time.clock() - tme) )
                    tme = time.clock()
            
            for r in region_ids:
                region = region_list[r]
                if not fix_mu:
                    mu[r] = np.mean(logb[region]) + prior_mean[r]*k0
                    mu[r] /= 1.0 + k0
                
                if not fix_sigmasq:
                    sigmasq[r] = np.mean( (logb[region]-mu[r])**2 )
                    sigmasq[r] += np.mean( var_theta[region] )
                    sigmasq[r] += k0*(mu[r]-prior_mean[r])**2
                    sigmasq[r] += 2.*b0/region_sizes[r]
                    sigmasq[r] /= (1 + 3./region_sizes[r] +
                                   2.*a0/region_sizes[r])
            
            if verbose:
                if timing: print >> sys.stderr, ( "Mean & variance time: %s" %
                                                  (time.clock() - tme) )
                if verbose > 1: print mu, sigmasq
        
        # Update Q-function value
        # NOTE: This need not increase at each iteration; indeed, it can
        # monotonically decrease in common cases (e.g. normal-normal model)
        iter += 1
        q_vec[iter] = -lib.loglik(theta, y, region_types,
                             basis, basist,
                             slice(None), theta,
                             mu, sigmasq,
                             log=log)
        q_vec[iter] += -np.sum( var_theta / 2.0 / sigmasq[region_types] )
        q_vec[iter] += -np.sum(0.5/sigmasq*k0*mu**2)
        q_vec[iter] += -np.sum( np.log(sigmasq) )
        
        # Using L_2 convergence criterion on estimated parameters of interest
        # (theta)
        delta_iteration = lib.l2_error( np.exp(logb), b_previous_iteration )
        if iter % INTERVAL == 0 and iter > 0:
            delta_interval = lib.l2_error( np.exp(logb), b_previous_interval )
            b_previous_interval = np.exp(logb)
            converged = (delta_interval < tol)
        
        if verbose:
            print q_vec[iter]
            print delta_iteration
            print iter
            if iter % INTERVAL == 0 and iter > 0: print delta_interval
        
        # Switch between optimizing over log(b) and b
        if converged:
            if last_switch < 0:
                log = not log
                converged = False
                last_switch = iter
                b_last_switch = np.exp(logb)
                
                if log: theta = np.log(theta)
                else: theta = np.exp(theta)
                
                if verbose:
                    print 'Last switch: %d' % last_switch
                    print 'Log: %s' % str(log)
            else:
                # Check if switching space helped
                delta = lib.l2_error( np.exp(logb), b_last_switch )
                converged = (delta < tol)
                
                if not converged:
                    # If it did help, keep going
                    b_last_switch = np.exp(logb)
                    last_switch = iter
                    log = not log
                    
                    if log: theta = np.log(theta)
                    else: theta = np.exp(theta)
                    
                    if verbose:
                        print 'Last switch: %d' % last_switch
                        print 'Log: %s' % str(log)
        
    
    # Halt all workers
    for k in range(1,n_proc):
        comm.send((None,None), dest=k, tag=STOPTAG)
    
    # Exponentiate coefficients, if needed
    if log: theta = np.exp(theta)
    
    # Return results
    out = {'theta' : theta,
           'var_theta' : var_theta,
           'mu' : mu,
           'sigmasq' : sigmasq,
           'region_ids' : region_ids}
    return out

def worker(comm, rank, n_proc, data, init, cfg):
    '''
    Worker-node process for parallel approximate EM algorithm. Receives
    parameters and commands from master node, sends updated estimates.

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
    template    = data['template']
    region_types = data['region_types']
    
    # Compute needed data properties
    chrom_length = y.size
    w = template.size/2 + 1
    
    # Extract needed initializations for parameters
    theta   = init['theta']
    mu      = init['mu']
    sigmasq = init['sigmasq']
    params  = np.array([mu, sigmasq])
    
    # Compute block width for parallel approximate E-step
    n_workers = n_proc - 1
    if cfg['estimation_params']['block_width'] is None:
        block_width = chrom_length / n_workers
    else:
        block_width = cfg['estimation_params']['block_width']
    
    # Prepare to receive tasks
    working = True
    status = MPI.Status()
    ret_val = np.empty(block_width, dtype=np.float)
    while working:
        # Receive task information
        start, log = comm.recv(source=MPIROOT, tag=MPI.ANY_TAG, status=status)
        
        if status.Get_tag() == STOPTAG:
            working = False
        elif status.Get_tag() == SYNCTAG:
            # Synchronize parameters (conditioning information)
            comm.Bcast(theta, root=MPIROOT)
            comm.Bcast(params, root=MPIROOT)
            mu, sigmasq = params
        elif status.Get_tag() == WORKTAG:            
            # Calculate subset of data to work on
            end = min(chrom_length, start + block_width)
            block = slice(max(start-w, 0), min(end+w, chrom_length))
            size_block  = block.stop - block.start
            
            subset = slice(w*(start!=0)+start-block.start,
                           size_block-w*(end!=chrom_length) - (block.stop-end))
            
            original = slice(start-block.start, size_block - (block.stop-end))
            
            # Setup initial return value
            ret_val[end-start:] = 0
            
            # Run optimization
            result = lib.deconvolve(lib.loglik_convolve, lib.dloglik_convolve,
                                    y[block], region_types[block], template,
                                    mu, sigmasq,
                                    subset=subset, theta0=theta[block],
                                    log=log,
                                    messages=0)
            
            # Build resulting subset of new theta
            theta_new = theta[block]
            theta_new[subset] = result[0]
            ret_val[:end-start] = theta_new[original]
            
            # Transmit result
            comm.Send(ret_val, dest=MPIROOT, tag=start)
        elif status.Get_tag() == UPDATETAG:
            # Update value of theta for next job within given outer loop
            comm.Recv(theta, source=MPIROOT, tag=MPI.ANY_TAG)

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
    init = initialize(data=data, cfg=cfg, rank=rank)
    
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

