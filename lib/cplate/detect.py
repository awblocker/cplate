import numpy as np
from scipy import stats

import lib_detect as lib

def calculate_fdr_threshold_vector(chrom, cfg, **kwargs):
    '''
    Compute vector of FDR-controlling detection thresholds by region.
    
    Parameters
    ----------
        - chrom : int
            Index of chromosome to analyze
        - cfg : dictionary
            Dictionary of configuration information, formatted as documented
            elsewhere.
        - **kwargs
            Optional arguments to avoid reloading data
    
    Returns
    -------
        Dictionary containing:
        - thresh_vec : ndarray
            Vector of detection thresholds by region
        - region_ids : integer ndarray
            Vector of region identifiers
        - alpha : float
            FDR
    '''
    # Extract commonly-used data from cfg
    alpha       = cfg['detection_params']['alpha']
    n_proc      = cfg['detection_params']['n_proc']
    maxima      = cfg['detection_params']['compute_maxima_only']
    method      = cfg['detection_params']['method_fdr']
    verbose     = cfg['detection_params']['verbose']
    
    # Load null coefficients
    if 'null' in kwargs.keys():
        null = kwargs['null']
    else:
        null_path = cfg['estimation_output']['null_coef_pattern']
        null_path = null_path.format(**cfg).strip() 
        null_path = null_path % chrom
        null = np.loadtxt(null_path)
    
    # Load nonnull coefficients
    if 'nonnull' in kwargs.keys():
        nonnull = kwargs['nonnull']
    else:
        nonnull_path = cfg['estimation_output']['coef_pattern']
        nonnull_path = nonnull_path.format(**cfg).strip() 
        nonnull_path = nonnull_path % chrom
        nonnull = np.loadtxt(nonnull_path)
    
    # Load region type information
    if 'region_types' in kwargs.keys() and 'region_ids' in kwargs.keys():
        region_types = kwargs['region_types']
        region_ids = kwargs['region_ids']
    else:
        with open(cfg['data']['regions_path'], 'rb') as f:
            lines_read = 0
            for line in f:
                lines_read += 1
                if lines_read == chrom:
                    region_types = np.fromstring(line.strip(), sep=' ',
                                                 dtype=int)
                    break
        
        region_types = region_types[:null.size]
        region_types -= region_types.min()
        region_ids = np.unique(region_types)
    
    if 'region_list' in kwargs.keys() and 'region_lengths' in kwargs.keys():
        region_list = kwargs['region_list']
        region_lengths = kwargs['region_lengths']
    else:
        region_list = []
        region_lengths = []
        for id in region_ids:
            region_list.append( np.where(region_types==id)[0] )
            region_lengths.append( np.sum(region_types==id) )
        
        region_lengths = np.array(region_lengths)
    
    # Calculate threshold for given FDR
    if method.lower() == 'bh':
        thresh_vec = lib.get_fdr_threshold_bh(null=null, nonnull=nonnull,
                                              region_list=region_list,
                                              alpha=alpha, maxima=maxima,
                                              n_proc=n_proc, verbose=verbose)
    elif method.lower() == 'direct':
        thresh_vec = lib.get_fdr_threshold_estimate(null, nonnull, region_list,
                                                    alpha, maxima=maxima,
                                                    n_proc=n_proc,
                                                    verbose=verbose)
    else:
        thresh_vec = lib.get_fdr_threshold(null, nonnull, region_list, alpha,
                                           maxima=maxima)

    result = {'thresh_vec' : thresh_vec,
              'region_ids' : region_ids,
              'alpha' : alpha}

    return result

def write_fdr_thresholds(result, cfg, chrom=1):
    '''
    Output FDR threshold vector to appropriate file.

    Parameters
    ----------
        - result : ndarray
            Dictionary as returned by calculate_fdr_threshold_vector.
        - cfg : dictionary
            Dictionary containing at least detection_output section wth
            appropriate parameters.

    Returns
    -------
        None
    '''
    # Output detection threshold by region to appropriate path
    out_path = cfg['detection_output']['fdr_pattern'].format(**cfg).strip()
    out_path = out_path % chrom
    
    n_regions = result['region_ids'].size
    
    np.savetxt(out_path, np.vstack((result['alpha']*np.ones(n_regions),
                                    result['region_ids'],
                                    result['thresh_vec'])).T,
               fmt="%.15g")

def detect(cfg, chrom=1):
    '''
    Coordinate detection procedure.
    
    Parameters
    ----------
        - cfg : dictionary
            Dictionary of parameters containing at least those relevant for
            detection.
        - chrom : int
            Index of chromosome to analyze
    '''
    
    # Load nonnull coefficients
    coef_path = cfg['estimation_output']['coef_pattern']
    coef_path = coef_path.format(**cfg).strip() 
    coef_path = coef_path % chrom
    coef = np.loadtxt(coef_path)
    
    # Load region_types
    with open(cfg['data']['regions_path'], 'rb') as f:
        lines_read = 0
        for line in f:
            lines_read += 1
            if lines_read == chrom:
                region_types = np.fromstring(line.strip(), sep=' ',
                                             dtype=int)
                break
    
    region_types = region_types[:coef.size]
    region_types -= region_types.min()
    region_ids = np.unique(region_types)
    
    # Obtain FDR-based detection thresholds
    results_fdr = calculate_fdr_threshold_vector(chrom=chrom, cfg=cfg,
                                                 nonnull=coef,
                                                 region_types=region_types,
                                                 region_ids=region_ids)
    
    # Output FDR thresholds
    write_fdr_thresholds(results_fdr, cfg, chrom=chrom)
    
    if cfg['detection_params']['use_bayes_se']:
        # Load standard errors
        se_path = cfg['estimation_output']['se_pattern']
        se_path = se_path.format(**cfg).strip() 
        se_path = se_path % chrom
        se = np.loadtxt(se_path)
        
        # Load parameters
        param_path = cfg['estimation_output']['param_pattern']
        param_path = param_path.format(**cfg).strip() 
        param_path = param_path % chrom
        mu, sigmasq = np.loadtxt(param_path, unpack=True, skiprows=1)[1:]
        
        mu = np.ravel(mu)
        sigmasq = np.ravel(sigmasq)
        
        # Compute n_se for detection
        n_se = -stats.norm.ppf(cfg['detection_params']['alpha'])
        
        # Detect positions based upon both FDR and Bayes criteria
        detected = np.where((coef > results_fdr['thresh_vec'][region_types]) &
                            (np.log(coef) - n_se*se > mu[region_types]))[0]        
    else:
        detected = np.arange(coef.size, dtype=np.integer)
    
    # Restrict to local maxima if request
    if cfg['detection_params']['detect_maxima_only']:
        detected = np.intersect1d(lib.find_maxima(coef), detected)
    
    # Output detected positions
    detected_path = cfg['detection_output']['detected_pattern'].format(**cfg)
    detected_path = detected_path.strip() % chrom
    np.savetxt(detected_path, detected, fmt='%d')
