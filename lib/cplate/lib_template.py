
import sys
import getopt

import numpy as np
from scipy import optimize

from cplate.libio import *

def mvlogit(p):
    '''
    Run multivariate logit transformation on given probability vector
    '''
    pm1 = 1.0 - np.sum(p[:-1])
    x   = np.log(p[:-1]) - np.log(pm1)
    return x

def invmvlogit(x):
    '''
    Inverse multivariate logit transformation
    '''
    p = np.zeros(x.size+1)
    denom = 1.0 + np.sum(np.exp(x))
    p[:-1] = np.exp(x) / denom
    p[-1]  = 1.0 / denom
    return p

def loglik(q, x, n, l0):
    '''
    Function to calculate log-likelihood of data given distribution of
    digestion errors (q) and baseline length (l0)
    
    Takes vector of unique observed lengths (x) and number of fragments
    observed at each length (n)
    '''
    # Calculate m and lMax
    m = -np.floor(l0 / 2)
    lMax = np.max(x)
    
    # Calculate probabilities from q vectors
    q1, q2 = np.zeros( (2, q.size+1+l0) )
    q1[:q.size] = q
    q2[-q.size:] = q
    p = np.convolve(q1, q2)[-2*m+1:]
    
    # Calculate log-likelihood of sample
    return np.sum( np.log(p[x])*n )

def obj(theta, x, n, l0):
    '''
    Objective function for MLE
    Incorporates multivariate logit transformation on q and negates
    ll to avoid complications with standard optimization routines
    '''
    # Reverse multivariate logit transformation on q
    q = invmvlogit(theta)
    
    # Calculate loglik
    return -loglik(q, x, n, l0)

def initialize(x, n, l0, eps=np.sqrt(np.finfo(float).eps)):
    '''
    Initialization for MLE based on Gaussian approximation
    Typically accelerates convergence
    '''
    # Calculate m and lMax
    m = -np.floor(l0 / 2)
    lMax = np.max(x)
    
    # Calculate summary statistics for lengths
    xbar = np.sum( x * n / n.sum() )
    v    = np.sum( (x-xbar)**2 * n / n.sum() )
    
    # Calculate Gaussian PDF for needed values
    q = np.exp( -(np.arange(m,lMax-l0-m+1))**2 / v )
    
    # Normalize q & handle small values
    q[q<eps] = eps
    return q/q.sum()

def rescale2(t):
    '''
    Rescale template t by 1/2; gives distribution of x/2 if x ~ t, where 
    non-integer values are handled by random rounding.

    If t.size = 2*w + 1, returns vector of size w + 1
    '''
    # Build transformation matrix
    # Only working with half of template t[w:] for simplicity
    w   = t.size / 2
    w2  = (w+1) / 2
    
    A = np.zeros((w2+1,w+1))
    
    # Rules for mode
    A[0,0]  = 1
    A[0,1]  = 1
    
    # Iterate over remaining output positions
    for i in xrange(1,w2+1):
        A[i,2*i-1]  = 0.5
        if 2*i < w+1:
            A[i,2*i] = 1
            if 2*i < w:
                A[i,2*i+1] = 0.5
    
    # Compute t2[w2:] as matrix product
    t2 = np.dot(A, t[w:])
    # Mirror positions
    t2 = np.r_[t2[:0:-1], t2]
    
    return t2

def estimateTemplate(x, n, l0, thresh=0.999, verbose=0):
    '''
    Function to calculate MLE for template distribution
    
    Takes vector of unique observed lengths (x) and number of fragments
    observed at each length (n), as well as baseline fragment length (l0)
    
    Returns estimated small template (t), complete template (tComplete),
    template width (w), error distribution (q), and estimated distribution of
    fragment lengths (p)
    
    Threshold specifies probability for template to cover; defaults to 0.999
    '''
    # Set dtypes as needed
    x = x.astype(int)
    
    # Initialize q
    q = initialize(x, n, l0)
    
    # Transform q via multivariate logit
    theta0 = mvlogit(q)
    
    # Run optimization
    if verbose > 1:
        iprint = verbose - 1
    else:
        iprint = -1
    
    theta, val, info = optimize.fmin_l_bfgs_b(obj, theta0, args=(x, n, l0),
                                              approx_grad=True,
                                              iprint=iprint)
    
    # Print diagnostic information if request
    if verbose > 0:
        print >> sys.stderr, "Log-likelihood = %g" % (-val)
        print >> sys.stderr, "Convergence = %s" % info['warnflag']
        if info['warnflag'] > 1:
            print >> sys.stderr, info['task']
    
    # Calculate p & t
    q = invmvlogit(theta)
    
    # Estimated distribution of summed lengths
    q1, q2 = np.zeros( (2, q.size+1+l0) )
    q1[:q.size] = q
    q2[-q.size:] = q
    p = np.convolve(q1, q2)[2*np.floor(l0/2)+1:]
    
    # Calculate large, complete template
    w = x.max() - l0 + np.floor(l0/2)
    q1, q2 = np.zeros( (2, 2*w + 1))
    q1[-q.size:] = q
    q2[:q.size]  = q[::-1]
    tComplete = np.convolve(q1, q2)
    
    # Find template width
    coverage = 2*np.cumsum(tComplete[tComplete.size/2+1:])
    coverage += tComplete[tComplete.size/2]
    w = np.min( np.where(coverage >= thresh) )
    
    # Output final template
    t = tComplete[tComplete.size/2 - w: tComplete.size/2 + 1 + w]
    t = t / t.sum()
    
    return t, tComplete, w, q, p

def buildTemplateFromDist(distFile, outFile, l0, coverage, verbose=0,
                          rescale=False):
    '''
    Wrapper function for template estimation process
    Calls estimateTemplate
    Takes distFile, outFile, l0, and coverage as inputs
    Writes final template to outFile
    '''
    # Read distribution from file
    x, n = np.loadtxt(distFile, unpack=True)
    
    # Estimate template
    t, tComplete, w, q, p = estimateTemplate(x, n, l0, coverage, verbose)
    
    if verbose > 0:
        print >> sys.stderr, 'w = %d' % w

    # Rescale template before output if requested
    if rescale:
        t   = rescale2(t)
    
    # Write template in column format to outFile
    np.savetxt(outFile, t[:,np.newaxis])
    
def estimateErrorDist(x, n, l0, thresh=0.999, verbose=0):
    '''
    Function to calculate MLE for template distribution
    
    Takes vector of unique observed lengths (x) and number of fragments
    observed at each length (n), as well as baseline fragment length (l0)
    
    Returns estimated small template (t), complete template (tComplete),
    template width (w), error distribution (q), and estimated distribution of
    fragment lengths (p)
    
    Threshold specifies probability for template to cover; defaults to 0.999
    '''
    # Set dtypes as needed
    x = x.astype(int)
    
    # Initialize q
    q = initialize(x, n, l0)
    
    # Transform q via multivariate logit
    theta0 = mvlogit(q)
    
    # Run optimization
    if verbose > 1:
        iprint = verbose - 1
    else:
        iprint = -1
    
    theta, val, info = optimize.fmin_l_bfgs_b(obj, theta0, args=(x, n, l0),
                                              approx_grad=True,
                                              iprint=iprint)
    
    # Print diagnostic information if request
    if verbose > 0:
        print >> sys.stderr, "Log-likelihood = %g" % (-val)
        print >> sys.stderr, "Convergence = %s" % info['warnflag']
        if info['warnflag'] > 1:
            print >> sys.stderr, info['task']
    
    # Calculate p & t
    q = invmvlogit(theta)
    
    return q

def buildErrorDistFromLengths(distFile, outFile, l0, coverage, verbose=0,
                              rescale=False):
    '''
    Wrapper function for template estimation process
    Calls estimateTemplate
    Takes distFile, outFile, l0, and coverage as inputs
    Writes final template to outFile
    '''
    # Read distribution from file
    x, n = np.loadtxt(distFile, unpack=True)
    
    # Estimate digestion error distribution
    q = estimateErrorDist(x, n, l0, coverage, verbose)
    e = np.arange(-np.floor(l0/2), q.size-np.floor(l0/2), dtype=int)
    
    if verbose > 0:
        print >> sys.stderr, 'w = %d' % w

    # Write digestion error distribution in column format to outFile
    result = np.rec.fromarrays([e,q])
    write_recarray_to_file(sys.stdout, result, header=False)

#def buildConditionalTemplateFromLengths(distFile, outFile, l0, coverage,
#                                        length_min=None, length_max=None,
#                                        verbose=0, rescale=False):
#    '''
#    Wrapper function for template estimation process, conditioning on a
#    particular range of fragment lengths.
#
#    First estimations digestion error distribution, then uses it estimate the
#    conditional template.
#
#    Takes distFile, outFile, l0, and coverage as inputs
#    Writes final template to outFile
#    '''
#    # Read distribution from file
#    x, n = np.loadtxt(distFile, unpack=True)
#    
#    # Estimate digestion error distribution
#    q = estimateErrorDist(x, n, l0, coverage, verbose)
#    e = np.arange(-np.floor(l0/2), q.size-np.floor(l0/2), dtype=int)
#
#    # Estimated distribution of summed lengths
#    q1, q2 = np.zeros( (2, q.size+1+l0) )
#    q1[:q.size] = q
#    q2[-q.size:] = q
#    p = np.convolve(q1, q2)[2*np.floor(l0/2)+1:]
#
#    # Compute conditional distribution of difference given the sum is between
#    # the given values
#    if length_min is None:
#        length_min = 1
#    if length_max is None:
#        length_max = np.max(x)
#
#    w = x.max() - l0 + np.floor(l0/2)
#    P_d = np.zeros((length_max - length_min + 1), 2 * w + 1)
#    for s in xrange(length_min - length_max + 1):
#        P_d[s, ] = 
#    
#    # Find template width
#    coverage = 2*np.cumsum(tComplete[tComplete.size/2+1:])
#    coverage += tComplete[tComplete.size/2]
#    w = np.min( np.where(coverage >= thresh) )
#    
#    # Output final template
#    t = tComplete[tComplete.size/2 - w : tComplete.size/2 + 1 + w]
#    t = t / t.sum()
#    
#    if verbose > 0:
#        print >> sys.stderr, 'w = %d' % w
#
#    # Write digestion error distribution in column format to outFile
#    result = np.rec.fromarrays([e,q])
#    write_recarray_to_file(sys.stdout, result, header=False)


