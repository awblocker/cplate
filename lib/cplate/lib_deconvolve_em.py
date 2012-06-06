# Load libraries
import numpy as np
from scipy import optimize, sparse
from scipy.sparse import sparsetools

# Define functions
def csr_scale_rows(A, x):
    sparsetools.csr_scale_rows(A.shape[0], A.shape[1],
                               A.indptr, A.indices, A.data,
                               x)
def csr_scale_columns(A, x):
    sparsetools.csr_scale_columns(A.shape[0], A.shape[1],
                                  A.indptr, A.indices, A.data,
                                  x)

def l2_error(x1, x2):
    return np.sqrt( np.mean( (x1 - x2)**2 ) )

def l1_error(x1, x2):
    return np.mean(np.abs(x1-x2))

def loglik(theta, y, region_types, X, Xt, subset, theta0,
           mu, sigmasq, omega=1.0, log=False):
    b = theta0.copy()
    b[subset] = theta
    logb = b
    if log: b = np.exp(logb)
    else: logb = np.log(b)
    
    lam = X * (omega*b)
    
    u = logb - mu[region_types]
    
    val = np.sum(lam) - np.sum( y * np.log(lam) )
    val += np.sum(u*u/sigmasq[region_types])/2.0
    val += np.log(sigmasq[region_types]).sum()/2.0
    return val

def loglik_convolve(theta, y, region_types, template, subset, theta0,
                    mu, sigmasq, omega=1.0, log=False):
    b = theta0.copy()
    b[subset] = theta
    logb = b
    if log: b = np.exp(logb)
    else: logb = np.log(b)
    
    lam = omega * np.convolve(b, template, mode='same')
    
    u = logb - mu[region_types]
    
    val = np.sum(lam) - np.sum( y * np.log(lam) )
    val += np.sum(u*u/sigmasq[region_types])/2.0
    val += np.log(sigmasq[region_types]).sum()/2.0
    
    return val

def dloglik(theta, y, region_types, X, Xt, subset, theta0,
            mu, sigmasq, omega=1.0, log=False):
    b = theta0.copy()
    b[subset] = theta
    logb = b
    if log: b = np.exp(logb)
    else: logb = np.log(b)
    #
    lam = X * (omega*b)
    #
    u = logb - mu[region_types]
    #
    grad = Xt * ( omega*(lam-y)/lam )
    grad += u/sigmasq[region_types]/b
    if log: grad *= b
    #
    return grad[subset]

def dloglik_convolve(theta, y, region_types, template, subset, theta0,
                     mu, sigmasq, omega=1.0, log=False):
    b = theta0.copy()
    b[subset] = theta
    logb = b
    if log: b = np.exp(logb)
    else: logb = np.log(b)
    
    lam = omega * np.convolve(b, template, mode='same')
    
    u = logb - mu[region_types]
    
    grad = omega * np.convolve((lam-y)/lam, template, mode='same')
    grad += u/sigmasq[region_types]/b
    if log: grad *= b
    #
    return grad[subset]

def ddloglik(theta, y, region_types, X, Xt, subset, theta0,
             mu, sigmasq, omega=1.0, log=True):
    b = theta0.copy()
    b[subset] = theta
    logb = b
    if log: b = np.exp(logb)
    else: logb = np.log(b)
    #
    lam = X * (omega*b)
    w = np.sqrt(y) / lam
    #
    Z = omega * X
    Zt = omega * Xt
    csr_scale_rows(Z, w)
    csr_scale_columns(Zt, w)
    if log:
        csr_scale_columns(Z, b)
        csr_scale_rows(Zt, b)
    H = Zt * Z
    if log: H = H + sparse.spdiags( Xt * ( omega*(lam-y)/lam )*b,
                                    0, X.shape[1], X.shape[1], 'csr' )
    #
    Sigma_inv = sparse.spdiags( np.ones(X.shape[1])/sigmasq[region_types], 0,
                               X.shape[1], X.shape[1],
                               'csr' )
    if not log:
        w = 1.0/b
        csr_scale_rows(Sigma_inv, w)
        csr_scale_columns(Sigma_inv, w)
        #
        u = logb - mu[region_types]
        #
        grad = u/sigmasq[region_types]/b
        #
        Sigma_inv = Sigma_inv - sparse.dia_matrix( (grad, 0),
                                                 (X.shape[1], X.shape[1]) )
    #
    H = H + Sigma_inv
    return H

def ddloglik_p(theta, p, y, region_types, X, Xt, subset, theta0,
               mu, sigmasq, omega=1.0, log=False):
    b = theta0.copy()
    b[subset] = theta
    logb = b
    if log: b = np.exp(logb)
    else: logb = np.log(b)
    #
    lam = X * (omega*b)
    w = np.sqrt(y) / lam
    #
    Z = omega * X
    Zt = omega * Xt
    csr_scale_rows(Z, w)
    csr_scale_columns(Zt, w)
    if log:
        csr_scale_columns(Z, b)
        csr_scale_rows(Zt, b)
    H = Zt * Z
    if log: H = H + sparse.spdiags( Xt * ( omega*(lam-y)/lam )*b,
                                    0, X.shape[1], X.shape[1], 'csr' )
    #
    Sigma_inv = sparse.spdiags( np.ones(X.shape[1])/sigmasq[region_types], 0,
                               X.shape[1], X.shape[1],
                               'csr' )
    if not log:
        w = 1.0/b
        csr_scale_rows(Sigma_inv, w)
        csr_scale_columns(Sigma_inv, w)
        #
        u = logb - mu[region_types]
        #
        grad = u/sigmasq[region_types]/b
        #
        Sigma_inv = Sigma_inv - sparse.dia_matrix( (grad, 0),
                                                 (X.shape[1], X.shape[1]) )
    #
    H = H + Sigma_inv
    H = H.tocsr()
    H = H[subset]
    H = H.tocsc()
    H = H[:,subset]
    H = H.tocsr()
    #
    return H * p

def ddloglik_diag(theta, y, region_types, X, Xt, subset, theta0,
                  mu, sigmasq, omega=1.0, log=True):
    b = theta0.copy()
    b[subset] = theta
    logb = b
    if log: b = np.exp(logb)
    else: logb = np.log(b)
    #
    lam = X * (omega*b)
    w = np.sqrt(y) / lam
    #
    Z = omega * X
    csr_scale_rows(Z, w)
    if log:
        csr_scale_columns(Z, b)
    Z.data = Z.data**2
    Hdiag = Z.sum(0)
    Hdiag = np.asarray(Hdiag).flatten()
    if log: Hdiag = Hdiag + (Xt * ( omega*(lam-y)/lam )*b)
    #
    Sigma_inv_diag = np.ones(X.shape[1])/sigmasq[region_types]
    if not log:
        Sigma_inv_diag /= b**2
        #
        u = logb - mu[region_types]
        #
        grad = u/sigmasq[region_types]/b
        #
        Sigma_inv_diag = Sigma_inv_diag - grad
    #
    Hdiag = Hdiag + Sigma_inv_diag
    return Hdiag

# Identify active components of basis
def find_active(y, w=38):
    N = y.shape[0]
    active = np.zeros(N, np.bool)
    for n in range(N):
        active[n] = (y[ np.maximum(0,n-w+1):np.minimum(n+w,N) ].max() > 0)
    return active

def deconvolve(loglik, dloglik, y, region_types, template,
               mu, sigmasq,
               subset=slice(None), theta0=None, omega=1.0, log=False, 
               lower_bound = np.sqrt( np.finfo(float).eps ),
               **kwargs):
    m = subset.stop - subset.start
    
    if theta0 is None:
        theta0 = y + 1
        if log: theta0 = np.log(theta0)
    if log:
        lower_bound = np.log(lower_bound)
    
    result = optimize.fmin_tnc( loglik, theta0[subset], dloglik,
                                args=(y, region_types, template, subset, theta0,
                                      mu, sigmasq, omega, log),
                                bounds = zip( np.ones(m)*lower_bound,
                                              np.ones(m)*np.Inf ),
                                **kwargs )
    return result

