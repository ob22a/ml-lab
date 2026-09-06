import numpy as np

def bernoulli_pmf(x, p):
    """Probability mass function of a Bernoulli distribution."""
    return (p ** x) * ((1 - p) ** (1 - x))

def binomial_pmf(k, n, p):
    """Probability mass function of a Binomial distribution."""
    import math
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def gaussian_pdf(x, mu, sigma):
    """Probability density function of a univariate Gaussian."""
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def multivariate_gaussian_pdf(X, mu, Sigma):
    """
    Probability density function of a multivariate Gaussian.
    X: (d,) or (n, d) array
    mu: (d,) array
    Sigma: (d, d) array
    """
    X = np.atleast_2d(X)
    d = mu.shape[0]
    det_Sigma = np.linalg.det(Sigma)
    inv_Sigma = np.linalg.inv(Sigma)
    
    # (x - mu)
    diff = X - mu
    
    # Quadratic form: (x - mu)^T Sigma^-1 (x - mu)
    # Using einsum for efficient batch computation
    quadratic_form = np.einsum('ij,jk,ik->i', diff, inv_Sigma, diff)
    
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * det_Sigma)
    pdf = norm_const * np.exp(-0.5 * quadratic_form)
    
    return pdf if pdf.shape[0] > 1 else pdf[0]

def expected_value(x, p_x):
    """Expected value of discrete random variable."""
    return np.sum(x * p_x)

def variance(x, p_x):
    """Variance of a discrete random variable."""
    mu = expected_value(x, p_x)
    return expected_value((x - mu)**2, p_x)

def empirical_covariance(x, y):
    """Empirical covariance of two arrays x and y (unbiased)."""
    assert len(x) == len(y)
    n = len(x)
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    return np.sum((x - mu_x) * (y - mu_y)) / (n - 1)
