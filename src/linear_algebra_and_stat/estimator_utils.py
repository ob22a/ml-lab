import numpy as np


def sample_mean(data):
    """Unbiased estimator of the population mean."""
    data = np.asarray(data, dtype=float)
    return float(np.sum(data) / len(data))


def first_observation(data):
    """Unbiased but high-variance estimator: ignore all but X_1."""
    return float(np.asarray(data, dtype=float)[0])


def shrunk_mean(data, shrinkage=0.7, target=0.0):
    """
    Biased, lower-variance estimator: pull the sample mean toward `target`.

    \\hat\\theta = (1 - \\lambda) * target + \\lambda * \\bar X
    with \\lambda = shrinkage. For shrinkage < 1 this is biased whenever
    E[\\bar X] \\neq target, but has smaller variance than \\bar X.
    """
    return (1.0 - shrinkage) * target + shrinkage * sample_mean(data)


def biased_variance(data):
    """MLE / biased estimator of the population variance (divide by n)."""
    data = np.asarray(data, dtype=float)
    mu_hat = sample_mean(data)
    return float(np.sum((data - mu_hat) ** 2) / len(data))


def unbiased_variance(data):
    """Unbiased estimator of the population variance (divide by n-1)."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n <= 1:
        return 0.0
    mu_hat = sample_mean(data)
    return float(np.sum((data - mu_hat) ** 2) / (n - 1))


def standard_error_of_mean(data, population_sigma=None):
    """
    Standard error of the sample mean.

    If population_sigma is unknown, estimate it with the unbiased sample std.
    """
    n = len(np.asarray(data))
    if population_sigma is not None:
        return float(population_sigma / np.sqrt(n))
    return float(np.sqrt(unbiased_variance(data)) / np.sqrt(n))


def empirical_bias(estimates, true_parameter):
    return float(np.mean(estimates) - true_parameter)


def empirical_variance(estimates):
    return unbiased_variance(estimates)


def empirical_mse(estimates, true_parameter):
    estimates = np.asarray(estimates, dtype=float)
    return float(np.mean((estimates - true_parameter) ** 2))


def evaluate_estimator(estimator_fn, samples, true_parameter):
    """
    Apply an estimator to many samples and report bias, variance, and MSE.

    `samples` has shape (n_reps, n).
    """
    estimates = np.array([estimator_fn(sample) for sample in samples], dtype=float)
    bias = empirical_bias(estimates, true_parameter)
    var = empirical_variance(estimates)
    mse = empirical_mse(estimates, true_parameter)
    # Identity uses the population second moment of the Monte Carlo cloud.
    pop_var = float(np.mean((estimates - np.mean(estimates)) ** 2))
    return {
        "estimates": estimates,
        "mean_estimate": float(np.mean(estimates)),
        "bias": bias,
        "variance": var,
        "mse": mse,
        "mse_from_identity": pop_var + bias**2,
    }


def mse_identity_holds(estimates, true_parameter, atol=1e-10):
    """Numerically verify MSE = Var + Bias^2 on a Monte Carlo sample."""
    bias = empirical_bias(estimates, true_parameter)
    var = empirical_variance(estimates)
    mse = empirical_mse(estimates, true_parameter)
    # empirical_variance uses ddof=1, while MSE uses the raw second moment,
    # so the identity is exact only in the population (ddof=0) sense.
    # Use the population variance of the estimates for a fair check:
    pop_var = float(np.mean((np.asarray(estimates) - np.mean(estimates)) ** 2))
    reconstructed = pop_var + bias**2
    return {
        "mse": mse,
        "var_ddof0": pop_var,
        "var_ddof1": var,
        "bias": bias,
        "bias_squared": bias**2,
        "var_plus_bias_sq": reconstructed,
        "abs_gap": abs(mse - reconstructed),
        "holds": abs(mse - reconstructed) <= atol,
    }


def consistency_trajectories(estimator_fn, sampler_fn, true_parameter, sample_sizes, n_reps=200, seed=42):
    """
    For each n, collect n_reps estimates. Used to visualize consistency:
    the cloud of estimates should collapse toward the true parameter.
    """
    rng = np.random.default_rng(seed)
    trajectories = {}
    for n in sample_sizes:
        estimates = []
        for _ in range(n_reps):
            child_seed = int(rng.integers(0, 2**31 - 1))
            sample = sampler_fn(size=n, seed=child_seed)
            estimates.append(estimator_fn(sample))
        trajectories[n] = np.asarray(estimates, dtype=float)
    return trajectories
