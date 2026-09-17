import numpy as np


def sample_normal(mu=0, sigma=1, size=1, seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mu, scale=sigma, size=size)


def sample_uniform(low=0, high=1, size=1, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=size)


def sample_exponential(scale=1.0, size=1, seed=None):
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=scale, size=size)


def sample_skewed(sigma=1.0, size=1, seed=None):
    """Highly skewed log-normal population."""
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=0.0, sigma=sigma, size=size)


def make_synthetic_populations(n_pop=100_000, seed=42):
    """
    Four populations used throughout the sampling / CLT laboratory.

    Uniform bounds are chosen so Var ≈ 1, matching the Normal(0, 1)
    and Exponential(scale=1) populations for a fair SE comparison.
    """
    return {
        "Normal": sample_normal(mu=0.0, sigma=1.0, size=n_pop, seed=seed),
        "Uniform": sample_uniform(low=-np.sqrt(3), high=np.sqrt(3), size=n_pop, seed=seed + 1),
        "Exponential": sample_exponential(scale=1.0, size=n_pop, seed=seed + 2),
        "Skewed (Log-Normal)": sample_skewed(sigma=1.0, size=n_pop, seed=seed + 3),
    }


def population_moments(data):
    """Treat `data` as a finite empirical population (divide by N, not N-1)."""
    data = np.asarray(data, dtype=float)
    mu = float(np.mean(data))
    # Population (not sample) moments: the array *is* the population.
    sigma2 = float(np.mean((data - mu) ** 2))
    return mu, sigma2, float(np.sqrt(sigma2))


def repeated_samples(population_func, n_samples, sample_size, seed=None, **kwargs):
    """
    Draw `n_samples` independent samples of size `sample_size`.

    `population_func` must accept `size=` and an optional `seed=`.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        child_seed = int(rng.integers(0, 2**31 - 1))
        samples.append(population_func(size=sample_size, seed=child_seed, **kwargs))
    return np.asarray(samples)


def sample_means_from_population(population, sample_size, n_reps=2000, seed=42):
    """Resample with replacement from a finite empirical population."""
    rng = np.random.default_rng(seed)
    population = np.asarray(population)
    draws = rng.choice(population, size=(n_reps, sample_size), replace=True)
    return draws.mean(axis=1)


def summarize_sampling_distribution(population, sample_size, n_reps=2000, seed=42):
    """
    Compare the empirical sampling distribution of the mean with theory.

    Returns a dict of population moments, theoretical SE, and Monte Carlo
    estimates of the mean / variance / SE of \\bar X.
    """
    mu, sigma2, sigma = population_moments(population)
    means = sample_means_from_population(population, sample_size, n_reps=n_reps, seed=seed)
    emp_mean = float(np.mean(means))
    emp_var = float(np.var(means, ddof=1))
    emp_se = float(np.sqrt(emp_var))
    theory_var = sigma2 / sample_size
    theory_se = sigma / np.sqrt(sample_size)
    return {
        "n": sample_size,
        "n_reps": n_reps,
        "pop_mean": mu,
        "pop_var": sigma2,
        "pop_std": sigma,
        "emp_mean_of_means": emp_mean,
        "theory_var_of_mean": theory_var,
        "emp_var_of_mean": emp_var,
        "theory_se": theory_se,
        "emp_se": emp_se,
        "means": means,
    }
