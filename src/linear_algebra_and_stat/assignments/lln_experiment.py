import numpy as np
import matplotlib.pyplot as plt

def run_lln_experiment(n_samples=1000, p=0.3):
    """
    Demonstrate the Law of Large Numbers using a Bernoulli distribution.
    Shows the running empirical mean converging to the true expectation.
    """
    # True expectation E[X] = p
    expected_value = p
    
    # Generate Bernoulli samples
    samples = np.random.binomial(1, p, size=n_samples)
    
    # Compute running mean
    running_mean = np.cumsum(samples) / np.arange(1, n_samples + 1)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, n_samples + 1), running_mean, label='Running Mean $\\bar{X}_n$', color='blue')
    plt.axhline(expected_value, color='red', linestyle='dashed', label='True Expectation $E[X]$')
    
    plt.title('Law of Large Numbers (Bernoulli Distribution)')
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Mean')
    plt.xscale('log') # Log scale helps see the initial variance and later convergence
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_lln_experiment()
