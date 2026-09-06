import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

def interactive_mle_experiment():
    """
    Interactive Maximum Likelihood Laboratory.
    Generates samples from a Gaussian and visualizes the likelihood surface.
    """
    np.random.seed(42)
    
    # Ground truth parameters
    true_mu = 5.0
    true_sigma = 2.0
    n_samples = 30
    
    # Generate observed data
    observations = np.random.normal(true_mu, true_sigma, n_samples)
    
    # Analytical MLE 
    mle_mu = np.mean(observations)
    mle_sigma = np.std(observations, ddof=0)
    
    # Pre-compute candidate spaces for likelihood curves
    mu_candidates = np.linspace(true_mu - 4, true_mu + 4, 200)
    sigma_candidates = np.linspace(0.5, true_sigma + 4, 200)

    # Set up interactive widgets
    mu_slider = widgets.FloatSlider(
        value=true_mu, min=1.0, max=9.0, step=0.1, 
        description='$\\mu$ (mean):', continuous_update=False
    )
    
    sigma_slider = widgets.FloatSlider(
        value=true_sigma, min=0.5, max=6.0, step=0.1, 
        description='$\\sigma$ (std):', continuous_update=False
    )

    output = widgets.Output()

    def update_plot(*args):
        mu_val = mu_slider.value
        sigma_val = sigma_slider.value
        
        with output:
            output.clear_output(wait=True)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # --- 1. Data and Candidate Distribution ---
            axes[0].hist(observations, bins=10, density=True, alpha=0.5, color='gray', label='Observations')
            x_axis = np.linspace(0, 10, 200)
            candidate_pdf = (1.0 / (sigma_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_axis - mu_val) / sigma_val)**2)
            
            axes[0].plot(x_axis, candidate_pdf, color='blue', lw=2, label=f'Candidate $N({mu_val:.1f}, {sigma_val**2:.2f})$')
            axes[0].set_title('Observations vs. Candidate Distribution')
            axes[0].legend()
            
            # --- 2. Likelihood surface w.r.t mu (keeping sigma fixed at slider value) ---
            likelihoods_mu = []
            for m in mu_candidates:
                probs = (1.0 / (sigma_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((observations - m) / sigma_val)**2)
                likelihoods_mu.append(np.prod(probs))
                
            axes[1].plot(mu_candidates, likelihoods_mu, color='green')
            axes[1].axvline(mle_mu, color='red', linestyle='--', label=f'Analytical MLE $\\mu = {mle_mu:.2f}$')
            axes[1].axvline(mu_val, color='blue', linestyle='-', label=f'Current $\\mu = {mu_val:.2f}$')
            axes[1].set_title(f'Likelihood $L(\\mu \\mid X, \\sigma={sigma_val:.1f})$')
            axes[1].set_xlabel('$\\mu$')
            axes[1].set_ylabel('Likelihood')
            axes[1].legend()
            
            # --- 3. Log-Likelihood surface w.r.t sigma (keeping mu fixed at slider value) ---
            log_likelihoods_sigma = []
            for s in sigma_candidates:
                probs = (1.0 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((observations - mu_val) / s)**2)
                # Add small epsilon to avoid log(0) if any probs underflow
                log_L = np.sum(np.log(np.maximum(probs, 1e-300)))
                log_likelihoods_sigma.append(log_L)
                
            axes[2].plot(sigma_candidates, log_likelihoods_sigma, color='purple')
            axes[2].axvline(mle_sigma, color='red', linestyle='--', label=f'Analytical MLE $\\sigma = {mle_sigma:.2f}$')
            axes[2].axvline(sigma_val, color='blue', linestyle='-', label=f'Current $\\sigma = {sigma_val:.2f}$')
            axes[2].set_title(f'Log-Likelihood $\\log L(\\sigma \\mid X, \\mu={mu_val:.1f})$')
            axes[2].set_xlabel('$\\sigma$')
            axes[2].set_ylabel('Log-Likelihood')
            axes[2].legend()
            
            plt.tight_layout()
            plt.show()

    mu_slider.observe(update_plot, names='value')
    sigma_slider.observe(update_plot, names='value')
    
    controls = widgets.VBox([
        widgets.HTML("<b>Adjust the candidate Gaussian parameters to maximize the likelihood of the observed data:</b>"),
        widgets.HBox([mu_slider, sigma_slider])
    ])
    
    update_plot()
    display(widgets.VBox([controls, output]))

def run_mle_experiment():
    """
    Fallback for non-interactive execution.
    """
    print("For interactive MLE visualization, run interactive_mle_experiment() in a Jupyter Notebook environment.")
    
if __name__ == "__main__":
    run_mle_experiment()
