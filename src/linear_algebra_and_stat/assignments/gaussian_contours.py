import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

def plot_gaussian_contour(mu, Sigma, ax, title):
    """
    Plots the contour of a 2D Gaussian.
    """
    x, y = np.mgrid[-4:4:.05, -4:4:.05]
    pos = np.dstack((x, y))
    
    try:
        inv_Sigma = np.linalg.inv(Sigma)
        det_Sigma = np.linalg.det(Sigma)
        
        if det_Sigma <= 0:
            ax.text(0, 0, "Invalid Covariance Matrix (Not Positive Definite)", 
                    ha='center', va='center', color='red')
            return
            
        # Calculate PDF
        diff = pos - mu
        quadratic_form = np.einsum('...j,jk,...k->...', diff, inv_Sigma, diff)
        norm_const = 1.0 / (2 * np.pi * np.sqrt(det_Sigma))
        z = norm_const * np.exp(-0.5 * quadratic_form)
        
        ax.contourf(x, y, z, levels=10, cmap='Blues')
        ax.contour(x, y, z, levels=10, colors='k', alpha=0.3)
        
        # Plot principal axes (eigenvectors scaled by eigenvalues)
        vals, vecs = np.linalg.eigh(Sigma)
        for val, vec in zip(vals, vecs.T):
            if val > 0:
                start, end = mu, mu + np.sqrt(val) * vec * 2 # scale for visibility
                ax.annotate('', xy=end, xytext=start,
                            arrowprops=dict(facecolor='red', edgecolor='red', width=2, headwidth=8))
                
    except np.linalg.LinAlgError:
        ax.text(0, 0, "Singular Covariance Matrix", ha='center', va='center', color='red')
        
    ax.set_title(title)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

def interactive_multivariate_gaussians():
    """
    Interactive visualization of how covariance matrices affect 2D Gaussians.
    """
    mu = np.array([0, 0])
    
    var_x_slider = widgets.FloatSlider(value=1.0, min=0.1, max=4.0, step=0.1, description='$\\sigma_x^2$:', continuous_update=False)
    var_y_slider = widgets.FloatSlider(value=1.0, min=0.1, max=4.0, step=0.1, description='$\\sigma_y^2$:', continuous_update=False)
    rho_slider = widgets.FloatSlider(value=0.0, min=-0.99, max=0.99, step=0.05, description='$\\rho$ (corr):', continuous_update=False)
    
    output = widgets.Output()
    
    def update_plot(*args):
        var_x = var_x_slider.value
        var_y = var_y_slider.value
        rho = rho_slider.value
        
        cov_xy = rho * np.sqrt(var_x) * np.sqrt(var_y)
        Sigma = np.array([[var_x, cov_xy], 
                          [cov_xy, var_y]])
        
        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            title = f"Covariance Matrix:\\n[[{var_x:.2f}, {cov_xy:.2f}],\\n [{cov_xy:.2f}, {var_y:.2f}]]"
            plot_gaussian_contour(mu, Sigma, ax, title)
            plt.show()

    var_x_slider.observe(update_plot, names='value')
    var_y_slider.observe(update_plot, names='value')
    rho_slider.observe(update_plot, names='value')
    
    controls = widgets.VBox([
        widgets.HTML("<b>Adjust Variances and Correlation to see the Mahalanobis contours transform:</b>"),
        widgets.HBox([var_x_slider, var_y_slider]),
        rho_slider
    ])
    
    update_plot()
    display(widgets.VBox([controls, output]))

def visualize_multivariate_gaussians():
    print("For interactive contour visualization, run interactive_multivariate_gaussians() in a Jupyter Notebook.")

if __name__ == "__main__":
    visualize_multivariate_gaussians()
