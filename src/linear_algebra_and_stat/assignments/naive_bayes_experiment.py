import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        
        for c in self.classes:
            X_c = X[y == c]
            # Small epsilon added to variance to prevent division by zero
            self.parameters[c] = {
                'mean': X_c.mean(axis=0),
                'var': X_c.var(axis=0) + 1e-4, 
                'prior': len(X_c) / len(X),
                'count': len(X_c)
            }
            
    def _calculate_log_likelihood(self, class_val, x):
        mean = self.parameters[class_val]['mean']
        var = self.parameters[class_val]['var']
        
        # log likelihood for all 64 pixels, assuming independence
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((x - mean) ** 2) / var)
        return log_prob
    
    def predict_single(self, x, return_details=False):
        posteriors = []
        details = {}
        
        for c in self.classes:
            prior = np.log(self.parameters[c]['prior'])
            likelihood = self._calculate_log_likelihood(c, x)
            posterior = prior + likelihood
            posteriors.append(posterior)
            
            if return_details:
                details[c] = {
                    'log_prior': prior,
                    'log_likelihood': likelihood,
                    'log_posterior': posterior
                }
                
        pred_class = self.classes[np.argmax(posteriors)]
        
        if return_details:
            return pred_class, details
        return pred_class

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

def visualize_naive_bayes_features():
    """
    Visualizes the mean and variance of each pixel for every digit class.
    Also displays the class distribution to check for bias.
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    
    model = GaussianNaiveBayes()
    model.fit(X, y)
    
    # Check data distribution for bias
    print("--- Data Distribution ---")
    total = len(y)
    for c in model.classes:
        count = model.parameters[c]['count']
        pct = (count / total) * 100
        print(f"Digit {c}: {count} samples ({pct:.1f}%)")
    print("-" * 25)
    
    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    fig.suptitle("Naive Bayes Learned Parameters (Gaussian per Pixel)", fontsize=16)
    
    for c in model.classes:
        mean_img = model.parameters[c]['mean'].reshape(8, 8)
        var_img = model.parameters[c]['var'].reshape(8, 8)
        
        # Plot Mean
        ax_mean = axes[0, c]
        ax_mean.imshow(mean_img, cmap='gray_r')
        ax_mean.set_title(f"Mean (Digit {c})")
        ax_mean.axis('off')
        
        # Plot Variance
        ax_var = axes[1, c]
        im_var = ax_var.imshow(var_img, cmap='inferno') # Hot means high variance
        ax_var.set_title(f"Variance (Digit {c})")
        ax_var.axis('off')
        
    plt.tight_layout()
    plt.show()

def interactive_naive_bayes_experiment():
    """
    Demonstrate Bayesian Classification interactively, exploring correct vs wrong predictions.
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    
    # Pre-predict the test set
    y_pred = model.predict(X_test)
    
    # Find indices
    correct_indices = np.where(y_test == y_pred)[0]
    wrong_indices = np.where(y_test != y_pred)[0]
    
    # Calculate accuracy
    train_pred = model.predict(X_train)
    train_acc = np.mean(y_train == train_pred) * 100
    test_acc = np.mean(y_test == y_pred) * 100
    
    acc_html = widgets.HTML(f"<b>Model Performance:</b><br/>"
                            f"&nbsp;&nbsp;Training Accuracy: <span style='color:green'>{train_acc:.1f}%</span><br/>"
                            f"&nbsp;&nbsp;Test Accuracy: <span style='color:blue'>{test_acc:.1f}%</span><br/><br/>"
                            f"<b>Explore the test set and observe the Naive Bayes probabilities:</b>")
    
    view_mode = widgets.ToggleButtons(
        options=['Correct Predictions', 'Wrong Predictions'],
        description='View Mode:',
        button_style='info'
    )
    
    sample_slider = widgets.IntSlider(
        value=0, min=0, max=len(correct_indices) - 1, step=1, 
        description='Sample Index:', continuous_update=False
    )
    
    output = widgets.Output()
    
    def on_view_mode_change(change):
        if change.new == 'Correct Predictions':
            sample_slider.max = max(0, len(correct_indices) - 1)
        else:
            sample_slider.max = max(0, len(wrong_indices) - 1)
        sample_slider.value = 0
        update_plot()
        
    view_mode.observe(on_view_mode_change, names='value')
    
    def update_plot(*args):
        mode = view_mode.value
        current_list = correct_indices if mode == 'Correct Predictions' else wrong_indices
        
        if len(current_list) == 0:
            with output:
                output.clear_output(wait=True)
                print(f"No {mode.lower()} found!")
            return
            
        list_idx = sample_slider.value
        sample_idx = current_list[list_idx]
        
        sample_x = X_test[sample_idx]
        true_y = y_test[sample_idx]
        
        pred_y, details = model.predict_single(sample_x, return_details=True)
        
        with output:
            output.clear_output(wait=True)
            fig = plt.figure(figsize=(14, 5))
            
            # Left: Show the digit image
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(sample_x.reshape(8, 8), cmap='gray_r')
            ax1.set_title(f"True Class: {true_y} | Predicted: {pred_y}", fontsize=14, 
                          color='green' if true_y == pred_y else 'red')
            ax1.axis('off')
            
            # Right: Show the log-posterior scores
            ax2 = plt.subplot(1, 2, 2)
            classes = sorted(details.keys())
            posteriors = [details[c]['log_posterior'] for c in classes]
            
            bars = ax2.bar(classes, posteriors, color='skyblue')
            bars[pred_y].set_color('orange')
            
            ax2.set_title("Log-Posterior Score per Class", fontsize=14)
            ax2.set_xlabel("Digit Class")
            ax2.set_ylabel("Log Posterior (closer to 0 is better)")
            ax2.set_xticks(classes)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.0f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, -15),  # 15 points vertical offset
                             textcoords="offset points",
                             ha='center', va='top', color='black', fontsize=9)
            
            plt.tight_layout()
            plt.show()
            
            # Print tabular details
            print(f"\\n--- Calculation Details (Dataset Index: {sample_idx}) ---")
            print("Class | Log Prior | Log Likelihood | Log Posterior")
            print("-" * 55)
            for c in classes:
                d = details[c]
                marker = "<-- Predicted" if c == pred_y else ""
                print(f"  {c}   | {d['log_prior']:9.2f} | {d['log_likelihood']:14.2f} | {d['log_posterior']:13.2f} {marker}")

    sample_slider.observe(update_plot, names='value')
    
    controls = widgets.VBox([
        acc_html,
        view_mode,
        sample_slider
    ])
    
    update_plot()
    display(widgets.VBox([controls, output]))

def interactive_canvas_naive_bayes_experiment():
    """
    An interactive 8x8 drawing canvas to test the Naive Bayes model.
    Clicks cycle through pixel intensities (0 -> 4 -> 8 -> 12 -> 16).
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    model = GaussianNaiveBayes()
    model.fit(X, y)
    
    # Store intensity in a dictionary mapping button to intensity
    btn_intensities = {}
    
    grid_buttons = []
    rows = []
    
    # Helper to convert 0-16 intensity to a hex color for the button
    def get_color_from_intensity(intensity):
        # 0 = white, 16 = dark gray/black
        # Invert intensity for RGB (0 -> 255, 16 -> 50)
        c = int(255 - (intensity / 16.0) * 200)
        return f'#{c:02x}{c:02x}{c:02x}'
    
    def on_btn_click(btn):
        # Increment intensity (0 -> 4 -> 8 -> 12 -> 16 -> 0)
        current = btn_intensities[btn]
        new_val = (current + 4) % 20
        btn_intensities[btn] = new_val
        btn.style.button_color = get_color_from_intensity(new_val)
        
    for i in range(8):
        row_buttons = []
        for j in range(8):
            btn = widgets.Button(
                description='',
                layout=widgets.Layout(width='35px', height='35px', margin='1px')
            )
            btn_intensities[btn] = 0.0
            btn.style.button_color = get_color_from_intensity(0)
            btn.on_click(on_btn_click)
            
            row_buttons.append(btn)
            grid_buttons.append(btn)
        rows.append(widgets.HBox(row_buttons))
    
    canvas = widgets.VBox(rows, layout=widgets.Layout(border='2px solid #555', width='max-content', padding='2px'))
    
    predict_btn = widgets.Button(description="Predict Digit", button_style='success')
    clear_btn = widgets.Button(description="Clear Canvas", button_style='warning')
    
    output = widgets.Output()
    
    def on_clear(b):
        for btn in grid_buttons:
            btn_intensities[btn] = 0.0
            btn.style.button_color = get_color_from_intensity(0)
        with output:
            output.clear_output()
            
    def on_predict(b):
        # Build image from button intensities
        img = np.array([btn_intensities[btn] for btn in grid_buttons])
        
        pred_y, details = model.predict_single(img, return_details=True)
        
        with output:
            output.clear_output(wait=True)
            fig = plt.figure(figsize=(10, 4))
            ax2 = plt.subplot(1, 1, 1)
            
            classes = sorted(details.keys())
            posteriors = [details[c]['log_posterior'] for c in classes]
            
            bars = ax2.bar(classes, posteriors, color='skyblue')
            bars[pred_y].set_color('orange')
            
            ax2.set_title(f"Predicted Digit: {pred_y}", fontsize=18, color='green')
            ax2.set_xlabel("Digit Class")
            ax2.set_ylabel("Log Posterior")
            ax2.set_xticks(classes)
            
            plt.tight_layout()
            plt.show()

    predict_btn.on_click(on_predict)
    clear_btn.on_click(on_clear)
    
    controls = widgets.VBox([
        widgets.HTML("<b>Draw a digit by clicking pixels. Each click makes the pixel darker!</b>"),
        canvas,
        widgets.HTML("<br/>"),
        widgets.HBox([predict_btn, clear_btn])
    ])
    
    display(widgets.HBox([controls, output]))

def run_naive_bayes_experiment():
    print("For interactive Naive Bayes visualization, run interactive_naive_bayes_experiment() or interactive_canvas_naive_bayes_experiment() in a Jupyter Notebook.")

if __name__ == "__main__":
    run_naive_bayes_experiment()
