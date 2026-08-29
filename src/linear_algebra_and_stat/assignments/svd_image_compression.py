import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add src to path to allow importing core
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.linear_algebra_and_stat.core.decomposition import SVDImageCompressor

DEFAULT_IMAGE_PATH = os.path.join(PROJECT_ROOT, 'data', 'test_image.jpg')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

def ensure_test_image_exists(image_path: str = DEFAULT_IMAGE_PATH):
    """
    Ensures that a test image exists in the data folder.
    If it doesn't exist, downloads/generates a standard sample image.
    """
    if not os.path.exists(image_path):
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        # Generate a synthetic RGB image if missing
        x, y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
        img_r = np.sin(x * 10) + np.cos(y * 10) + x**2 + y**2
        img_g = np.cos(x * 10) + np.sin(y * 10) + x**2 + y**2
        img_b = np.sin(x * 5) + np.cos(y * 15) + x**2 + y**2
        
        img = np.stack([img_r, img_g, img_b], axis=-1)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        Image.fromarray(img).save(image_path)

def load_image(color=False):
    ensure_test_image_exists()
    img = Image.open(DEFAULT_IMAGE_PATH)
    if not color:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    return np.asarray(img, dtype=np.float64)

def run_grayscale_experiment():
    print("--- SVD Experiment ---")
    img = load_image(color=False)
    compressor = SVDImageCompressor().fit(img)
    
    k_values = [1, 5, 10, 20, 50, 100]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, k in enumerate(k_values):
        compressed = compressor.compress(k)
        stats = compressor.memory_stats(k)
        
        # Calculate actual MSE
        actual_mse = np.mean((img - compressed)**2)
        
        # Theoretical MSE (sum of discarded singular values squared / total pixels)
        discarded_sq = np.sum(compressor.S[k:] ** 2)
        theoretical_mse = discarded_sq / (img.shape[0] * img.shape[1])
        
        print(f"Rank k={k:3d} | Size Reduction: {stats['reduction_pct']:6.2f}% | MSE: {actual_mse:8.2f} (Theoretical: {theoretical_mse:8.2f})")
        
        if idx < len(axes):
            axes[idx].imshow(compressed, cmap='gray')
            axes[idx].set_title(f"Rank {k}\n{stats['reduction_pct']:.1f}% reduction")
            axes[idx].axis('off')
            
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'svd_grayscale_progressive.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved grayscale progressive reconstruction to {save_path}\n")

def run_color_experiment():
    print("--- Color SVD Experiment ---")
    img = load_image(color=True)
    compressor = SVDImageCompressor().fit(img)
    
    k_values = [1, 5, 10, 20, 50, 150]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, k in enumerate(k_values):
        compressed = compressor.compress(k).astype(np.uint8)
        stats = compressor.memory_stats(k)
        
        actual_mse = np.mean((img - compressed)**2)
        print(f"Rank k={k:3d} | Size Reduction: {stats['reduction_pct']:6.2f}% | MSE: {actual_mse:8.2f}")
        
        if idx < len(axes):
            axes[idx].imshow(compressed)
            axes[idx].set_title(f"Rank {k}\n{stats['reduction_pct']:.1f}% reduction")
            axes[idx].axis('off')
            
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'svd_color_progressive.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved color progressive reconstruction to {save_path}\n")

def interactive_svd_reconstruction(img):
    import ipywidgets as widgets
    from IPython.display import display

    compressor = SVDImageCompressor().fit(img)
    max_k = min(100, compressor.shape[0], compressor.shape[1])

    def get_segment(start, end):
        """Sum the SVD components from start through end."""
        segment = np.zeros(compressor.shape, dtype=np.float64)

        for rank in range(start, end + 1):
            segment += compressor.get_rank_1_term(rank - 1)

        return segment

    def prepare_image(image):
        """Prepare an image for display."""
        if compressor.is_color:
            return np.clip(image, 0, 255).astype(np.uint8)

        return np.clip(image, 0, 255)

    def show_image(ax, image, title, equation):
        """Display an SVD image with its title and equation."""
        image = prepare_image(image)

        if compressor.is_color:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap='gray', vmin=0, vmax=255)

        ax.set_title(title)
        ax.axis('off')
        ax.set_xlabel(equation, fontsize=11)

    left_start = widgets.IntSlider(
        value=1,
        min=1,
        max=max_k,
        step=1,
        description='Start:',
        continuous_update=False
    )

    left_end = widgets.IntSlider(
        value=5,
        min=1,
        max=max_k,
        step=1,
        description='End:',
        continuous_update=False
    )

    aggregate_rank = widgets.IntSlider(
        value=5,
        min=1,
        max=max_k,
        step=1,
        description='Rank:',
        continuous_update=False
    )

    aggregate_prev = widgets.Button(
        description='◀ -1',
        button_style='info'
    )

    aggregate_next = widgets.Button(
        description='+1 ▶',
        button_style='info'
    )

    output = widgets.Output()

    def update_plot(*args):
        start = left_start.value
        end = left_end.value
        rank = aggregate_rank.value

        # Keep the selected range valid.
        if start > end:
            if args and args[0].get('owner') is left_start:
                left_end.value = start
            else:
                left_start.value = end
            return

        selected = get_segment(start, end)
        aggregate = compressor.compress(rank, clip=True)

        with output:
            output.clear_output(wait=True)

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(17, 5.5)
            )

            if start == end:
                title = f"Rank {start}"
                equation = rf"$A_{{{start}}}$"
            else:
                title = f"Ranks {start}–{end}"
                equation = rf"$\sum_{{k={start}}}^{{{end}}} A_k$"

            show_image(
                axes[0],
                selected,
                title,
                equation
            )

            show_image(
                axes[1],
                aggregate,
                f"Reconstruction — Rank {rank}",
                rf"$A_{{{rank}}} = \sum_{{k=1}}^{{{rank}}} A_k$"
            )

            if compressor.is_color:
                singular_values = compressor.S[0][:max_k]
            else:
                singular_values = compressor.S[:max_k]

            axes[2].bar(
                range(1, max_k + 1),
                singular_values,
                alpha=0.6
            )

            # Selected range
            axes[2].axvspan(
                start - 0.5,
                end + 0.5,
                alpha=0.25
            )

            # Current reconstruction rank
            axes[2].axvline(
                rank,
                linestyle='--',
                linewidth=2
            )

            axes[2].set_title("Singular Values")
            axes[2].set_xlabel("Rank")
            axes[2].set_ylabel("Magnitude")

            plt.tight_layout()
            plt.show()

    def aggregate_prev_click(button):
        if aggregate_rank.value > 1:
            aggregate_rank.value -= 1

    def aggregate_next_click(button):
        if aggregate_rank.value < max_k:
            aggregate_rank.value += 1

    left_start.observe(update_plot, names='value')
    left_end.observe(update_plot, names='value')
    aggregate_rank.observe(update_plot, names='value')

    aggregate_prev.on_click(aggregate_prev_click)
    aggregate_next.on_click(aggregate_next_click)

    left_controls = widgets.VBox([
        widgets.HTML("<b>Selected Components</b>"),
        left_start,
        left_end
    ])

    right_controls = widgets.VBox([
        widgets.HTML("<b>Complete Reconstruction</b>"),
        aggregate_rank,
        widgets.HBox([
            aggregate_prev,
            aggregate_next
        ])
    ])

    controls = widgets.HBox(
        [left_controls, right_controls],
        layout=widgets.Layout(
            justify_content='space-between',
            width='100%'
        )
    )

    # Initial display
    update_plot()

    display(
        widgets.VBox([
            controls,
            output
        ])
    )

if __name__ == "__main__":
    run_grayscale_experiment()
    run_color_experiment()
