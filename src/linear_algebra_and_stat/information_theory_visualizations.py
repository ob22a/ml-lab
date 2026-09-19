"""Plotting and interactive laboratories for Notebook 06."""

import math

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from IPython.display import display
from sklearn.feature_selection import mutual_info_regression
from . import information_theory as it


def plot_probability_vs_information():
    p = np.geomspace(0.001, 1, 500)
    info = it.self_information(p)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(p, info, color="steelblue", lw=2)
    examples = [(1, "certain: 0 bits"), (0.5, "fair coin: 1 bit"), (0.125, "1 in 8: 3 bits"), (0.01, "1 in 100: 6.64 bits")]
    for probability, label in examples:
        value = it.self_information(probability)
        ax.scatter(probability, value, color="red", zorder=3)
        ax.annotate(label, (probability, value), xytext=(8, 8), textcoords="offset points")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Probability of observed outcome (rarer to the right)")
    ax.set_ylabel("Self-information (bits)")
    ax.set_title(r"Rare outcomes carry more information: $I(x)=-\log_2p(x)$")
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_repetition_code_lengths(probabilities=(0.5, 0.8, 0.95), n=100):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for p in probabilities:
        h = it.entropy([p, 1 - p])
        axes[0].plot(np.arange(1, n + 1), np.arange(1, n + 1) * h, label=f"p={p}, H={h:.2f}")
    axes[0].plot(np.arange(1, n + 1), np.arange(1, n + 1), "k--", label="fair source: n bits")
    axes[0].set_title("Theoretical minimum average description length")
    axes[0].set_xlabel("Number of symbols")
    axes[0].set_ylabel("Expected bits (nH)")
    axes[0].legend(fontsize=8)

    p_grid = np.linspace(0.001, 0.999, 400)
    h_grid = np.array([it.entropy([p, 1 - p]) for p in p_grid])
    axes[1].plot(p_grid, h_grid, color="purple", lw=2)
    axes[1].axvline(0.5, color="red", ls="--", label="maximum at p=0.5")
    axes[1].axhline(1, color="black", ls=":")
    axes[1].set_title("Binary entropy: repetition lowers average bits")
    axes[1].set_xlabel("P(1)")
    axes[1].set_ylabel("bits/symbol")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_entropy_distributions(distributions):
    fig, axes = plt.subplots(1, len(distributions), figsize=(4.2 * len(distributions), 4), sharey=True)
    if len(distributions) == 1:
        axes = [axes]
    for ax, (name, p) in zip(axes, distributions.items()):
        p = np.asarray(p)
        ax.bar(np.arange(len(p)), p, color="steelblue", alpha=0.8)
        ax.set_title(f"{name}\nH={it.entropy(p):.3f} bits")
        ax.set_xlabel("Outcome")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Probability")
    plt.tight_layout()
    plt.show()


def interactive_entropy():
    def update(p1, p2, p3):
        raw = np.array([p1, p2, p3, 1.0])
        p = raw / raw.sum()
        info = it.self_information(p)
        h = it.entropy(p)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        axes[0].bar(range(4), p, color="steelblue")
        axes[0].set_title(f"Valid normalized distribution\nH={h:.3f}, max=2 bits")
        axes[0].set_ylabel("Probability")
        axes[1].bar(range(4), info, color="orange")
        axes[1].set_title("Self-information of each outcome")
        axes[1].set_ylabel("bits")
        for ax in axes:
            ax.set_xticks(range(4), ["A", "B", "C", "D"])
            ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    display(widgets.interactive(
        update,
        p1=widgets.FloatSlider(value=1, min=0.05, max=5, step=0.05, description="weight A", continuous_update=False),
        p2=widgets.FloatSlider(value=1, min=0.05, max=5, step=0.05, description="weight B", continuous_update=False),
        p3=widgets.FloatSlider(value=1, min=0.05, max=5, step=0.05, description="weight C", continuous_update=False),
    ))


def plot_joint_table(joint, x_labels=None, y_labels=None, title="Joint distribution"):
    joint = it.validate_joint(joint)
    x_labels = x_labels or [f"x{i}" for i in range(joint.shape[0])]
    y_labels = y_labels or [f"y{i}" for i in range(joint.shape[1])]
    px, py = it.marginals(joint)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    im = axes[0].imshow(joint, cmap="Blues", vmin=0, vmax=max(0.01, joint.max()))
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            axes[0].text(j, i, f"{joint[i,j]:.2f}", ha="center", va="center")
    axes[0].set_xticks(range(len(y_labels)), y_labels)
    axes[0].set_yticks(range(len(x_labels)), x_labels)
    axes[0].set_xlabel("Y")
    axes[0].set_ylabel("X")
    axes[0].set_title(f"{title}\nH(X,Y)={it.joint_entropy(joint):.3f}")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].bar(x_labels, px, color="steelblue")
    axes[1].set_title(f"Marginal P(X)\nH(X)={it.entropy(px):.3f}")
    axes[2].bar(y_labels, py, color="orange")
    axes[2].set_title(f"Marginal P(Y)\nH(Y)={it.entropy(py):.3f}")
    for ax in axes[1:]:
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_conditional_cases(cases):
    fig, axes = plt.subplots(1, len(cases), figsize=(4.4 * len(cases), 4))
    if len(cases) == 1:
        axes = [axes]
    for ax, (name, joint) in zip(axes, cases.items()):
        joint = it.validate_joint(joint)
        im = ax.imshow(joint, cmap="Blues", vmin=0, vmax=max(v.max() for v in cases.values()))
        h_y = it.entropy(joint.sum(axis=0))
        h_cond = it.conditional_entropy(joint)
        mi = it.mutual_information(joint)
        ax.set_title(f"{name}\nH(Y)={h_y:.2f}, H(Y|X)={h_cond:.2f}, I={mi:.2f}")
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        for i in range(joint.shape[0]):
            for j in range(joint.shape[1]):
                ax.text(j, i, f"{joint[i,j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.show()


def dependence_samples(kind="linear", noise=0.3, n=700, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    if kind == "independent":
        y = rng.normal(0, 1, n)
    elif kind == "linear":
        y = 1.5 * x + rng.normal(0, noise, n)
    elif kind == "quadratic":
        y = x**2 + rng.normal(0, noise, n)
    elif kind == "sinusoidal":
        y = np.sin(3 * x) + rng.normal(0, noise, n)
    else:
        raise ValueError(kind)
    corr = float(np.corrcoef(x, y)[0, 1])
    mi = float(mutual_info_regression(x[:, None], y, random_state=seed)[0])
    return x, y, corr, mi


def plot_dependence_laboratory(noise=0.25):
    kinds = ["independent", "linear", "quadratic", "sinusoidal"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, kind in zip(axes, kinds):
        x, y, corr, mi = dependence_samples(kind, noise=noise)
        ax.scatter(x, y, s=10, alpha=0.45, color="steelblue")
        ax.set_title(f"{kind}\nPearson r={corr:.3f}, MI≈{mi:.3f} nats")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_mi_vs_noise(kind="quadratic", noise_grid=None):
    noise_grid = np.linspace(0.02, 2.0, 25) if noise_grid is None else np.asarray(noise_grid)
    correlations, mis = [], []
    for i, noise in enumerate(noise_grid):
        _, _, corr, mi = dependence_samples(kind, noise=noise, seed=100 + i)
        correlations.append(abs(corr))
        mis.append(mi)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(noise_grid, mis, "o-", label="estimated MI (nats)", color="purple")
    ax.plot(noise_grid, correlations, "o-", label="|Pearson r|", color="orange")
    ax.set_xlabel("Noise standard deviation")
    ax.set_title(f"Noise erases information in a {kind} relationship")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def interactive_mutual_information():
    def update(kind, noise):
        x, y, corr, mi = dependence_samples(kind, noise=noise)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(x, y, s=12, alpha=0.45)
        ax.set_title(f"{kind}: r={corr:.3f}, estimated MI={mi:.3f} nats")
        ax.grid(True, alpha=0.3)
        plt.show()

    display(widgets.interactive(
        update,
        kind=widgets.Dropdown(options=["independent", "linear", "quadratic", "sinusoidal"], value="quadratic"),
        noise=widgets.FloatSlider(min=0.02, max=2, step=0.05, value=0.25, continuous_update=False),
    ))


def plot_language_statistics(stats_by_language):
    names = list(stats_by_language)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for name in names:
        info = stats_by_language[name]["characters"]
        order = np.argsort(info["probabilities"])[-20:][::-1]
        symbols = [info["symbols"][i].replace(" ", "␠") for i in order]
        print(f"\n{name}: top-20 characters (␠ means space ' '): {', '.join(symbols)}")
        axes[0, 0].plot(range(len(order)), info["probabilities"][order], "o-", label=name)
    axes[0, 0].set_xticks(range(20), range(1, 21))
    axes[0, 0].set_title("Top-20 character probabilities (ranked)")
    axes[0, 0].set_xlabel("Frequency rank")
    axes[0, 0].legend()

    entropy_vals = [stats_by_language[n]["characters"]["entropy"] for n in names]
    conditional_vals = [stats_by_language[n]["adjacent"]["H_next_given_current"] for n in names]
    mi_vals = [stats_by_language[n]["adjacent"]["MI_adjacent"] for n in names]
    axes[0, 1].bar(names, entropy_vals, color="steelblue")
    axes[0, 1].set_title("Marginal character entropy H(C)")
    axes[1, 0].bar(names, conditional_vals, color="orange")
    axes[1, 0].set_title(r"Next-character uncertainty $H(C_{i+1}|C_i)$")
    axes[1, 1].bar(names, mi_vals, color="purple")
    axes[1, 1].set_title(r"Adjacent-character information $I(C_i;C_{i+1})$")
    for ax in axes.ravel():
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_plaintext_cipher_comparison(plain, cipher):
    p_info = it.character_information(plain)
    c_info = it.character_information(cipher)
    p_adj = it.adjacent_character_information(plain)
    c_adj = it.adjacent_character_information(cipher)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, text, name, color in [(axes[0], plain, "Plaintext", "steelblue"), (axes[1], cipher, "Substitution ciphertext", "orange")]:
        symbols, p = it.empirical_distribution(text)
        p = np.sort(p)[::-1]
        ax.bar(range(min(30, len(p))), p[:30], color=color)
        ax.set_title(f"{name}\nH(C)={it.entropy(p):.3f} bits")
        ax.set_xlabel("Character frequency rank")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        f"A one-to-one relabeling preserves entropy and adjacency MI: "
        f"{p_adj['MI_adjacent']:.3f} vs {c_adj['MI_adjacent']:.3f} bits",
        y=1.03,
    )
    plt.tight_layout()
    plt.show()


def plot_information_gain(results):
    names = list(results)
    h_parent = [results[n]["H_target"] for n in names]
    h_after = [results[n]["H_target_given_feature"] for n in names]
    ig = [results[n]["information_gain"] for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].bar(x - 0.18, h_parent, width=0.36, label="H(Y) before", color="grey")
    axes[0].bar(x + 0.18, h_after, width=0.36, label="H(Y|X) after", color="steelblue")
    axes[0].set_xticks(x, names)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_title("Parent uncertainty → weighted child uncertainty")
    axes[0].legend()
    axes[1].bar(names, ig, color="orange")
    axes[1].set_xticklabels(range(len(names)), names, rotation=45, ha='right')
    axes[1].set_title("Information gain = uncertainty removed")
    best = max(results, key=lambda name: results[name]["information_gain"])
    joint = results[best]["joint"]
    branch_totals = joint.sum(axis=1, keepdims=True)
    branch_class_p = np.divide(joint, branch_totals, out=np.zeros_like(joint), where=branch_totals > 0)
    width = 0.35
    positions = np.arange(joint.shape[0])
    axes[2].bar(positions - width / 2, branch_class_p[:, 0], width, label="low value", color="steelblue")
    axes[2].bar(positions + width / 2, branch_class_p[:, 1], width, label="high value", color="orange")
    axes[2].set_xticks(positions, [f"branch {value}" for value in results[best]["feature_values"]])
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f"Best split: {best}\nclass distribution inside each branch")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_ylabel("bits")
        ax.grid(True, axis="y", alpha=0.3)
    axes[2].set_ylabel("class probability")
    plt.tight_layout()
    plt.show()


def plot_cross_entropy_penalty():
    p = np.geomspace(0.001, 1, 500)
    loss = -np.log(p)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(p, loss, color="red", lw=2)
    for q in [0.9, 0.7, 0.5, 0.1, 0.01]:
        ax.scatter(q, -math.log(q), color="black")
        ax.annotate(f"p={q}: {-math.log(q):.2f}", (q, -math.log(q)), xytext=(6, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Probability assigned to the true outcome (confidently wrong → right)")
    ax.set_ylabel("Loss (nats)")
    ax.set_title(r"Cross-entropy's logarithmic penalty: $-\log q_{\rm true}$")
    ax.grid(True, alpha=0.3)
    plt.show()


def interactive_cross_entropy():
    def update(probability):
        loss = -math.log(probability)
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.barh(["true outcome"], [loss], color="red")
        ax.set_xlim(0, 7)
        ax.set_title(f"q(true)={probability:.3f} → -log q={loss:.3f} nats")
        ax.grid(True, axis="x", alpha=0.3)
        plt.show()
    display(widgets.interactive(
        update,
        probability=widgets.FloatSlider(min=0.001, max=0.999, step=0.001, value=0.7, continuous_update=False),
    ))


def plot_binary_cross_entropy():
    q = np.linspace(0.001, 0.999, 500)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(q, it.binary_cross_entropy(1, q), label=r"true $y=1$: $-\ln\hat y$", color="steelblue")
    ax.plot(q, it.binary_cross_entropy(0, q), label=r"true $y=0$: $-\ln(1-\hat y)$", color="orange")
    ax.set_xlabel(r"Predicted probability $\hat y=P(Y=1)$")
    ax.set_ylabel("loss (nats)")
    ax.set_ylim(0, 7)
    ax.set_title("Binary cross-entropy punishes confident errors without bound")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_equal_accuracy_models(y, qa, qb):
    y = np.asarray(y)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, q, name, color in [(axes[0], qa, "A: moderate confidence", "steelblue"), (axes[1], qb, "B: catastrophic wrong confidence", "orange")]:
        correct_prob = np.where(y == 1, q, 1 - q)
        accuracy = np.mean((q >= 0.5) == y)
        ce = np.mean(it.binary_cross_entropy(y, q))
        ax.bar(range(len(y)), -np.log(correct_prob), color=np.where(correct_prob >= 0.5, color, "red"))
        ax.set_title(f"{name}\naccuracy={accuracy:.2f}, cross-entropy={ce:.3f}")
        ax.set_xlabel("Example")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("per-example negative log probability")
    plt.tight_layout()
    plt.show()


def plot_kl_experiment(p, q_models):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(p))
    width = 0.8 / (len(q_models) + 1)
    axes[0].bar(x - 0.4 + width / 2, p, width, label="true P", color="black")
    metrics = []
    for i, (name, q) in enumerate(q_models.items(), 1):
        axes[0].bar(x - 0.4 + width * (i + 0.5), q, width, label=name)
        metrics.append((name, it.entropy(p), it.cross_entropy(p, q), it.kl_divergence(p, q)))
    axes[0].set_title("True distribution and model distributions")
    axes[0].set_xticks(x)
    axes[0].legend(fontsize=8)
    names = [m[0] for m in metrics]
    axes[1].bar(names, [m[1] for m in metrics], label="H(P)", color="grey")
    axes[1].bar(names, [m[3] for m in metrics], bottom=[m[1] for m in metrics], label="KL(P||Q)", color="orange")
    axes[1].scatter(names, [m[2] for m in metrics], color="red", label="H(P,Q)", zorder=3)
    axes[1].set_title("Cross-entropy = unavoidable entropy + model mismatch")
    axes[1].set_ylabel("bits")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    return metrics
