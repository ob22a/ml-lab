import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import ipywidgets as widgets
from IPython.display import display

from . import hypothesis_testing
from . import sampling_utils


def plot_se_shrinkage(sigma=1.0, n_max=400):
    """Make SE ∝ 1/√n visually obvious, including the x² data-cost rule."""
    n_vals = np.arange(2, n_max + 1)
    se_vals = sigma / np.sqrt(n_vals)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_vals, se_vals, color="blue", lw=2, label=r"$SE = \sigma / \sqrt{n}$")

    markers = [
        (4, "2× lower SE → 4× data", "orange"),
        (9, "3× lower SE → 9× data", "green"),
        (100, "10× lower SE → 100× data", "red"),
    ]
    se1 = sigma / np.sqrt(1)
    ax.axhline(se1, color="grey", ls=":", alpha=0.6, label=r"$n=1$ reference ($\sigma$)")
    for factor, label, color in markers:
        n = factor
        se = sigma / np.sqrt(n)
        ax.scatter([n], [se], color=color, s=60, zorder=5)
        ax.annotate(
            label,
            xy=(n, se),
            xytext=(n + 18, se + 0.12),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=color),
            color=color,
        )

    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel(r"$SE(\bar X)$")
    ax.set_title("Uncertainty of the mean shrinks like $1/\\sqrt{n}$, not $1/n$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_sampling_distribution_convergence(populations, sample_sizes, n_reps=1500, seed=42):
    """
    Grid: each row is a population; column 0 is the population itself;
    later columns are sampling distributions of the mean at increasing n.
    """
    rng = np.random.default_rng(seed)
    n_rows, n_cols = len(populations), len(sample_sizes) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.1 * n_rows), squeeze=False)
    plt.subplots_adjust(hspace=0.55, wspace=0.35)

    for i, (pop_name, pop_data) in enumerate(populations.items()):
        pop_data = np.asarray(pop_data)
        mu, _, sigma = sampling_utils.population_moments(pop_data)

        axes[i, 0].hist(pop_data, bins=50, density=True, alpha=0.65, color="grey")
        axes[i, 0].axvline(mu, color="black", ls="--", lw=1.5, label=rf"$\mu={mu:.2f}$")
        axes[i, 0].set_title(f"{pop_name} population")
        axes[i, 0].set_ylabel("Density")
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].legend(fontsize=8)

        for j, n in enumerate(sample_sizes):
            draws = rng.choice(pop_data, size=(n_reps, n), replace=True)
            sample_means = draws.mean(axis=1)
            se = sigma / np.sqrt(n)

            ax = axes[i, j + 1]
            ax.hist(sample_means, bins=40, density=True, alpha=0.65, color="steelblue")
            xs = np.linspace(sample_means.min(), sample_means.max(), 200)
            ax.plot(xs, stats.norm.pdf(xs, mu, se), "r-", lw=2, label="CLT $N(\\mu, SE)$")
            ax.axvline(mu, color="black", ls="--", lw=1.2)
            ax.set_title(f"{pop_name}: $n={n}$\nemp SE={sample_means.std(ddof=1):.3f}  th. SE={se:.3f}")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    plt.show()


def print_sampling_comparison(populations, sample_size=30, n_reps=2000, seed=42):
    """Empirical vs theoretical mean / variance / SE for every population."""
    header = (
        f"{'Population':<22} {'mu':>8} {'E[Xbar]':>8} {'sig2/n':>10} "
        f"{'Var(Xbar)':>10} {'sig/sqrt(n)':>12} {'emp SE':>8}"
    )
    print(f"Sampling distribution of the mean at n={sample_size} ({n_reps} repetitions)")
    print(header)
    print("-" * len(header))
    for name, data in populations.items():
        summary = sampling_utils.summarize_sampling_distribution(
            data, sample_size, n_reps=n_reps, seed=seed
        )
        print(
            f"{name:<22} {summary['pop_mean']:8.3f} {summary['emp_mean_of_means']:8.3f} "
            f"{summary['theory_var_of_mean']:10.4f} {summary['emp_var_of_mean']:10.4f} "
            f"{summary['theory_se']:12.3f} {summary['emp_se']:8.3f}"
        )


def plot_real_population_and_samples(values, name, sample_sizes=(5, 30, 100), n_reps=1500, seed=42):
    populations = {name: np.asarray(values, dtype=float)}
    plot_sampling_distribution_convergence(populations, list(sample_sizes), n_reps=n_reps, seed=seed)



def plot_estimator_comparison(results, true_parameter, title="Estimator sampling distributions"):
    """
    Overlay histograms of several estimators.

    `results` is a dict name -> evaluate_estimator output.
    The visual point: 'centered on the truth' vs 'consistently close'.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(4.4 * len(results), 4.2), sharey=True)
    if len(results) == 1:
        axes = [axes]
    colors = ["steelblue", "orange", "seagreen", "purple"]

    for ax, (name, res), color in zip(axes, results.items(), colors):
        ax.hist(res["estimates"], bins=35, density=True, alpha=0.65, color=color)
        ax.axvline(true_parameter, color="black", ls="--", lw=2, label=rf"true $\theta={true_parameter}$")
        ax.axvline(res["mean_estimate"], color="red", ls="-", lw=2, label=f"mean estimate={res['mean_estimate']:.2f}")
        ax.set_title(
            f"{name}\n"
            f"bias={res['bias']:.3f}  var={res['variance']:.3f}\n"
            f"MSE={res['mse']:.3f}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, y=1.03)
    plt.tight_layout()
    plt.show()


def plot_consistency(trajectories, true_parameter, ylabel=r"$\hat\theta$"):
    """Cloud of estimates at each n; consistent estimators collapse to θ."""
    ns = sorted(trajectories)
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(ns))
    data = [trajectories[n] for n in ns]
    ax.violinplot(data, positions=positions, showmeans=True, showextrema=False)
    ax.axhline(true_parameter, color="red", ls="--", lw=2, label=rf"true $\theta={true_parameter}$")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel(ylabel)
    ax.set_title("Consistency: the estimator concentrates on the parameter as $n$ grows")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_mse_identity(check, title="Monte Carlo check of MSE = Var + Bias²"):
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["MSE", "Var", r"Bias$^2$", r"Var + Bias$^2$"]
    values = [check["mse"], check["var_ddof0"], check["bias_squared"], check["var_plus_bias_sq"]]
    colors = ["purple", "steelblue", "orange", "seagreen"]
    ax.bar(labels, values, color=colors, alpha=0.8)
    ax.set_title(title + f"\n|MSE − (Var + Bias²)| = {check['abs_gap']:.2e}")
    ax.grid(True, axis="y", alpha=0.3)
    plt.show()


def plot_t_vs_normal():
    x = np.linspace(-5, 5, 500)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, stats.norm.pdf(x), "k-", lw=3, label="Standard Normal")
    for df, color in zip([2, 5, 30], ["red", "orange", "green"]):
        ax.plot(x, stats.t.pdf(x, df), color=color, lw=2, ls="--", label=f"$t$ (df={df})")

    # Shade the extra tail mass of t_2 relative to normal in the far tail
    tail = x > 2.0
    ax.fill_between(x[tail], stats.norm.pdf(x[tail]), stats.t.pdf(x[tail], 2),
                    color="red", alpha=0.15, label="extra tail mass (df=2)")
    ax.set_title("Student's $t$ has heavier tails than Normal; they agree as df grows")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_repeated_confidence_intervals(n=20, confidence=0.95, num_intervals=50, seed=42,
                                       mu_true=0.0, sigma_true=1.0):
    """Static coverage picture (also used by the interactive widget)."""
    rng = np.random.default_rng(seed)
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    hits = 0

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(num_intervals):
        sample = rng.normal(mu_true, sigma_true, n)
        mean = sample.mean()
        se = sample.std(ddof=1) / np.sqrt(n)
        low, high = mean - t_crit * se, mean + t_crit * se
        contains = low <= mu_true <= high
        hits += int(contains)
        color = "steelblue" if contains else "red"
        ax.plot([low, high], [i, i], color=color, lw=2)
        ax.plot(mean, i, color=color, marker="o", markersize=4)

    ax.axvline(mu_true, color="black", ls="--", label=f"true $\\mu={mu_true}$")
    ax.set_xlabel("Estimate")
    ax.set_ylabel("Sample index")
    ax.set_title(
        f"{int(confidence * 100)}% t-intervals, n={n}: "
        f"{hits}/{num_intervals} contain $\\mu$ "
        f"({100 * hits / num_intervals:.1f}%)\n"
        "Blue = covers the fixed parameter; red = misses"
    )
    ax.legend()
    plt.show()
    return hits / num_intervals


def interactive_confidence_intervals():
    def plot_ci(n, confidence, num_intervals):
        plot_repeated_confidence_intervals(
            n=n, confidence=confidence / 100.0, num_intervals=num_intervals
        )

    ui = widgets.interactive(
        plot_ci,
        n=widgets.IntSlider(min=5, max=100, step=5, value=20, description="Sample size n",
                            continuous_update=False),
        confidence=widgets.Dropdown(options=[90, 95, 99], value=95, description="Confidence %"),
        num_intervals=widgets.IntSlider(min=10, max=100, step=10, value=50, description="Num intervals",
                                        continuous_update=False),
    )
    display(ui)

def plot_null_pvalues(p_values, alpha=0.05):
    p_values = np.asarray(p_values)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(p_values, bins=40, density=True, color="purple", alpha=0.7, edgecolor="white")
    ax.axhline(1.0, color="black", ls="--", label="Uniform(0,1) density")
    ax.axvline(alpha, color="red", ls="--", label=rf"$\alpha={alpha}$")
    rate = float(np.mean(p_values < alpha))
    ax.set_title(
        f"p-values under a true $H_0$ are approximately uniform\n"
        f"Empirical Type I rate = {rate:.3f} (target $\\alpha={alpha}$)"
    )
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()
    return rate


def plot_one_vs_two_tailed(t_obs=2.1, df=20, alpha=0.05):
    x = np.linspace(-4.5, 4.5, 500)
    pdf = stats.t.pdf(x, df)
    t_crit_two = stats.t.ppf(1 - alpha / 2, df)
    t_crit_one = stats.t.ppf(1 - alpha, df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, title, crits, shade_left in [
        (axes[0], "Two-tailed $H_a: \\mu \\neq \\mu_0$", (-t_crit_two, t_crit_two), True),
        (axes[1], "One-tailed $H_a: \\mu > \\mu_0$", (t_crit_one,), False),
    ]:
        ax.plot(x, pdf, color="steelblue", lw=2, label="$t$ null density")
        if shade_left:
            ax.fill_between(x, pdf, where=(x <= crits[0]) | (x >= crits[1]),
                            color="red", alpha=0.3, label=rf"rejection region ($\alpha={alpha}$)")
            ax.axvline(crits[0], color="red", ls="--")
            ax.axvline(crits[1], color="red", ls="--")
        else:
            ax.fill_between(x, pdf, where=(x >= crits[0]),
                            color="red", alpha=0.3, label=rf"rejection region ($\alpha={alpha}$)")
            ax.axvline(crits[0], color="red", ls="--")
        ax.axvline(t_obs, color="black", lw=2, label=f"observed $t={t_obs}$")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("The same $t$ can be significant one-tailed and not two-tailed", y=1.03)
    plt.tight_layout()
    plt.show()


def plot_pvalue_region(t_obs=2.4, df=19):
    """Shade the two-tailed p-value region on the null t distribution."""
    x = np.linspace(-5, 5, 600)
    pdf = stats.t.pdf(x, df)
    p = 2 * stats.t.sf(abs(t_obs), df)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, pdf, color="steelblue", lw=2)
    ax.fill_between(x, pdf, where=(x <= -abs(t_obs)) | (x >= abs(t_obs)),
                    color="purple", alpha=0.35, label=f"p-value region = {p:.3f}")
    ax.axvline(t_obs, color="black", lw=2, label=f"observed $t={t_obs}$")
    ax.axvline(-t_obs, color="black", lw=1, ls=":")
    ax.set_title(r"$p = P(|T| \geq |t_{\mathrm{obs}}| \mid H_0)$ — a tail probability under the null")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


def plot_power_curves(n=20, effect_size=0.5, noise=1.0, alpha=0.05, alternative="greater"):
    """Static snapshot of H0 vs H1, rejection region, Type I / II, power."""
    mu_0, mu_a = 0.0, effect_size
    se = noise / np.sqrt(n)
    x_min = min(mu_0, mu_a) - 4 * se
    x_max = max(mu_0, mu_a) + 4 * se
    x = np.linspace(x_min, x_max, 600)
    y0 = stats.norm.pdf(x, mu_0, se)
    ya = stats.norm.pdf(x, mu_a, se)

    if alternative == "greater":
        crit = stats.norm.ppf(1 - alpha, mu_0, se)
        power = 1 - stats.norm.cdf(crit, mu_a, se)
        beta = stats.norm.cdf(crit, mu_a, se)
    else:
        crit = stats.norm.ppf(1 - alpha / 2, mu_0, se)
        power = 1 - (stats.norm.cdf(crit, mu_a, se) - stats.norm.cdf(-crit + 2 * mu_0, mu_a, se))
        beta = 1 - power

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y0, color="blue", lw=2, label="$H_0$ sampling dist. of $\\bar X$")
    ax.plot(x, ya, color="orange", lw=2, label="$H_1$ sampling dist. of $\\bar X$")
    ax.fill_between(x, y0, where=(x >= crit), color="blue", alpha=0.25,
                    label=rf"Type I ($\alpha={alpha}$)")
    ax.fill_between(x, ya, where=(x < crit), color="orange", alpha=0.25,
                    label=rf"Type II ($\beta={beta:.2f}$)")
    ax.axvline(crit, color="red", ls="--", label=f"critical value {crit:.2f}")
    ax.set_title(f"Power $= 1-\\beta = {power:.2f}$   (n={n}, effect={effect_size}, $\\sigma$={noise})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.show()
    return power


def interactive_power_experiment():
    def plot_power(n, effect_size, noise, alpha):
        plot_power_curves(n=n, effect_size=effect_size, noise=noise, alpha=alpha)

    ui = widgets.interactive(
        plot_power,
        n=widgets.IntSlider(min=5, max=150, step=5, value=20, description="Sample size",
                            continuous_update=False),
        effect_size=widgets.FloatSlider(min=0.1, max=2.0, step=0.1, value=0.5, description="Effect size",
                                        continuous_update=False),
        noise=widgets.FloatSlider(min=0.5, max=3.0, step=0.1, value=1.0, description="Noise (std)",
                                  continuous_update=False),
        alpha=widgets.Dropdown(options=[0.01, 0.05, 0.10], value=0.05, description=r"Alpha ($\alpha$)"),
    )
    display(ui)


def plot_power_relationships(n_grid=None):
    """Four-panel quantitative summary of what moves power."""
    if n_grid is None:
        n_grid = np.array([5, 10, 20, 40, 80, 160])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def power_of(n, effect, noise, alpha):
        se = noise / np.sqrt(n)
        crit = stats.norm.ppf(1 - alpha, 0, se)
        return 1 - stats.norm.cdf(crit, effect, se)

    axes[0, 0].plot(n_grid, [power_of(n, 0.5, 1.0, 0.05) for n in n_grid], "o-", color="blue")
    axes[0, 0].set_title("Larger $n$ → higher power")
    axes[0, 0].set_xlabel("n")

    effects = np.linspace(0.1, 1.5, 12)
    axes[0, 1].plot(effects, [power_of(20, e, 1.0, 0.05) for e in effects], "o-", color="orange")
    axes[0, 1].set_title("Larger effect → higher power")
    axes[0, 1].set_xlabel("effect size")

    noises = np.linspace(0.4, 2.5, 12)
    axes[1, 0].plot(noises, [power_of(20, 0.5, s, 0.05) for s in noises], "o-", color="green")
    axes[1, 0].set_title("More noise → lower power")
    axes[1, 0].set_xlabel(r"$\sigma$")

    alphas = [0.01, 0.05, 0.10]
    axes[1, 1].bar([str(a) for a in alphas], [power_of(20, 0.5, 1.0, a) for a in alphas],
                   color="purple", alpha=0.7)
    axes[1, 1].set_title(r"Larger $\alpha$ → higher power (and more Type I)")
    axes[1, 1].set_xlabel(r"$\alpha$")

    for ax in axes.ravel():
        ax.set_ylabel("Power")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_paired_vs_independent(baseline, after):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    n = len(baseline)
    axes[0].scatter(np.zeros(n) + 0.05 * np.random.default_rng(0).standard_normal(n),
                    baseline, alpha=0.6, color="steelblue", label="before")
    axes[0].scatter(np.ones(n) + 0.05 * np.random.default_rng(1).standard_normal(n),
                    after, alpha=0.6, color="orange", label="after")
    for b, a in zip(baseline, after):
        axes[0].plot([0, 1], [b, a], color="grey", alpha=0.25)
    axes[0].set_xticks([0, 1], ["Before", "After"])
    axes[0].set_title("Paired observations share a baseline")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    diffs = np.asarray(after) - np.asarray(baseline)
    axes[1].hist(diffs, bins=12, color="seagreen", alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="black", ls="--", label="null $\\mu_D=0$")
    axes[1].axvline(diffs.mean(), color="red", label=f"mean diff={diffs.mean():.2f}")
    axes[1].set_title("Paired test uses the differences")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_mcnemar_table(table):
    grid = np.array([
        [table["both_correct"], table["a_only"]],
        [table["b_only"], table["both_wrong"]],
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="Blues")
    labels = [["Both correct\n$N_{11}$", "A only\n$N_{10}$"],
              ["B only\n$N_{01}$", "Both wrong\n$N_{00}$"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{grid[i, j]}", ha="center", va="center", fontsize=11)
    ax.set_xticks([0, 1], ["B correct", "B wrong"])
    ax.set_yticks([0, 1], ["A correct", "A wrong"])
    ax.set_title("McNemar uses only the discordant cells $N_{10}$ and $N_{01}$")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.show()


def plot_ttest_assumption_simulation(n_small=8, n_large=80, n_reps=4000, seed=42, alpha=0.05):
    """
    Type I error of the one-sample t-test on exponential data.
    Small n: the normality assumption is strained. Large n: CLT helps.
    """
    rng = np.random.default_rng(seed)
    # Exponential(1) has mean 1. Test H0: mu = 1, which is true.
    rates = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, n, title in [
        (axes[0], n_small, f"Small sample (n={n_small})"),
        (axes[1], n_large, f"Large sample (n={n_large})"),
    ]:
        pvals = []
        for _ in range(n_reps):
            sample = rng.exponential(1.0, n)
            _, p, _ = hypothesis_testing.one_sample_t_test(sample, 1.0)
            pvals.append(p)
        pvals = np.asarray(pvals)
        rate = float(np.mean(pvals < alpha))
        rates[n] = rate
        ax.hist(pvals, bins=30, density=True, color="teal", alpha=0.7)
        ax.axhline(1.0, color="black", ls="--")
        ax.axvline(alpha, color="red", ls="--")
        ax.set_title(f"{title}\nType I rate = {rate:.3f} (target {alpha})")
        ax.set_xlabel("p-value")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return rates


def plot_ci_vs_hypothesis(sample_mean, se, null_value, alpha=0.05, df=None):
    if df is None:
        crit = stats.norm.ppf(1 - alpha / 2)
        dist_name = "Normal"
        pdf = lambda x: stats.norm.pdf(x, null_value, se)
    else:
        crit = stats.t.ppf(1 - alpha / 2, df)
        dist_name = f"$t_{{{df}}}$"
        # scale a standard t to the SE of the mean
        pdf = lambda x: stats.t.pdf((x - null_value) / se, df) / se

    margin = crit * se
    ci_lower, ci_upper = sample_mean - margin, sample_mean + margin
    rejects = (null_value < ci_lower) or (null_value > ci_upper)

    x = np.linspace(min(null_value, sample_mean) - 4 * se,
                    max(null_value, sample_mean) + 4 * se, 400)
    y_null = pdf(x)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y_null, color="blue", label=f"Null sampling dist. ({dist_name})")
    ax.fill_between(x, y_null, where=(x < null_value - crit * se) | (x > null_value + crit * se),
                    color="red", alpha=0.25, label="rejection region")
    y_ci = np.max(y_null) * 1.12
    ax.plot([ci_lower, ci_upper], [y_ci, y_ci], color="green", lw=4, label="95% CI for $\\mu$")
    ax.plot(sample_mean, y_ci, marker="o", color="green", markersize=8)
    ax.axvline(null_value, color="black", ls="--", label=f"null value {null_value}")
    decision = "reject $H_0$" if rejects else "fail to reject $H_0$"
    ax.set_title(f"CI vs two-sided test: null is {'outside' if rejects else 'inside'} the CI → {decision}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.show()
    return ci_lower, ci_upper, rejects


def plot_practical_vs_statistical():
    """
    Left: tiny effect, huge n → significant but irrelevant.
    Right: meaningful effect, tiny n → not significant.
    """
    rng = np.random.default_rng(1)
    # 0.3-point accuracy gap, enormous n → tiny p, irrelevant effect.
    tiny_a = rng.normal(0.800, 0.05, 80_000)
    tiny_b = rng.normal(0.803, 0.05, 80_000)
    t_tiny, p_tiny, _ = hypothesis_testing.two_sample_t_test_welch(tiny_b, tiny_a)

    # 12-point gap, n=8 each, noisy → often fails to reach 0.05.
    big_a = rng.normal(0.70, 0.16, 8)
    big_b = rng.normal(0.82, 0.16, 8)
    t_big, p_big, _ = hypothesis_testing.two_sample_t_test_welch(big_b, big_a)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(tiny_a, bins=40, alpha=0.5, color="steelblue", density=True, label="A: 0.800")
    axes[0].hist(tiny_b, bins=40, alpha=0.5, color="orange", density=True, label="B: 0.803")
    axes[0].set_title(f"Tiny effect (+0.003), n=80,000\np={p_tiny:.1e}  (statistically significant)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(big_a, bins=8, alpha=0.5, color="steelblue", density=True, label="A: 0.70")
    axes[1].hist(big_b, bins=8, alpha=0.5, color="orange", density=True, label="B: 0.82")
    axes[1].set_title(f"Meaningful effect (+0.12), n=8\np={p_big:.3f}  (fails to reach 0.05)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return {"tiny_p": p_tiny, "meaningful_p": p_big, "tiny_t": t_tiny, "meaningful_t": t_big}


def plot_anova_intuition(cases, seed=42):
    """
    cases: list of dicts with keys name, means, within_std, n
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 4.6), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    colors = ["steelblue", "orange", "seagreen", "purple"]

    for ax, case in zip(axes, cases):
        groups = [rng.normal(m, case["within_std"], case["n"]) for m in case["means"]]
        res = hypothesis_testing.anova_components(*groups)
        for g, color, mean in zip(groups, colors, case["means"]):
            ax.scatter(np.full(len(g), mean) + 0.08 * rng.standard_normal(len(g)),
                       g, alpha=0.55, color=color)
            ax.plot([mean - 0.25, mean + 0.25], [np.mean(g), np.mean(g)], color="black", lw=2)
        ax.axhline(res["overall_mean"], color="black", ls="--", label="overall mean")
        ax.set_title(
            f"{case['name']}\n"
            f"F={res['F_stat']:.2f}, p={res['p_val']:.3f}\n"
            f"$MS_B$={res['MS_B']:.2f}, $MS_W$={res['MS_W']:.2f}"
        )
        ax.set_xlabel("designed group mean")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Observation")
    plt.tight_layout()
    plt.show()


def plot_anova_simulation_cases(n_reps=400, n=20, seed=42):
    """Case A same means, B separated, C separated + extra noise."""
    rng = np.random.default_rng(seed)
    specs = {
        "A: same means": ([0, 0, 0], 1.0),
        "B: separated": ([-1.2, 0.0, 1.2], 1.0),
        "C: separated + noise": ([-1.2, 0.0, 1.2], 2.5),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (name, (means, noise)) in zip(axes, specs.items()):
        f_stats = []
        for _ in range(n_reps):
            groups = [rng.normal(m, noise, n) for m in means]
            f_stats.append(hypothesis_testing.anova_components(*groups)["F_stat"])
        ax.hist(f_stats, bins=30, color="slateblue", alpha=0.75, density=True)
        ax.axvline(1.0, color="black", ls="--", label="F ≈ 1 under $H_0$")
        ax.axvline(np.mean(f_stats), color="red", label=f"mean F={np.mean(f_stats):.2f}")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def interactive_anova():
    def plot_anova(mean_A, mean_B, mean_C, within_std, n):
        rng = np.random.default_rng(42)
        groups = [
            rng.normal(mean_A, within_std, n),
            rng.normal(mean_B, within_std, n),
            rng.normal(mean_C, within_std, n),
        ]
        res = hypothesis_testing.anova_components(*groups)
        colors = ["steelblue", "orange", "seagreen"]
        labels = ["A", "B", "C"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.4, 1]})
        for g, color, lab in zip(groups, colors, labels):
            axes[0].scatter([lab] * n, g, alpha=0.55, color=color)
            axes[0].scatter([lab], [np.mean(g)], color="black", marker="X", s=110)
        axes[0].axhline(res["overall_mean"], color="black", ls="--",
                        label=f"overall mean {res['overall_mean']:.2f}")
        axes[0].set_title(
            f"F={res['F_stat']:.2f}, p={res['p_val']:.4f}\n"
            f"$SS_B$={res['SS_B']:.2f}, $SS_W$={res['SS_W']:.2f}, "
            f"$MS_B$={res['MS_B']:.2f}, $MS_W$={res['MS_W']:.2f}"
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(["$SS_B$", "$SS_W$"], [res["SS_B"], res["SS_W"]],
                    color=["orange", "steelblue"], alpha=0.8)
        axes[1].set_title("Between vs within sum of squares")
        axes[1].grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    ui = widgets.interactive(
        plot_anova,
        mean_A=widgets.FloatSlider(min=-5, max=5, step=0.5, value=0.0, description="Mean A",
                                   continuous_update=False),
        mean_B=widgets.FloatSlider(min=-5, max=5, step=0.5, value=1.0, description="Mean B",
                                   continuous_update=False),
        mean_C=widgets.FloatSlider(min=-5, max=5, step=0.5, value=-1.0, description="Mean C",
                                   continuous_update=False),
        within_std=widgets.FloatSlider(min=0.3, max=4.0, step=0.1, value=1.0, description="Within std",
                                       continuous_update=False),
        n=widgets.IntSlider(min=8, max=60, step=2, value=20, description="n per group",
                            continuous_update=False),
    )
    display(ui)

def plot_unified_inference_chain(population, sample, mu0, seed=42):
    """Visual story: population → sample → sampling dist → CI → test."""
    rng = np.random.default_rng(seed)
    population = np.asarray(population, dtype=float)
    sample = np.asarray(sample, dtype=float)
    mu, _, sigma = sampling_utils.population_moments(population)
    n = len(sample)
    means = sampling_utils.sample_means_from_population(population, n, n_reps=1200, seed=seed)
    low, high, est, se, _ = hypothesis_testing.one_sample_t_ci(sample, 0.95)
    t_stat, p_val, df = hypothesis_testing.one_sample_t_test(sample, mu0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(population, bins=50, color="grey", alpha=0.7, density=True)
    axes[0, 0].axvline(mu, color="black", ls="--", label=f"pop $\\mu={mu:.2f}$")
    axes[0, 0].set_title("1. Population")
    axes[0, 0].legend()

    axes[0, 1].hist(sample, bins=12, color="steelblue", alpha=0.75)
    axes[0, 1].axvline(est, color="red", label=f"$\\bar X={est:.2f}$")
    axes[0, 1].set_title(f"2. One sample (n={n})")
    axes[0, 1].legend()

    axes[1, 0].hist(means, bins=35, color="steelblue", alpha=0.7, density=True)
    xs = np.linspace(means.min(), means.max(), 200)
    axes[1, 0].plot(xs, stats.norm.pdf(xs, mu, sigma / np.sqrt(n)), "r-")
    axes[1, 0].axvline(est, color="red", label="this sample's mean")
    axes[1, 0].set_title(r"3. Sampling distribution of $\bar X$")
    axes[1, 0].legend()

    axes[1, 1].errorbar([0], [est], yerr=[[est - low], [high - est]], fmt="o",
                        color="green", capsize=8, lw=2, label="95% CI")
    axes[1, 1].axhline(mu, color="black", ls="--", label="true $\\mu$")
    axes[1, 1].axhline(mu0, color="purple", ls=":", label=f"$H_0$ value {mu0}")
    axes[1, 1].set_xlim(-1, 1)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_title(f"4. CI and test  t={t_stat:.2f}, p={p_val:.3f}, df={df}")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return {"mu": mu, "estimate": est, "se": se, "ci": (low, high), "t": t_stat, "p": p_val}


def plot_model_comparison(acc_diffs, observed_diff, ci, p_value):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(acc_diffs, bins=30, color="steelblue", alpha=0.7, density=True,
            label="accuracy difference across repeated test sets")
    ax.axvline(0, color="black", ls="--", label="no difference")
    ax.axvline(observed_diff, color="red", lw=2, label=f"observed Δ = {observed_diff:.3f}")
    ax.plot(ci, [0.15, 0.15], color="green", lw=4, label=f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    ax.set_title(f"Is model B genuinely better?  p={p_value:.3f}")
    ax.set_xlabel("Accuracy$_B$ − Accuracy$_A$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.show()


def plot_cdf_to_pvalue(t_obs=2.3, df=19):
    """
    Show the CDF → tail-probability pipeline for one observed t statistic.

    Left: CDF evaluated at t_obs.
    Middle: the three p-values (left / right / two-sided) on the null density.
    Right: same idea on the CDF curve itself.
    """
    x = np.linspace(-5.0, 5.0, 700)
    pdf = stats.t.pdf(x, df)
    cdf = stats.t.cdf(x, df)
    F_obs = float(stats.t.cdf(t_obs, df))
    p_right = float(stats.t.sf(t_obs, df))
    p_left = float(stats.t.cdf(t_obs, df))
    p_two = float(2.0 * stats.t.sf(abs(t_obs), df))
    p_left_if_positive = float(stats.t.cdf(t_obs, df))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    axes[0].plot(x, cdf, color="steelblue", lw=2)
    axes[0].axvline(t_obs, color="black", lw=1.5)
    axes[0].axhline(F_obs, color="black", ls=":", lw=1)
    axes[0].scatter([t_obs], [F_obs], color="red", zorder=5, s=40)
    axes[0].set_title(rf"$F_{{t_{{{df}}}}}({t_obs}) = {F_obs:.3f}$")
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$F(t)=P(T\leq t)$")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, pdf, color="steelblue", lw=2)
    axes[1].fill_between(x, pdf, where=(x >= t_obs), color="orange", alpha=0.45,
                         label=rf"right tail $1-F={p_right:.3f}$")
    axes[1].fill_between(x, pdf, where=(x <= -abs(t_obs)), color="purple", alpha=0.35,
                         label=rf"mirror tail (two-sided extra)")
    axes[1].axvline(t_obs, color="black", lw=1.5, label=rf"$t_{{\mathrm{{obs}}}}={t_obs}$")
    axes[1].axvline(-abs(t_obs), color="black", ls=":", lw=1)
    axes[1].set_title("Null $t$ density: the p-value is shaded mass")
    axes[1].set_xlabel(r"$t$")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    labels = [r"$H_1:\mu>\mu_0$", r"$H_1:\mu<\mu_0$", r"$H_1:\mu\neq\mu_0$"]
    values = [p_right, p_left_if_positive, p_two]
    colors = ["orange", "steelblue", "purple"]
    bars = axes[2].bar(labels, values, color=colors, alpha=0.8)
    axes[2].axhline(0.05, color="red", ls="--", label=r"$\alpha=0.05$")
    for bar, val in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                     ha="center", fontsize=9)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("p-value")
    axes[2].set_title("Same $t$, different $H_1$ → different $p$")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        rf"$t={t_obs}$, df={df}:  $F={F_obs:.3f}$,  "
        rf"$p_{{\mathrm{{right}}}}={p_right:.3f}$,  $p_{{\mathrm{{two}}}}={p_two:.3f}$",
        y=1.03,
    )
    plt.tight_layout()
    plt.show()
    return {"F": F_obs, "p_right": p_right, "p_left": p_left, "p_two": p_two}


def plot_critical_region(alpha=0.05, n=30, sigma=1.0, mu0=0.0):
    """
    Critical region on the z-scale and on the sample-mean scale.

    One-sided α=0.05 uses Z_c = Φ^{-1}(1-α) ≈ 1.645.
    Two-sided α=0.05 uses ±z_{1-α/2} ≈ ±1.96.
    """
    z_one = float(stats.norm.ppf(1.0 - alpha))
    z_two = float(stats.norm.ppf(1.0 - alpha / 2.0))
    se = sigma / np.sqrt(n)
    xbar_c = mu0 + z_one * se

    z = np.linspace(-4.0, 4.0, 700)
    pdf = stats.norm.pdf(z)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    axes[0].plot(z, pdf, color="steelblue", lw=2)
    axes[0].fill_between(z, pdf, where=(z >= z_one), color="red", alpha=0.35,
                         label=rf"reject $H_0$  ($Z\geq {z_one:.3f}$)")
    axes[0].axvline(z_one, color="red", ls="--")
    axes[0].set_title(rf"One-sided critical region, $\alpha={alpha}$" + "\n"
                      + rf"$Z_c \approx {z_one:.3f}$  (leaves probability $\alpha$ in the right tail)")
    axes[0].set_xlabel(r"$Z=(\bar X-\mu_0)/(\sigma/\sqrt{n})$")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(z, pdf, color="steelblue", lw=2)
    axes[1].fill_between(z, pdf, where=(z <= -z_two) | (z >= z_two),
                         color="red", alpha=0.35,
                         label=rf"reject $H_0$  ($|Z|\geq {z_two:.3f}$)")
    axes[1].axvline(z_two, color="red", ls="--")
    axes[1].axvline(-z_two, color="red", ls="--")
    axes[1].set_title(rf"Two-sided critical region, $\alpha={alpha}$" + "\n"
                      + rf"$Z_c \approx {z_two:.3f}$  (leaves probability $\alpha/2$ in each tail)")
    axes[1].set_xlabel(r"$Z$")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        rf"On the $\bar X$ scale (n={n}, $\sigma={sigma}$): one-sided reject if "
        rf"$\bar X > {xbar_c:.3f}$",
        y=1.04,
    )
    plt.tight_layout()
    plt.show()
    return {"z_one": z_one, "z_two": z_two, "xbar_critical_one": xbar_c}


def plot_mcnemar_null(n10=30, n01=10):
    """χ²_1 tail for the asymptotic McNemar statistic, plus the Z² link."""
    chi2_obs, p_asym = hypothesis_testing.mcnemar_chi2_from_counts(
        n10, n01, continuity_correction=False
    )
    p_exact = hypothesis_testing.mcnemar_exact_from_counts(n10, n01)
    z_eq = np.sqrt(chi2_obs)
    p_from_z = float(2.0 * stats.norm.sf(z_eq))

    x = np.linspace(0.0, max(18.0, chi2_obs + 4), 500)
    pdf = stats.chi2.pdf(x, df=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(x, pdf, color="steelblue", lw=2)
    axes[0].fill_between(x, pdf, where=(x >= chi2_obs), color="purple", alpha=0.4,
                         label=rf"$P(\chi^2_1 \geq {chi2_obs:.1f})={p_asym:.4f}$")
    axes[0].axvline(chi2_obs, color="black", lw=1.5)
    axes[0].set_title(r"Asymptotic McNemar: $p=1-F_{\chi^2_1}(\chi^2_{\mathrm{obs}})$")
    axes[0].set_xlabel(r"$\chi^2$")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    n = n10 + n01
    ks = np.arange(0, n + 1)
    pmf = stats.binom.pmf(ks, n, 0.5)
    k_obs = n10
    k_ext = max(n10, n01)
    extreme = (ks <= (n - k_ext)) | (ks >= k_ext)
    axes[1].bar(ks, pmf, color="lightgrey", width=0.9)
    axes[1].bar(ks[extreme], pmf[extreme], color="purple", width=0.9, alpha=0.85,
                label=rf"exact two-sided $p={p_exact:.4f}$")
    axes[1].axvline(k_obs, color="black", lw=1.5, label=rf"$X={k_obs}$")
    axes[1].set_title(rf"Exact McNemar: $X\sim Binomial({n}, 0.5)$")
    axes[1].set_xlabel("A-wins among discordant pairs")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        rf"$b={n10}$, $c={n01}$, $\chi^2={chi2_obs:.1f}$, "
        rf"$|Z|=\sqrt{{\chi^2}}\approx{z_eq:.3f}$,  "
        rf"$2(1-\Phi(|Z|))={p_from_z:.4f}$",
        y=1.04,
    )
    plt.tight_layout()
    plt.show()
    return {
        "chi2": chi2_obs,
        "p_asym": p_asym,
        "p_exact": p_exact,
        "p_from_z": p_from_z,
    }


def plot_fwer_and_tukey(tukey_result):
    """
    Left: uncorrected FWER explosion as the number of pairs grows.
    Right: Tukey intervals; reject a pair iff 0 is outside the interval.
    """
    alpha = tukey_result["alpha"]
    ks = np.arange(2, 9)
    n_pairs = ks * (ks - 1) // 2
    fwer = [hypothesis_testing.family_wise_error_rate(alpha, int(m)) for m in n_pairs]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].plot(n_pairs, fwer, "o-", color="crimson")
    axes[0].axhline(alpha, color="black", ls="--",
                    label=rf"target FWER $=\alpha={alpha}$")
    axes[0].set_xlabel(r"number of pairwise tests $\binom{k}{2}$")
    axes[0].set_ylabel("P(at least one false rejection)")
    axes[0].set_title("Uncorrected pairwise $t$-tests inflate FWER")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    pairs = tukey_result["pairs"]
    y = np.arange(len(pairs))
    for i, pair in enumerate(pairs):
        color = "crimson" if pair["reject"] else "steelblue"
        axes[1].plot([pair["lower"], pair["upper"]], [i, i], color=color, lw=3)
        axes[1].plot(pair["meandiff"], i, "o", color=color)
    axes[1].axvline(0.0, color="black", ls="--", label="no difference")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(
        [f"{p['group_i']} vs {p['group_j']}" for p in pairs]
    )
    axes[1].set_xlabel(r"Tukey simultaneous CI for $\mu_j-\mu_i$")
    axes[1].set_title(
        rf"Tukey HSD ($q^*={tukey_result['q_crit']:.3f}$): "
        "red = 0 outside interval → reject"
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
