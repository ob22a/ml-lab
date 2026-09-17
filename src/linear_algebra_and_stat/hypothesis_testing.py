import numpy as np
import scipy.stats as stats


def one_sample_t_test(data, mu_0, alternative="two-sided"):
    """
    One-sample t-statistic and p-value from the mathematical formulas.

    t = (\\bar X - mu_0) / (s / sqrt(n)),  df = n - 1
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    mean_val = float(np.mean(data))
    s = float(np.std(data, ddof=1))
    se = s / np.sqrt(n)
    t_stat = (mean_val - mu_0) / se
    df = n - 1
    p_val = _t_pvalue(t_stat, df, alternative)
    return t_stat, p_val, df


def one_sample_z_test(data, mu_0, sigma, alternative="two-sided"):
    """z-test when the population standard deviation is known."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    z_stat = (float(np.mean(data)) - mu_0) / (sigma / np.sqrt(n))
    p_val = _z_pvalue(z_stat, alternative)
    return z_stat, p_val


def one_sample_t_ci(data, confidence=0.95):
    """t-based CI for a mean when sigma is unknown."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    mean_val = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n))
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    return mean_val - t_crit * se, mean_val + t_crit * se, mean_val, se, t_crit


def two_sample_t_test_welch(data_a, data_b, alternative="two-sided"):
    """Welch's t-test: independent samples, no equal-variance assumption."""
    data_a = np.asarray(data_a, dtype=float)
    data_b = np.asarray(data_b, dtype=float)
    n_a, n_b = len(data_a), len(data_b)
    mean_a, mean_b = float(np.mean(data_a)), float(np.mean(data_b))
    var_a, var_b = float(np.var(data_a, ddof=1)), float(np.var(data_b, ddof=1))

    se = np.sqrt(var_a / n_a + var_b / n_b)
    t_stat = (mean_a - mean_b) / se

    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = ((var_a / n_a) ** 2) / (n_a - 1) + ((var_b / n_b) ** 2) / (n_b - 1)
    df = df_num / df_den
    p_val = _t_pvalue(t_stat, df, alternative)
    return t_stat, p_val, df


def pooled_two_sample_t_test(data_a, data_b, alternative="two-sided"):
    """
    Classical two-sample t-test with pooled variance.

    Appropriate only when the two populations can reasonably be assumed
    to share a common variance.
    """
    data_a = np.asarray(data_a, dtype=float)
    data_b = np.asarray(data_b, dtype=float)
    n_a, n_b = len(data_a), len(data_b)
    var_a, var_b = float(np.var(data_a, ddof=1)), float(np.var(data_b, ddof=1))
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    se = np.sqrt(pooled_var * (1 / n_a + 1 / n_b))
    t_stat = (float(np.mean(data_a)) - float(np.mean(data_b))) / se
    df = n_a + n_b - 2
    p_val = _t_pvalue(t_stat, df, alternative)
    return t_stat, p_val, df, pooled_var


def paired_t_test(data_a, data_b, alternative="two-sided"):
    """Paired t-test: one-sample test on the differences."""
    differences = np.asarray(data_a, dtype=float) - np.asarray(data_b, dtype=float)
    return one_sample_t_test(differences, 0.0, alternative=alternative)


def mcnemar_table(y_true, y_pred_a, y_pred_b):
    """2x2 agreement table for two classifiers on the same examples."""
    correct_a = np.asarray(y_true) == np.asarray(y_pred_a)
    correct_b = np.asarray(y_true) == np.asarray(y_pred_b)
    n11 = int(np.sum(correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    return {"both_correct": n11, "both_wrong": n00, "a_only": n10, "b_only": n01}


def mcnemars_test(y_true, y_pred_a, y_pred_b, continuity_correction=True):
    """
    McNemar's test on discordant pairs.

    chi2 = (|n10 - n01| - c)^2 / (n10 + n01),  c = 1 if continuity_correction.
    """
    table = mcnemar_table(y_true, y_pred_a, y_pred_b)
    n10, n01 = table["a_only"], table["b_only"]
    correction = 1.0 if continuity_correction else 0.0
    denom = n10 + n01
    if denom == 0:
        return 0.0, 1.0, table
    chi2_stat = (abs(n10 - n01) - correction) ** 2 / denom
    # Survival function: P(χ²_1 ≥ χ²_obs). Prefer sf to 1-cdf in the far tail.
    p_val = float(stats.chi2.sf(chi2_stat, df=1))
    return float(chi2_stat), float(p_val), table


def mcnemar_chi2_from_counts(n10, n01, continuity_correction=True):
    """Asymptotic McNemar χ² and p-value from discordant counts only."""
    correction = 1.0 if continuity_correction else 0.0
    denom = n10 + n01
    if denom == 0:
        return 0.0, 1.0
    chi2_stat = (abs(n10 - n01) - correction) ** 2 / denom
    return float(chi2_stat), float(stats.chi2.sf(chi2_stat, df=1))


def mcnemar_exact_from_counts(n10, n01):
    """
    Exact McNemar test: among n = n10 + n01 discordant pairs,
    X = n10 ~ Binomial(n, 1/2) under H0.

    Two-sided p-value: twice the probability of a tail at least as
    extreme as the observed winner count, clipped at 1.
    """
    n = int(n10 + n01)
    if n == 0:
        return 1.0
    k_ext = max(int(n10), int(n01))
    p_one_tail = float(stats.binom.sf(k_ext - 1, n, 0.5))
    return float(min(1.0, 2.0 * p_one_tail))


def anova_components(*groups):
    """One-way ANOVA sums of squares, mean squares, F, and p-value."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    k = len(groups)
    all_data = np.concatenate(groups)
    n = len(all_data)
    overall_mean = float(np.mean(all_data))

    ss_w = 0.0
    ss_b = 0.0
    for group in groups:
        nj = len(group)
        group_mean = float(np.mean(group))
        ss_w += float(np.sum((group - group_mean) ** 2))
        ss_b += nj * (group_mean - overall_mean) ** 2

    ss_t = float(np.sum((all_data - overall_mean) ** 2))
    df_b = k - 1
    df_w = n - k
    ms_b = ss_b / df_b if df_b > 0 else 0.0
    ms_w = ss_w / df_w if df_w > 0 else 0.0
    f_stat = ms_b / ms_w if ms_w > 0 else 0.0
    p_val = float(stats.f.sf(f_stat, dfn=df_b, dfd=df_w)) if ms_w > 0 else 1.0

    return {
        "SS_T": ss_t,
        "SS_W": ss_w,
        "SS_B": ss_b,
        "df_W": df_w,
        "df_B": df_b,
        "MS_W": ms_w,
        "MS_B": ms_b,
        "F_stat": f_stat,
        "p_val": p_val,
        "overall_mean": overall_mean,
        "group_means": [float(np.mean(g)) for g in groups],
        "group_sizes": [len(g) for g in groups],
    }


def compare_one_sample_with_scipy(data, mu_0):
    """Manual one-sample t-test versus SciPy."""
    t_manual, p_manual, df = one_sample_t_test(data, mu_0)
    scipy_res = stats.ttest_1samp(data, mu_0)
    return {
        "manual_t": t_manual,
        "scipy_t": float(scipy_res.statistic),
        "manual_p": p_manual,
        "scipy_p": float(scipy_res.pvalue),
        "df": df,
        "t_gap": abs(t_manual - scipy_res.statistic),
        "p_gap": abs(p_manual - scipy_res.pvalue),
    }


def compare_welch_with_scipy(data_a, data_b):
    t_manual, p_manual, df = two_sample_t_test_welch(data_a, data_b)
    scipy_res = stats.ttest_ind(data_a, data_b, equal_var=False)
    return {
        "manual_t": t_manual,
        "scipy_t": float(scipy_res.statistic),
        "manual_p": p_manual,
        "scipy_p": float(scipy_res.pvalue),
        "df": df,
        "t_gap": abs(t_manual - scipy_res.statistic),
        "p_gap": abs(p_manual - scipy_res.pvalue),
    }


def compare_anova_with_scipy(*groups):
    scratch = anova_components(*groups)
    scipy_res = stats.f_oneway(*groups)
    return scratch, {
        "scipy_F": float(scipy_res.statistic),
        "scipy_p": float(scipy_res.pvalue),
        "F_gap": abs(scratch["F_stat"] - scipy_res.statistic),
        "p_gap": abs(scratch["p_val"] - scipy_res.pvalue),
    }


def tukey_hsd_pairs(*groups, alpha=0.05, names=None):
    """
    Tukey–Kramer simultaneous pairwise comparisons.

    For groups i, j the studentized-range statistic is

        q_ij = |x̄_i - x̄_j| / sqrt(MS_W * (1/n_i + 1/n_j) / 2)

    and the simultaneous CI is

        (x̄_i - x̄_j) ± q_{α,k,N-k} * sqrt(MS_W * (1/n_i + 1/n_j) / 2).

    Reject H0: μ_i = μ_j at family-wise level α iff 0 is *outside* that interval.
    """
    groups = [np.asarray(g, dtype=float) for g in groups]
    anova = anova_components(*groups)
    k = len(groups)
    df_w = anova["df_W"]
    ms_w = anova["MS_W"]
    means = anova["group_means"]
    ns = anova["group_sizes"]
    if names is None:
        names = [f"G{i + 1}" for i in range(k)]
    q_crit = float(stats.studentized_range.ppf(1.0 - alpha, k, df_w))

    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = means[j] - means[i]
            se_q = np.sqrt(ms_w * (1.0 / ns[i] + 1.0 / ns[j]) / 2.0)
            margin = q_crit * se_q
            low, high = float(diff - margin), float(diff + margin)
            q_obs = float(abs(diff) / se_q) if se_q > 0 else 0.0
            p_val = float(stats.studentized_range.sf(q_obs, k, df_w)) if se_q > 0 else 1.0
            contains_zero = (low <= 0.0 <= high)
            pairs.append({
                "group_i": names[i],
                "group_j": names[j],
                "meandiff": float(diff),
                "lower": low,
                "upper": high,
                "q_obs": q_obs,
                "p_adj": p_val,
                "reject": (not contains_zero),
                "zero_in_interval": contains_zero,
            })
    return {
        "alpha": alpha,
        "k": k,
        "df_W": df_w,
        "MS_W": ms_w,
        "q_crit": q_crit,
        "group_means": dict(zip(names, means)),
        "pairs": pairs,
        "anova": anova,
    }


def family_wise_error_rate(alpha, n_tests):
    """FWER if n_tests independent tests are each run at level alpha."""
    return float(1.0 - (1.0 - alpha) ** n_tests)


def _t_pvalue(t_stat, df, alternative):
    """
    Convert a t statistic into a p-value via the null CDF / survival function.

    two-sided:  p = 2 * P(T ≥ |t|) = 2 * sf(|t|, df)
    greater:    p = P(T ≥ t)       = sf(t, df)   = 1 - F(t)
    less:       p = P(T ≤ t)       = cdf(t, df)  = F(t)
    """
    if alternative == "two-sided":
        return float(2.0 * stats.t.sf(np.abs(t_stat), df))
    if alternative == "greater":
        return float(stats.t.sf(t_stat, df))
    if alternative == "less":
        return float(stats.t.cdf(t_stat, df))
    raise ValueError(f"Unknown alternative: {alternative}")


def _z_pvalue(z_stat, alternative):
    if alternative == "two-sided":
        return float(2.0 * stats.norm.sf(np.abs(z_stat)))
    if alternative == "greater":
        return float(stats.norm.sf(z_stat))
    if alternative == "less":
        return float(stats.norm.cdf(z_stat))
    raise ValueError(f"Unknown alternative: {alternative}")
