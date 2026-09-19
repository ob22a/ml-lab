"""From-scratch information-theory utilities used by Notebook 06.

Discrete quantities are exact for the supplied probability tables. Quantities
estimated from samples (especially continuous mutual information) are explicitly
kept separate in the notebook.
"""

from collections import Counter
import math
import unicodedata
import numpy as np


def validate_probabilities(probabilities, *, atol=1e-10):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("probabilities must be a non-empty 1D array")
    if np.any(p < 0):
        raise ValueError("probabilities cannot be negative")
    if not np.isclose(p.sum(), 1.0, atol=atol):
        raise ValueError(f"probabilities must sum to 1, got {p.sum()}")
    return p


def self_information(probability, base=2):
    """Information in one outcome: -log_base(p)."""
    p = np.asarray(probability, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    with np.errstate(divide="ignore"):
        values = -np.log(p) / np.log(base)
    return float(values) if values.ndim == 0 else values


def entropy(probabilities, base=2):
    """Shannon entropy, with the convention 0 log 0 = 0."""
    p = validate_probabilities(probabilities)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log(p[positive]) / np.log(base)))


def empirical_distribution(values):
    values = list(values)
    if not values:
        raise ValueError("values cannot be empty")
    counts = Counter(values)
    keys = sorted(counts, key=str)
    probabilities = np.array([counts[k] / len(values) for k in keys])
    return keys, probabilities


def joint_distribution(x, y):
    x, y = list(x), list(y)
    if len(x) != len(y) or not x:
        raise ValueError("x and y must have equal, positive length")
    x_values, y_values = sorted(set(x), key=str), sorted(set(y), key=str)
    xi, yi = {v: i for i, v in enumerate(x_values)}, {v: i for i, v in enumerate(y_values)}
    table = np.zeros((len(x_values), len(y_values)), dtype=float)
    for xv, yv in zip(x, y):
        table[xi[xv], yi[yv]] += 1
    table /= len(x)
    return x_values, y_values, table


def validate_joint(joint, *, atol=1e-10):
    pxy = np.asarray(joint, dtype=float)
    if pxy.ndim != 2 or np.any(pxy < 0):
        raise ValueError("joint must be a non-negative 2D probability table")
    if not np.isclose(pxy.sum(), 1.0, atol=atol):
        raise ValueError(f"joint probabilities must sum to 1, got {pxy.sum()}")
    return pxy


def marginals(joint):
    pxy = validate_joint(joint)
    return pxy.sum(axis=1), pxy.sum(axis=0)


def joint_entropy(joint, base=2):
    return entropy(validate_joint(joint).ravel(), base=base)


def conditional_distributions(joint, condition_on="x"):
    pxy = validate_joint(joint)
    if condition_on == "x":
        px = pxy.sum(axis=1, keepdims=True)
        return np.divide(pxy, px, out=np.zeros_like(pxy), where=px > 0)
    if condition_on == "y":
        py = pxy.sum(axis=0, keepdims=True)
        return np.divide(pxy, py, out=np.zeros_like(pxy), where=py > 0)
    raise ValueError("condition_on must be 'x' or 'y'")


def conditional_entropy(joint, target="y", base=2):
    """H(Y|X) when target='y'; H(X|Y) when target='x'."""
    pxy = validate_joint(joint)
    px, py = marginals(pxy)
    if target == "y":
        cond = conditional_distributions(pxy, "x")
        return float(sum(px[i] * entropy(row, base) for i, row in enumerate(cond) if px[i] > 0))
    if target == "x":
        cond = conditional_distributions(pxy, "y")
        return float(sum(py[j] * entropy(cond[:, j], base) for j in range(len(py)) if py[j] > 0))
    raise ValueError("target must be 'x' or 'y'")


def mutual_information(joint, base=2):
    """Exact discrete I(X;Y) from a joint probability table."""
    pxy = validate_joint(joint)
    px, py = marginals(pxy)
    product = px[:, None] * py[None, :]
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / product[mask]) / np.log(base)))


def cross_entropy(p, q, base=2):
    p = validate_probabilities(p)
    q = validate_probabilities(q)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if np.any((p > 0) & (q == 0)):
        return math.inf
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(q[mask]) / np.log(base)))


def kl_divergence(p, q, base=2):
    p = validate_probabilities(p)
    q = validate_probabilities(q)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if np.any((p > 0) & (q == 0)):
        return math.inf
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask]) / np.log(base)))


def categorical_cross_entropy(y_true, probabilities, base=math.e, eps=1e-15):
    y = np.asarray(y_true, dtype=float)
    q = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0)
    if y.shape != q.shape:
        raise ValueError("one-hot labels and probabilities must have equal shape")
    losses = -np.sum(y * np.log(q) / np.log(base), axis=-1)
    return float(losses) if losses.ndim == 0 else losses


def binary_cross_entropy(y_true, probability_one, base=math.e, eps=1e-15):
    y = np.asarray(y_true, dtype=float)
    q = np.clip(np.asarray(probability_one, dtype=float), eps, 1 - eps)
    losses = -(y * np.log(q) + (1 - y) * np.log(1 - q)) / np.log(base)
    return float(losses) if losses.ndim == 0 else losses


def softmax(logits):
    z = np.asarray(logits, dtype=float)
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def preprocess_text(text, *, lowercase=True, keep_spaces=True, keep_punctuation=False):
    """Unicode-aware preprocessing; never transliterates or strips script letters."""
    text = unicodedata.normalize("NFC", text)
    if lowercase:
        text = text.lower()
    output = []
    previous_space = False
    for char in text:
        category = unicodedata.category(char)
        if char.isspace():
            if keep_spaces and not previous_space:
                output.append(" ")
            previous_space = True
        elif category.startswith(("L", "M")) or category.startswith("N"):
            output.append(char)
            previous_space = False
        elif category.startswith("P"):
            if keep_punctuation:
                output.append(char)
                previous_space = False
            else:
                if keep_spaces and not previous_space:
                    output.append(" ")
                    previous_space = True
    return "".join(output).strip()


def character_information(text):
    symbols, p = empirical_distribution(text)
    return {"symbols": symbols, "probabilities": p, "entropy": entropy(p)}


def adjacent_character_information(text):
    if len(text) < 2:
        raise ValueError("text must contain at least two symbols")
    current, following = text[:-1], text[1:]
    x_values, y_values, joint = joint_distribution(current, following)
    py = joint.sum(axis=0)
    h_next = entropy(py)
    h_next_given_current = conditional_entropy(joint, target="y")
    mi = mutual_information(joint)
    return {
        "current_symbols": x_values,
        "next_symbols": y_values,
        "joint": joint,
        "H_next": h_next,
        "H_next_given_current": h_next_given_current,
        "MI_adjacent": mi,
        "MI_identity": entropy(joint.sum(axis=1)) + entropy(py) - joint_entropy(joint),
    }


def monoalphabetic_substitution(text, seed=42):
    symbols = sorted(set(text), key=str)
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(symbols))
    mapping = dict(zip(symbols, shuffled))
    return "".join(mapping[c] for c in text), mapping


def information_gain(feature, target):
    """Exact discrete information gain H(Y)-H(Y|X), plus its components."""
    x_values, y_values, joint = joint_distribution(feature, target)
    h_y = entropy(joint.sum(axis=0))
    h_y_given_x = conditional_entropy(joint, target="y")
    return {
        "feature_values": x_values,
        "target_values": y_values,
        "joint": joint,
        "H_target": h_y,
        "H_target_given_feature": h_y_given_x,
        "information_gain": h_y - h_y_given_x,
        "mutual_information": mutual_information(joint),
    }


def quantile_bins(values, n_bins=5):
    """Dependency-light quantile binning for controlled exact-discrete MI."""
    values = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(values[~np.isnan(values)], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.zeros(len(values), dtype=int), edges
    bins = np.digitize(values, edges[1:-1], right=True)
    return bins, edges
