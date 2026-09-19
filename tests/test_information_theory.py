import numpy as np

from src.linear_algebra_and_stat import information_theory as it


def test_self_information_and_uniform_entropy():
    assert it.self_information(1 / 8) == 3.0
    assert it.entropy([0.25] * 4) == 2.0


def test_chain_rule_for_joint_entropy():
    joint = np.array([[0.45, 0.05], [0.10, 0.40]])
    h_joint = it.joint_entropy(joint)
    h_x = it.entropy(joint.sum(axis=1))
    assert np.isclose(h_joint, h_x + it.conditional_entropy(joint, target="y"))


def test_mutual_information_limiting_cases():
    independent = np.array([[0.25, 0.25], [0.25, 0.25]])
    deterministic = np.array([[0.5, 0.0], [0.0, 0.5]])
    assert np.isclose(it.mutual_information(independent), 0.0)
    assert np.isclose(it.mutual_information(deterministic), 1.0)
    assert np.isclose(it.conditional_entropy(deterministic), 0.0)


def test_cross_entropy_kl_identity():
    p = np.array([0.6, 0.25, 0.1, 0.05])
    q = np.array([0.45, 0.3, 0.15, 0.1])
    assert np.isclose(
        it.cross_entropy(p, q),
        it.entropy(p) + it.kl_divergence(p, q),
    )


def test_categorical_cross_entropy_selects_true_class():
    y = np.array([0, 0, 1])
    q = np.array([0.2, 0.3, 0.5])
    assert np.isclose(it.categorical_cross_entropy(y, q), -np.log(0.5))


def test_substitution_preserves_character_information():
    text = "information theory studies uncertainty " * 20
    cipher, _ = it.monoalphabetic_substitution(text)
    plain_adj = it.adjacent_character_information(text)
    cipher_adj = it.adjacent_character_information(cipher)
    assert np.isclose(
        it.character_information(text)["entropy"],
        it.character_information(cipher)["entropy"],
    )
    assert np.isclose(plain_adj["MI_adjacent"], cipher_adj["MI_adjacent"])


def test_information_gain_equals_mutual_information():
    feature = [0, 0, 1, 1, 1, 0]
    target = [0, 0, 1, 1, 0, 0]
    result = it.information_gain(feature, target)
    assert np.isclose(result["information_gain"], result["mutual_information"])
