from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from itertools import product
from typing import Any


def wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (point, low, high). Robust at the extremes (0 or 1 successes)
    where the normal approximation degenerates, which is the regime of the
    near-ceiling and near-floor auditor recalls reported here.
    """

    if trials <= 0:
        return 0.0, 0.0, 0.0
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2 * trials)) / denom
    margin = (z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def mean(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers) if numbers else 0.0


def sample_std(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    if len(numbers) < 2:
        return 0.0
    center = mean(numbers)
    return math.sqrt(sum((value - center) ** 2 for value in numbers) / (len(numbers) - 1))


def bootstrap_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    draws: int = 2000,
    seed: int = 1729,
) -> tuple[float | None, float | None]:
    numbers = [float(value) for value in values]
    if not numbers:
        return None, None
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    rng = random.Random(seed)
    boot_means = []
    for _ in range(max(1, draws)):
        sample = [numbers[rng.randrange(len(numbers))] for _ in numbers]
        boot_means.append(mean(sample))
    boot_means.sort()
    alpha = max(0.0, min(1.0, 1.0 - confidence))
    lower_index = min(len(boot_means) - 1, max(0, int((alpha / 2.0) * len(boot_means))))
    upper_index = min(len(boot_means) - 1, max(0, int((1.0 - alpha / 2.0) * len(boot_means)) - 1))
    return boot_means[lower_index], boot_means[upper_index]


def summarize_metric(values: Iterable[float], *, prefix: str) -> dict[str, float | None]:
    numbers = [float(value) for value in values]
    ci_low, ci_high = bootstrap_ci(numbers)
    return {
        f"{prefix}_mean": mean(numbers),
        f"{prefix}_std": sample_std(numbers),
        f"{prefix}_ci_low": ci_low,
        f"{prefix}_ci_high": ci_high,
    }


def paired_bootstrap_difference(
    candidate_by_key: Mapping[Any, float],
    baseline_by_key: Mapping[Any, float],
    *,
    confidence: float = 0.95,
    draws: int = 2000,
    seed: int = 2026,
) -> dict[str, float | int | None]:
    keys = sorted(set(candidate_by_key) & set(baseline_by_key), key=str)
    differences = [float(candidate_by_key[key]) - float(baseline_by_key[key]) for key in keys]
    if not differences:
        return {
            "paired_n": 0,
            "mean_delta": None,
            "delta_ci_low": None,
            "delta_ci_high": None,
            "bootstrap_p_value": None,
            "permutation_p_value": None,
            "p_value": None,
            "cohens_d": None,
            "cliffs_delta": None,
        }
    ci_low, ci_high = bootstrap_ci(differences, confidence=confidence, draws=draws, seed=seed)
    rng = random.Random(seed + 1)
    boot_means = []
    for _ in range(max(1, draws)):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        boot_means.append(mean(sample))
    less_or_equal_zero = sum(1 for value in boot_means if value <= 0.0) / len(boot_means)
    greater_or_equal_zero = sum(1 for value in boot_means if value >= 0.0) / len(boot_means)
    bootstrap_p_value = min(1.0, 2.0 * min(less_or_equal_zero, greater_or_equal_zero))
    permutation_p_value = paired_permutation_p_value(differences, draws=draws, seed=seed + 2)
    return {
        "paired_n": len(differences),
        "mean_delta": mean(differences),
        "delta_ci_low": ci_low,
        "delta_ci_high": ci_high,
        "bootstrap_p_value": bootstrap_p_value,
        "permutation_p_value": permutation_p_value,
        "p_value": bootstrap_p_value,
        "cohens_d": paired_cohens_d(differences),
        "cliffs_delta": cliffs_delta(
            (candidate_by_key[key] for key in keys),
            (baseline_by_key[key] for key in keys),
        ),
    }


def benjamini_hochberg(p_values: Mapping[Any, float | None]) -> dict[Any, float | None]:
    """Benjamini-Hochberg FDR-adjusted q-values for a family of p-values.

    Keys with ``None`` p-values are passed through unchanged so callers can mix
    testable and untestable rows in one family.
    """

    testable = [(key, float(p)) for key, p in p_values.items() if p is not None]
    adjusted: dict[Any, float | None] = dict.fromkeys(p_values, None)
    if not testable:
        return adjusted
    testable.sort(key=lambda item: (item[1], str(item[0])))
    m = len(testable)
    running_min = 1.0
    raw = [min(1.0, p * m / rank) for rank, (_, p) in enumerate(testable, start=1)]
    for index in range(m - 1, -1, -1):
        running_min = min(running_min, raw[index])
        adjusted[testable[index][0]] = running_min
    return adjusted


def paired_cohens_d(differences: Iterable[float]) -> float | None:
    """Standardized effect size for paired deltas: mean(delta) / std(delta)."""

    numbers = [float(value) for value in differences]
    if len(numbers) < 2:
        return None
    center = mean(numbers)
    spread = sample_std(numbers)
    # Near-constant deltas: float noise makes spread tiny but nonzero, which
    # would explode the ratio; report it as undefined instead.
    if spread <= 1e-12 * max(1.0, abs(center)):
        return 0.0 if center == 0.0 else None
    return center / spread


def random_effects_meta(
    effects_by_group: Mapping[Any, tuple[float, float]],
) -> dict[str, float | int | None]:
    """DerSimonian-Laird random-effects pooling of per-group (effect, variance) pairs.

    Use for cross-scenario conclusions ("model X underperforms the baseline
    overall") where each scenario contributes its own paired-delta mean and
    squared standard error, and scenarios are treated as a random effect
    rather than averaged as if exchangeable.
    """

    pairs = [
        (float(effect), float(variance))
        for effect, variance in effects_by_group.values()
        if variance is not None and float(variance) > 0.0
    ]
    if not pairs:
        return {
            "group_count": 0,
            "pooled_effect": None,
            "pooled_ci_low": None,
            "pooled_ci_high": None,
            "tau_squared": None,
            "i_squared": None,
            "q_statistic": None,
        }
    weights = [1.0 / variance for _, variance in pairs]
    fixed_effect = sum(weight * effect for weight, (effect, _) in zip(weights, pairs)) / sum(weights)
    q_statistic = sum(weight * (effect - fixed_effect) ** 2 for weight, (effect, _) in zip(weights, pairs))
    df = len(pairs) - 1
    if df > 0:
        scale = sum(weights) - sum(weight**2 for weight in weights) / sum(weights)
        tau_squared = max(0.0, (q_statistic - df) / scale) if scale > 0.0 else 0.0
        i_squared = max(0.0, (q_statistic - df) / q_statistic) if q_statistic > 0.0 else 0.0
    else:
        tau_squared = 0.0
        i_squared = 0.0
    adjusted = [1.0 / (variance + tau_squared) for _, variance in pairs]
    pooled = sum(weight * effect for weight, (effect, _) in zip(adjusted, pairs)) / sum(adjusted)
    pooled_se = math.sqrt(1.0 / sum(adjusted))
    return {
        "group_count": len(pairs),
        "pooled_effect": pooled,
        "pooled_ci_low": pooled - 1.96 * pooled_se,
        "pooled_ci_high": pooled + 1.96 * pooled_se,
        "tau_squared": tau_squared,
        "i_squared": i_squared,
        "q_statistic": q_statistic,
    }


def kendall_tau(scores_a: Mapping[Any, float], scores_b: Mapping[Any, float]) -> float | None:
    """Kendall tau-b rank correlation over the shared keys of two score mappings.

    Used for ranking-stability questions: how much does an agent leaderboard
    reorder between two execution-assumption levels.
    """

    keys = sorted(set(scores_a) & set(scores_b), key=str)
    if len(keys) < 2:
        return None
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            delta_a = float(scores_a[keys[i]]) - float(scores_a[keys[j]])
            delta_b = float(scores_b[keys[i]]) - float(scores_b[keys[j]])
            if delta_a == 0.0 and delta_b == 0.0:
                continue
            if delta_a == 0.0:
                ties_a += 1
            elif delta_b == 0.0:
                ties_b += 1
            elif delta_a * delta_b > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_a) * (concordant + discordant + ties_b)
    )
    if denominator == 0.0:
        return None
    return (concordant - discordant) / denominator


def top_k_jaccard(scores_a: Mapping[Any, float], scores_b: Mapping[Any, float], *, k: int = 3) -> float | None:
    """Jaccard similarity of the top-k key sets under two score mappings."""

    keys = set(scores_a) & set(scores_b)
    if not keys or k <= 0:
        return None
    top_a = set(sorted(keys, key=lambda key: (-float(scores_a[key]), str(key)))[:k])
    top_b = set(sorted(keys, key=lambda key: (-float(scores_b[key]), str(key)))[:k])
    union = top_a | top_b
    if not union:
        return None
    return len(top_a & top_b) / len(union)


def variance_components(values_by_group: Mapping[Any, Iterable[float]]) -> dict[str, float | int | None]:
    """Decompose repeated measurements into between-group and within-group variance.

    For matrix runs, groups are market seeds and the within-group values are
    repeated provider samples at a fixed seed: between-group variance reflects
    market-path sensitivity, within-group variance reflects model stochasticity.
    Within-group variance is None unless at least one group has two samples.
    """

    groups = {key: [float(value) for value in values] for key, values in values_by_group.items()}
    groups = {key: values for key, values in groups.items() if values}
    if not groups:
        return {
            "group_count": 0,
            "total_n": 0,
            "between_group_variance": None,
            "within_group_variance": None,
            "within_group_share": None,
        }
    group_means = [mean(values) for values in groups.values()]
    within_list = [sample_std(values) ** 2 for values in groups.values() if len(values) >= 2]
    within = mean(within_list) if within_list else None
    # The variance of the group means estimates sigma_b^2 + sigma_w^2/n, not
    # sigma_b^2, so using it directly inflates the denominator and biases the
    # within-group share downward -- i.e. it makes sampling noise look smaller
    # than it is. Subtract the sampling component, floored at zero because the
    # unbiased estimator can go negative when the true between variance is near
    # zero.
    between = None
    if len(group_means) >= 2:
        naive_between = sample_std(group_means) ** 2
        sizes = [len(values) for values in groups.values() if len(values) >= 2]
        if within is not None and sizes:
            mean_size = mean([float(size) for size in sizes])
            between = max(0.0, naive_between - within / mean_size)
        else:
            between = naive_between
    share: float | None = None
    if between is not None and within is not None and (between + within) > 0.0:
        share = within / (between + within)
    return {
        "group_count": len(groups),
        "total_n": sum(len(values) for values in groups.values()),
        "between_group_variance": between,
        "within_group_variance": within,
        "within_group_share": share,
    }


def cliffs_delta(candidate: Iterable[float], baseline: Iterable[float]) -> float | None:
    """Nonparametric effect size in [-1, 1]: P(candidate > baseline) - P(candidate < baseline)."""

    left = [float(value) for value in candidate]
    right = [float(value) for value in baseline]
    if not left or not right:
        return None
    greater = sum(1 for a in left for b in right if a > b)
    lesser = sum(1 for a in left for b in right if a < b)
    return (greater - lesser) / (len(left) * len(right))


def paired_permutation_p_value(
    differences: Iterable[float],
    *,
    draws: int = 2000,
    seed: int = 2028,
) -> float | None:
    """Two-sided paired sign-flip permutation p-value for matched run deltas."""

    numbers = [float(value) for value in differences]
    if not numbers:
        return None
    observed = abs(mean(numbers))
    if observed == 0.0:
        return 1.0
    tolerance = 1e-15
    if len(numbers) <= 16:
        total = 0
        extreme = 0
        for signs in product((-1.0, 1.0), repeat=len(numbers)):
            value = abs(mean(value * sign for value, sign in zip(numbers, signs)))
            total += 1
            if value + tolerance >= observed:
                extreme += 1
        return extreme / total

    rng = random.Random(seed)
    total = max(1, draws)
    extreme = 0
    for _ in range(total):
        value = abs(mean(value if rng.random() < 0.5 else -value for value in numbers))
        if value + tolerance >= observed:
            extreme += 1
    # A finite Monte Carlo sample cannot establish a zero tail probability.
    # Include the observed allocation as an additional draw (the standard
    # plus-one correction) so downstream tables never report an impossible
    # p=0 merely because no sampled sign vector was as extreme.
    return (extreme + 1) / (total + 1)
