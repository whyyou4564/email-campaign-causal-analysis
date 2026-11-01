"""独立样本上的增益排序与离线投放评价。"""
import numpy as np
from scipy import stats
from sklift.metrics import qini_auc_score, uplift_auc_score
from statsmodels.stats.proportion import confint_proportions_2indep

from .config import SEED

def ranking(scores):
    """同分者用固定随机数打散；打散过程不读取 treatment 或 outcome。"""
    scores = np.asarray(scores, float)
    tie_breaker = np.random.default_rng(SEED).random(len(scores))
    return np.lexsort((tie_breaker, -scores))

def top_policy(scores, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    policy = np.zeros(len(scores), dtype=float)
    policy[ranking(scores)[:int(round(len(scores) * fraction))]] = 1
    return policy

def policy_effect(y, treatment, policy):
    """两组 masked outcome 均值差，等价以评估集处理比例做 IPW。"""
    y, treatment, policy = np.asarray(y), np.asarray(treatment), np.asarray(policy)
    treated = treatment == 1
    # 未被策略选中的客户结果置零，而非从分母中删除。
    outcome_t = (policy * y)[treated]
    outcome_c = (policy * y)[~treated]
    effect = outcome_t.mean() - outcome_c.mean()
    variance = outcome_t.var(ddof=1) / len(outcome_t) + outcome_c.var(ddof=1) / len(outcome_c)
    se = np.sqrt(variance)
    return {
        "incremental_rate_population": float(effect),
        "ci_low": float(effect - 1.96 * se),
        "ci_high": float(effect + 1.96 * se),
        "se": float(se),
    }

def gain_curve(y, treatment, scores):
    order = ranking(scores)
    y, treatment = np.asarray(y), np.asarray(treatment)
    probability = treatment.mean()
    transformed = y * treatment / probability - y * (1 - treatment) / (1 - probability)
    # 单位是“如果给全部入选客户发邮件，预计增加的购买人数”。
    gain = np.r_[0, np.cumsum(transformed[order])]
    fraction = np.arange(len(y) + 1) / len(y)
    return fraction, gain

def uplift_at_fraction(y, treatment, scores, fraction):
    selected = top_policy(scores, fraction).astype(bool)
    y, treatment = np.asarray(y), np.asarray(treatment)
    a = y[selected & (treatment == 1)]
    b = y[selected & (treatment == 0)]
    if not len(a) or not len(b):
        return {"uplift": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_t": len(a), "n_c": len(b)}
    low, high = confint_proportions_2indep(
        a.sum(), len(a), b.sum(), len(b), method="newcomb", compare="diff"
    )
    return {
        "uplift": float(a.mean() - b.mean()), "ci_low": float(low), "ci_high": float(high),
        "n_t": len(a), "n_c": len(b),
    }

def ranking_metrics(y, treatment, scores):
    fraction, gain = gain_curve(y, treatment, scores)
    baseline = fraction * gain[-1]
    area = np.trapezoid(gain / len(y), fraction)
    area_above_random = np.trapezoid((gain - baseline) / len(y), fraction)
    return {
        "auuc_raw_per_customer": float(area),
        "auuc_above_random_per_customer": float(area_above_random),
        "qini_normalized_sklift": float(qini_auc_score(y, scores, treatment)),
        "uplift_auc_normalized_sklift": float(uplift_auc_score(y, scores, treatment)),
        "uplift_at_10": uplift_at_fraction(y, treatment, scores, .1)["uplift"],
        "uplift_at_30": uplift_at_fraction(y, treatment, scores, .3)["uplift"],
    }

def paired_bootstrap(y, treatment, policies, repetitions=2000):
    """固定已学习策略，分组重抽样；不把训练不确定性纳入区间。"""
    y, treatment = np.asarray(y), np.asarray(treatment)
    group_t, group_c = np.flatnonzero(treatment == 1), np.flatnonzero(treatment == 0)
    rng = np.random.default_rng(SEED)
    draws = {name: np.empty(repetitions) for name in policies}
    for repeat in range(repetitions):
        sample_t = rng.choice(group_t, len(group_t), replace=True)
        sample_c = rng.choice(group_c, len(group_c), replace=True)
        for name, policy in policies.items():
            policy = np.asarray(policy)
            draws[name][repeat] = (
                (policy[sample_t] * y[sample_t]).mean()
                - (policy[sample_c] * y[sample_c]).mean()
            )
    return draws
