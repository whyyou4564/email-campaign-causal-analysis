"""阶段三：实验平衡、率/均值差检验、多重修正与设计功效。"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep, proportion_confint,
    proportion_effectsize, proportions_ztest,
)

from .config import ARMS, PRE_COLUMNS, OUTCOMES, OUT, SEED

def mean_difference(treated, control, alpha=.05):
    """Welch 均值差，允许方差不同，区间使用 Satterthwaite 自由度。"""
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)
    variance_t = treated.var(ddof=1) / len(treated)
    variance_c = control.var(ddof=1) / len(control)
    standard_error = np.sqrt(variance_t + variance_c)
    difference = treated.mean() - control.mean()
    if standard_error == 0:
        return {
            "effect": float(difference), "ci_low": float(difference),
            "ci_high": float(difference), "se": 0.,
            "p_value": float(difference == 0),
        }
    dof = (variance_t + variance_c)**2 / (
        variance_t**2 / (len(treated) - 1) + variance_c**2 / (len(control) - 1)
    )
    width = stats.t.ppf(1 - alpha / 2, dof) * standard_error
    return {
        "effect": float(difference), "ci_low": float(difference - width),
        "ci_high": float(difference + width), "se": float(standard_error),
        "p_value": float(2 * stats.t.sf(abs(difference / standard_error), dof)),
    }

def rate_difference(treated, control, alpha=.05):
    """两比例 pooled z 检验，差值区间采用 Newcombe-Wilson。"""
    treated, control = np.asarray(treated), np.asarray(control)
    successes = [treated.sum(), control.sum()]
    sizes = [len(treated), len(control)]
    if sum(successes) in [0, sum(sizes)]:
        p_value = 1.
    else:
        _, p_value = proportions_ztest(successes, sizes)
    low, high = confint_proportions_2indep(
        successes[0], sizes[0], successes[1], sizes[1],
        method="newcomb", compare="diff", alpha=alpha,
    )
    return {
        "effect": float(treated.mean() - control.mean()),
        "ci_low": float(low), "ci_high": float(high),
        "se": float(np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))),
        "p_value": float(p_value),
    }

def balance(df):
    # 公开数据没有分配日志，1:1:1 必须明确是计划分配假设。
    counts = df.segment.value_counts().reindex(ARMS).to_numpy()
    srm = stats.chisquare(counts, np.repeat(len(df) / 3, 3))
    rows = []
    comparisons = [
        ("Mens E-Mail", "No E-Mail"), ("Womens E-Mail", "No E-Mail"),
        ("Mens E-Mail", "Womens E-Mail"),
    ]
    for treatment, control in comparisons:
        group_t, group_c = df[df.segment == treatment], df[df.segment == control]
        for feature in PRE_COLUMNS:
            levels = sorted(df[feature].unique()) if df[feature].dtype == "object" else [None]
            for level in levels:
                if level is None:
                    x, y = group_t[feature].astype(float), group_c[feature].astype(float)
                else:
                    x = (group_t[feature] == level).astype(float)
                    y = (group_c[feature] == level).astype(float)
                pooled_sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
                rows.append({
                    "treatment": treatment, "control": control,
                    "feature": feature, "level": level,
                    "mean_t": x.mean(), "mean_c": y.mean(),
                    "smd": (x.mean() - y.mean()) / pooled_sd if pooled_sd else 0,
                })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "balance_smd.csv", index=False)
    result = {
        "planned_ratio_assumption": "1:1:1; allocation logs unavailable",
        "srm_chi2": float(srm.statistic), "srm_p": float(srm.pvalue),
        "srm_flag": bool(srm.pvalue < .001),
        "max_abs_smd": float(table.smd.abs().max()), "threshold_smd": .1,
        "warning": "Balance and SRM checks cannot prove random assignment.",
    }
    (OUT / "experiment_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

def comparison_groups(df, treatment, control):
    if treatment == "Any E-Mail":
        group_t = df[df.segment != "No E-Mail"]
    else:
        group_t = df[df.segment == treatment]
    return group_t, df[df.segment == control]

def ab_analysis(df, boot_reps=2000):
    summaries, rows = [], []
    for arm in ARMS:
        group = df[df.segment == arm]
        for metric in OUTCOMES:
            y = group[metric]
            if metric == "spend":
                width = stats.t.ppf(.975, len(y) - 1) * y.std(ddof=1) / np.sqrt(len(y))
                low, high = y.mean() - width, y.mean() + width
            else:
                low, high = proportion_confint(y.sum(), len(y), method="wilson")
            summaries.append({
                "arm": arm, "metric": metric, "n": len(y), "total": y.sum(),
                "mean": y.mean(), "ci_low": low, "ci_high": high,
            })

    rng = np.random.default_rng(SEED)
    comparisons = [
        ("Mens E-Mail", "No E-Mail"), ("Womens E-Mail", "No E-Mail"),
        ("Mens E-Mail", "Womens E-Mail"), ("Any E-Mail", "No E-Mail"),
    ]
    for treatment, control in comparisons:
        group_t, group_c = comparison_groups(df, treatment, control)
        for metric in OUTCOMES:
            x, y = group_t[metric].to_numpy(), group_c[metric].to_numpy()
            method = mean_difference if metric == "spend" else rate_difference
            result = method(x, y)
            result.update({
                "treatment": treatment, "control": control, "metric": metric,
                "n_t": len(x), "n_c": len(y), "mean_t": x.mean(), "mean_c": y.mean(),
                "relative_lift": result["effect"] / y.mean(),
                "method": "Welch t" if metric == "spend" else "two-proportion z / Newcombe CI",
            })
            if metric == "spend":
                # 逐组重抽样，目标仍是包含零值的算术均值差。
                draws = [
                    rng.choice(x, len(x), replace=True).mean()
                    - rng.choice(y, len(y), replace=True).mean()
                    for _ in range(boot_reps)
                ]
                result["bootstrap_ci_low"], result["bootstrap_ci_high"] = np.quantile(draws, [.025, .975])
            rows.append(result)

    table = pd.DataFrame(rows)
    # 同一固定家族：四个对比 × 三个结果 = 十二个检验。
    table["p_holm"] = multipletests(table.p_value, method="holm")[1]
    table["reject_holm_05"] = table.p_holm < .05
    simultaneous_low, simultaneous_high = [], []
    for row in table.itertuples():
        group_t, group_c = comparison_groups(df, row.treatment, row.control)
        method = mean_difference if row.metric == "spend" else rate_difference
        interval = method(group_t[row.metric], group_c[row.metric], alpha=.05 / len(table))
        simultaneous_low.append(interval["ci_low"])
        simultaneous_high.append(interval["ci_high"])
    table["ci_low_bonferroni"], table["ci_high_bonferroni"] = simultaneous_low, simultaneous_high
    pd.DataFrame(summaries).to_csv(OUT / "group_metrics.csv", index=False)
    table.to_csv(OUT / "ab_results.csv", index=False)
    return table

def power_analysis(df):
    """未来实验的 MDE 场景规划，不以观测效应计算事后功效。"""
    baseline = df.loc[df.segment == "No E-Mail", "conversion"].mean()
    control_size = (df.segment == "No E-Mail").sum()
    engine = NormalIndPower()
    rows = []
    for ratio in [1, 2]:
        for alpha in [.05, .05 / 12]:
            for minimum_effect in [.002, .005, .01]:
                standardized_effect = abs(proportion_effectsize(baseline + minimum_effect, baseline))
                required = engine.solve_power(
                    standardized_effect, power=.8, alpha=alpha, ratio=ratio
                )
                rows.append({
                    "p0_planning_estimate": baseline, "absolute_mde": minimum_effect,
                    "ratio_t_to_c": ratio, "alpha": alpha, "power_target": .8,
                    "n_control_required": int(np.ceil(required)),
                    "n_treatment_required": int(np.ceil(required * ratio)),
                    "scenario_power_at_current_n": engine.power(
                        standardized_effect, control_size, alpha=alpha, ratio=ratio
                    ),
                })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "power_sample_size.csv", index=False)
    return result
