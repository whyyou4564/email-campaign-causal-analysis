"""阶段四：ITT 效应、协变量调整和探索性异质性。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .config import FEATURES, OUTCOMES, OUT
from .statistics import rate_difference, mean_difference

def select_comparison(df, treatment):
    if treatment == "Any E-Mail":
        subset = df.copy()
    else:
        subset = df[df.segment.isin([treatment, "No E-Mail"])].copy()
    indicator = (subset.segment != "No E-Mail").astype(int)
    return subset, indicator

def centered_covariates(df):
    features = df[FEATURES].copy()
    features["history"] = np.log1p(features["history"])
    features = pd.get_dummies(features, drop_first=True, dtype=float).astype(float)
    return features - features.mean()

def causal_analysis(df):
    adjusted_rows, subgroup_rows, interaction_rows = [], [], []
    # 预先固定的分群，不按结果寻找最佳切点。
    strata = {
        "newbie": df.newbie.astype(str),
        "mens": df.mens.astype(str),
        "womens": df.womens.astype(str),
        "recency": pd.Series(np.where(df.recency <= 6, "1-6 months", "7-12 months"), index=df.index),
        "history": pd.Series(np.where(df.history <= 200, "<=200 USD", ">200 USD"), index=df.index),
        "channel": df.channel,
    }
    for treatment in ["Any E-Mail", "Mens E-Mail", "Womens E-Mail"]:
        subset, indicator = select_comparison(df, treatment)
        covariates = centered_covariates(subset)
        design = pd.concat([
            pd.Series(1.0, index=subset.index, name="intercept"),
            indicator.rename("treatment"),
            covariates.add_prefix("x_"),
            covariates.mul(indicator, axis=0).add_prefix("interaction_"),
        ], axis=1)
        for metric in OUTCOMES:
            # Lin 完全交互 OLS，HC2 处理异方差；概率差不等于 logistic 的 OR。
            fitted = sm.OLS(subset[metric], design).fit(cov_type="HC2")
            interval = fitted.conf_int().loc["treatment"]
            adjusted_rows.append({
                "treatment": treatment, "metric": metric,
                "adjusted_ate": fitted.params["treatment"],
                "ci_low": interval.iloc[0], "ci_high": interval.iloc[1],
                "p_value": fitted.pvalues["treatment"],
                "method": "Lin fully interacted OLS; centered pre-features; HC2",
            })
            for feature, labels_all in strata.items():
                labels = labels_all.loc[subset.index]
                dummy = pd.get_dummies(labels, drop_first=True, dtype=float)
                subgroup_design = pd.concat([
                    pd.Series(1.0, index=subset.index, name="intercept"),
                    indicator.rename("treatment"),
                    dummy.add_prefix("group_"),
                    dummy.mul(indicator, axis=0).add_prefix("heterogeneity_"),
                ], axis=1)
                fit = sm.OLS(subset[metric], subgroup_design).fit(cov_type="HC2")
                names = [name for name in subgroup_design if name.startswith("heterogeneity_")]
                restriction = np.zeros((len(names), len(subgroup_design.columns)))
                for number, name in enumerate(names):
                    restriction[number, subgroup_design.columns.get_loc(name)] = 1
                # 直接检验组间效应差，避免“一组显著、另一组不显著”的错误推断。
                wald = fit.wald_test(restriction, scalar=True)
                interaction_rows.append({
                    "treatment": treatment, "metric": metric,
                    "feature": feature, "p_value": float(wald.pvalue),
                })
                for level in sorted(labels.unique()):
                    group = subset[labels == level]
                    a = group.loc[group.segment != "No E-Mail", metric]
                    b = group.loc[group.segment == "No E-Mail", metric]
                    effect = (mean_difference if metric == "spend" else rate_difference)(a, b)
                    subgroup_rows.append({
                        "treatment": treatment, "metric": metric,
                        "feature": feature, "level": level,
                        "n_t": len(a), "n_c": len(b),
                        "mean_t": a.mean(), "mean_c": b.mean(), **effect,
                    })
    adjusted = pd.DataFrame(adjusted_rows)
    adjusted["p_holm"] = multipletests(adjusted.p_value, method="holm")[1]
    subgroups = pd.DataFrame(subgroup_rows)
    subgroups["q_bh_exploratory"] = multipletests(subgroups.p_value, method="fdr_bh")[1]
    interactions = pd.DataFrame(interaction_rows)
    interactions["q_bh_exploratory"] = multipletests(interactions.p_value, method="fdr_bh")[1]
    adjusted.to_csv(OUT / "ate_adjusted.csv", index=False)
    subgroups.to_csv(OUT / "heterogeneous_effects.csv", index=False)
    interactions.to_csv(OUT / "heterogeneity_tests.csv", index=False)
    print("CAUSAL: adjusted ATE, subgroup effects and interaction tests saved.")
    return adjusted, subgroups, interactions
