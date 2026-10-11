"""固定预算上限，保持验证集选择与测试集评价分离。"""
import json

import numpy as np
import pandas as pd

from .config import OUT, DEFAULT_COST, DEFAULT_MARGIN
from .evaluation import top_policy, policy_effect, paired_bootstrap

def budget_analysis():
    lock = json.loads((OUT / "policy_lock.json").read_text())
    validation = pd.read_csv(OUT / "validation_policy_selection.csv")
    test = pd.read_csv(OUT / "test_predictions.csv")
    y, treatment = test.conversion.to_numpy(), test.treatment.to_numpy()
    model = lock["selected_model"]
    policies, selections = {}, []

    # 预算场景是业务约束，不根据测试集结果挑选一个好看的上限。
    for cap in [.1, .3, .5]:
        available = validation[validation.fraction <= cap + 1e-8]
        chosen = available.sort_values(
            ["expected_profit_per_customer", "fraction"], ascending=[False, True]
        ).iloc[0]
        fraction = float(chosen.fraction)
        suffix = f"{int(cap * 100)}%"
        policy = top_policy(test[model].to_numpy(), fraction)
        policies[f"Selected {suffix}"] = policy
        policies[f"Random expected {suffix}"] = np.repeat(policy.mean(), len(y))
        policies[f"Response prediction {suffix}"] = top_policy(
            test["Response prediction"].to_numpy(), fraction
        )
        selections.append({"budget_cap": cap, "selected_fraction": fraction, "model": model})

    draws = paired_bootstrap(y, treatment, policies)
    rows, contrasts = [], []
    for selected in selections:
        suffix = f"{int(selected['budget_cap'] * 100)}%"
        for strategy in ["Selected", "Random expected", "Response prediction"]:
            name = f"{strategy} {suffix}"
            policy = policies[name]
            effect = policy_effect(y, treatment, policy)["incremental_rate_population"]
            profit_draws = 10000 * (
                DEFAULT_MARGIN * draws[name] - DEFAULT_COST * policy.mean()
            )
            low, high = np.quantile(profit_draws, [.025, .975])
            rows.append({
                **selected, "strategy": strategy,
                "incremental_conversions_per_10000": effect * 10000,
                "email_cost_per_10000": policy.mean() * 10000 * DEFAULT_COST,
                "net_profit_per_10000": effect * 10000 * DEFAULT_MARGIN - policy.mean() * 10000 * DEFAULT_COST,
                "profit_ci_low": low, "profit_ci_high": high,
            })
        for comparator in ["Random expected", "Response prediction"]:
            first, second = f"Selected {suffix}", f"{comparator} {suffix}"
            difference = 10000 * DEFAULT_MARGIN * (draws[first] - draws[second])
            low, high = np.quantile(difference, [.025, .975])
            estimate = policy_effect(y, treatment, policies[first] - policies[second])
            contrasts.append({
                **selected, "comparator": comparator,
                "profit_difference_per_10000": 10000 * DEFAULT_MARGIN * estimate["incremental_rate_population"],
                "ci_low": low, "ci_high": high,
                "interpretation": "exploratory budget scenario; fixed policy; individual 95% CI",
            })
    pd.DataFrame(rows).to_csv(OUT / "budget_locked_policies.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "budget_policy_contrasts.csv", index=False)
    print("BUDGET", pd.DataFrame(rows).to_string(index=False))
    return pd.DataFrame(rows), pd.DataFrame(contrasts)
