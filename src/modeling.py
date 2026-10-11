"""阶段五、六：预测基线、S/T/X Learner，以及预算模拟。"""
import json
import platform
import importlib.metadata

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import FEATURES, SEED, OUT, DEFAULT_COST, DEFAULT_MARGIN
from .evaluation import (
    ranking, ranking_metrics, gain_curve, top_policy, policy_effect,
    uplift_at_fraction, paired_bootstrap,
)

def feature_frame(df):
    """白名单防止 visit/conversion/spend/segment 意外进入模型。"""
    features = df.loc[:, FEATURES].copy()
    features["history"] = np.log1p(features["history"])
    return features

def make_preprocessor():
    return ColumnTransformer([
        ("numeric", StandardScaler(), ["recency", "history", "mens", "womens", "newbie"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["zip_code", "channel"]),
    ])

def classifier():
    # 固定保守参数；不做测试集调参、不因稀少购买事件追逐复杂模型。
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=.05, max_leaf_nodes=7,
        min_samples_leaf=300, l2_regularization=20,
        early_stopping=False, random_state=SEED,
    )

class SLearner:
    def fit(self, x, y, treatment):
        self.model = classifier().fit(np.column_stack([x, treatment]), y)
        return self

    def predict(self, x):
        p1 = self.model.predict_proba(np.column_stack([x, np.ones(len(x))]))[:, 1]
        p0 = self.model.predict_proba(np.column_stack([x, np.zeros(len(x))]))[:, 1]
        return p1 - p0

class TLearner:
    def fit(self, x, y, treatment):
        self.control = classifier().fit(x[treatment == 0], y[treatment == 0])
        self.treated = classifier().fit(x[treatment == 1], y[treatment == 1])
        return self

    def predict(self, x):
        return self.treated.predict_proba(x)[:, 1] - self.control.predict_proba(x)[:, 1]

class XLearner:
    def fit(self, x, y, treatment):
        # 三折交叉拟合响应模型，避免用本行训练过的预测生成本行伪效应。
        self.probability = treatment.mean()
        p0, p1 = np.zeros(len(y)), np.zeros(len(y))
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        for train, holdout in folds.split(x, treatment):
            arm_train = treatment[train]
            model0 = classifier().fit(x[train][arm_train == 0], y[train][arm_train == 0])
            model1 = classifier().fit(x[train][arm_train == 1], y[train][arm_train == 1])
            p0[holdout] = model0.predict_proba(x[holdout])[:, 1]
            p1[holdout] = model1.predict_proba(x[holdout])[:, 1]
        pseudo0 = p1[treatment == 0] - y[treatment == 0]
        pseudo1 = y[treatment == 1] - p0[treatment == 1]
        base = HistGradientBoostingRegressor(
            max_iter=120, learning_rate=.05, max_leaf_nodes=7,
            min_samples_leaf=400, l2_regularization=30,
            early_stopping=False, random_state=SEED,
        )
        self.effect0 = clone(base).fit(x[treatment == 0], pseudo0)
        self.effect1 = clone(base).fit(x[treatment == 1], pseudo1)
        return self

    def predict(self, x):
        # 标准 X-Learner 组合：e*tau0 + (1-e)*tau1。
        return self.probability * self.effect0.predict(x) + (1 - self.probability) * self.effect1.predict(x)

def split_data(df):
    # 按实验分组分层，绝不按结果变量寻找有利的划分。
    train_ids, temporary_ids = train_test_split(
        df.index.to_numpy(), test_size=.4, stratify=df.segment, random_state=SEED
    )
    validation_ids, test_ids = train_test_split(
        temporary_ids, test_size=.5, stratify=df.loc[temporary_ids, "segment"],
        random_state=SEED + 1,
    )
    return train_ids, validation_ids, test_ids

def run_models(df):
    train_ids, validation_ids, test_ids = split_data(df)
    split_rows = []
    for split_name, ids in [
        ("train", train_ids), ("validation", validation_ids), ("test", test_ids)
    ]:
        for arm, group in df.loc[ids].groupby("segment"):
            split_rows.append({
                "split": split_name, "arm": arm, "n": len(group),
                "conversions": int(group.conversion.sum()), "visits": int(group.visit.sum()),
            })
    pd.DataFrame(split_rows).to_csv(OUT / "split_summary.csv", index=False)
    pd.DataFrame({
        "row_id": np.r_[train_ids, validation_ids, test_ids],
        "split": ["train"] * len(train_ids) + ["validation"] * len(validation_ids) + ["test"] * len(test_ids),
    }).to_csv(OUT / "split_assignments.csv", index=False)

    preprocessor = make_preprocessor()
    x_train = preprocessor.fit_transform(feature_frame(df.loc[train_ids]))
    x_validation = preprocessor.transform(feature_frame(df.loc[validation_ids]))
    x_test = preprocessor.transform(feature_frame(df.loc[test_ids]))
    t_train = (df.loc[train_ids, "segment"] != "No E-Mail").to_numpy(int)
    t_validation = (df.loc[validation_ids, "segment"] != "No E-Mail").to_numpy(int)
    t_test = (df.loc[test_ids, "segment"] != "No E-Mail").to_numpy(int)
    score_rows, curve_rows, decile_rows, predictive_rows = [], [], [], []
    outcomes = {}
    for metric in ["conversion", "visit"]:
        print(f"MODELING {metric}: train={len(train_ids)}, validation={len(validation_ids)}, test={len(test_ids)}", flush=True)
        y_train = df.loc[train_ids, metric].to_numpy()
        y_validation = df.loc[validation_ids, metric].to_numpy()
        y_test = df.loc[test_ids, metric].to_numpy()
        baseline = LogisticRegression(C=1., max_iter=1000, random_state=SEED).fit(x_train, y_train)
        score_sets = {
            "validation": {"Response prediction": baseline.predict_proba(x_validation)[:, 1]},
            "test": {"Response prediction": baseline.predict_proba(x_test)[:, 1]},
        }
        # visit 任务的响应模型仍用同一标签名，仅为统一表结构。
        for name, learner in [("S-Learner", SLearner()), ("T-Learner", TLearner()), ("X-Learner", XLearner())]:
            learner.fit(x_train, y_train, t_train)
            score_sets["validation"][name] = learner.predict(x_validation)
            score_sets["test"][name] = learner.predict(x_test)
        for split_name, y, t in [
            ("validation", y_validation, t_validation), ("test", y_test, t_test)
        ]:
            for name, scores in score_sets[split_name].items():
                score_rows.append({
                    "outcome": metric, "split": split_name, "model": name,
                    **ranking_metrics(y, t, scores),
                })
            response_probability = score_sets[split_name]["Response prediction"]
            predictive_rows.append({
                "outcome": metric, "split": split_name,
                "roc_auc": roc_auc_score(y, response_probability),
                "pr_auc": average_precision_score(y, response_probability),
                "prevalence": y.mean(),
                "brier_score": brier_score_loss(y, response_probability),
            })
        # 测试集曲线及分位数组只用于报告，不用于选模型或选择投放比例。
        for name, scores in score_sets["test"].items():
            fractions, gains = gain_curve(y_test, t_test, scores)
            for index in np.linspace(0, len(y_test), 201).astype(int):
                curve_rows.append({
                    "outcome": metric, "model": name,
                    "fraction": fractions[index], "incremental_conversions": gains[index],
                })
            order = ranking(scores)
            for decile, ids in enumerate(np.array_split(order, 10), start=1):
                a, b = y_test[ids][t_test[ids] == 1], y_test[ids][t_test[ids] == 0]
                interval = uplift_at_fraction(y_test[ids], t_test[ids], scores[ids], 1)
                decile_rows.append({
                    "outcome": metric, "model": name, "decile": decile,
                    "n_t": len(a), "n_c": len(b),
                    "events_t": a.sum(), "events_c": b.sum(),
                    "mean_predicted_score": scores[ids].mean(),
                    "observed_uplift": interval["uplift"],
                    "ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
                })
        outcomes[metric] = (score_sets, y_validation, y_test)

    scores_table = pd.DataFrame(score_rows)
    scores_table.to_csv(OUT / "uplift_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUT / "uplift_curves.csv", index=False)
    pd.DataFrame(decile_rows).to_csv(OUT / "uplift_deciles.csv", index=False)
    pd.DataFrame(predictive_rows).to_csv(OUT / "prediction_metrics.csv", index=False)

    scores, y_validation, y_test = outcomes["conversion"]
    validation_metrics = scores_table[
        (scores_table.outcome == "conversion") & (scores_table.split == "validation")
        & (scores_table.model != "Response prediction")
    ]
    chosen_model = validation_metrics.sort_values(
        "auuc_above_random_per_customer", ascending=False
    ).iloc[0]["model"]

    # 根据验证集净收益选比例，包括完全不发和全量发送。
    validation_policy_rows = []
    for fraction in np.linspace(0, 1, 11):
        policy = top_policy(scores["validation"][chosen_model], fraction)
        result = policy_effect(y_validation, t_validation, policy)
        profit = DEFAULT_MARGIN * result["incremental_rate_population"] - DEFAULT_COST * policy.mean()
        validation_policy_rows.append({
            "fraction": fraction, "model": chosen_model,
            **result, "expected_profit_per_customer": profit,
        })
    validation_policy_table = pd.DataFrame(validation_policy_rows)
    chosen_fraction = float(validation_policy_table.sort_values(
        ["expected_profit_per_customer", "fraction"], ascending=[False, True]
    ).iloc[0]["fraction"])
    validation_policy_table.to_csv(OUT / "validation_policy_selection.csv", index=False)

    lock = {
        "selected_model": chosen_model, "selected_fraction": chosen_fraction,
        "selected_on": "validation only; no test tuning",
        "selection_rule": "uplift model with largest validation AUUC above random, then fraction with largest validation profit",
        "cost_usd_per_email": DEFAULT_COST, "assumed_margin_usd_per_incremental_purchase": DEFAULT_MARGIN,
        "mixture_policy": "Any email retains randomized Mens/Womens mixture; does not select email content.",
        "features": FEATURES, "seed": SEED,
        "hyperparameters_fixed": True,
    }
    # 保存选择，再开展测试集业务评价。
    (OUT / "policy_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")

    n_test = len(y_test)
    deployment_n = 10000
    simulation_rows = []
    rng = np.random.default_rng(SEED)
    random_scores = rng.random(n_test)
    test_scores = scores["test"]
    for fraction in np.linspace(0, 1, 11):
        policies = {
            "Random expected": np.repeat(fraction, n_test),
            "Random realized": top_policy(random_scores, fraction),
            "Response prediction": top_policy(test_scores["Response prediction"], fraction),
            chosen_model: top_policy(test_scores[chosen_model], fraction),
        }
        for strategy, policy in policies.items():
            result = policy_effect(y_test, t_test, policy)
            messages = deployment_n * policy.mean()
            gains = deployment_n * result["incremental_rate_population"]
            cost = messages * DEFAULT_COST
            simulation_rows.append({
                "strategy": strategy, "fraction": fraction, "population": deployment_n,
                "expected_emails": messages, "incremental_conversions": gains,
                "conversion_ci_low": deployment_n * result["ci_low"],
                "conversion_ci_high": deployment_n * result["ci_high"],
                "email_cost_usd": cost,
                "net_profit_usd": gains * DEFAULT_MARGIN - cost,
                "net_profit_ci_low": deployment_n * DEFAULT_MARGIN * result["ci_low"] - cost,
                "net_profit_ci_high": deployment_n * DEFAULT_MARGIN * result["ci_high"] - cost,
            })
    simulations = pd.DataFrame(simulation_rows)
    simulations.to_csv(OUT / "business_simulation.csv", index=False)

    # 固定验证集选出的比例，做与随机投放、响应预测及全量发送的成对比较。
    chosen_policy = top_policy(test_scores[chosen_model], chosen_fraction)
    policies = {
        "Selected policy": chosen_policy,
        "Random expected same budget": np.repeat(chosen_policy.mean(), n_test),
        "Response prediction same budget": top_policy(test_scores["Response prediction"], chosen_fraction),
        "All send": np.ones(n_test),
        "No send": np.zeros(n_test),
    }
    draws = paired_bootstrap(y_test, t_test, policies)
    fixed_rows = []
    for name, policy in policies.items():
        effect = policy_effect(y_test, t_test, policy)["incremental_rate_population"]
        profits = deployment_n * (
            DEFAULT_MARGIN * draws[name] - DEFAULT_COST * policy.mean()
        )
        low, high = np.quantile(profits, [.025, .975])
        fixed_rows.append({
            "policy": name, "fraction": policy.mean(),
            "incremental_conversions_per_10000": deployment_n * effect,
            "net_profit_per_10000": deployment_n * (
                DEFAULT_MARGIN * effect - DEFAULT_COST * policy.mean()
            ),
            "bootstrap_profit_ci_low": low, "bootstrap_profit_ci_high": high,
        })
    pd.DataFrame(fixed_rows).to_csv(OUT / "locked_policy_test.csv", index=False)
    contrast_rows = []
    for comparator in ["Random expected same budget", "Response prediction same budget", "All send"]:
        paired_profits = deployment_n * (
            DEFAULT_MARGIN * (draws["Selected policy"] - draws[comparator])
            - DEFAULT_COST * (chosen_policy.mean() - policies[comparator].mean())
        )
        low, high = np.quantile(paired_profits, [.025, .975])
        contrast_rows.append({
            "comparator": comparator, "profit_difference_mean": paired_profits.mean(),
            "ci_low": low, "ci_high": high,
            "method": "paired arm-stratified bootstrap; fixed learned policies; individual 95% CI",
        })
    pd.DataFrame(contrast_rows).to_csv(OUT / "policy_contrasts.csv", index=False)

    # 敏感性场景中的最佳比例仍只在验证集选择，测试集仅评价。
    sensitivity_rows = []
    for cost in [.005, .02, .10, .50]:
        for margin in [10., 30., 60.]:
            validation_profits = (
                validation_policy_table.incremental_rate_population * margin
                - validation_policy_table.fraction * cost
            )
            fraction = float(validation_policy_table.loc[validation_profits.idxmax(), "fraction"])
            policy = top_policy(test_scores[chosen_model], fraction)
            result = policy_effect(y_test, t_test, policy)
            sensitivity_rows.append({
                "email_cost_usd": cost, "margin_usd": margin,
                "fraction_selected_on_validation": fraction,
                "test_profit_per_10000": deployment_n * (
                    result["incremental_rate_population"] * margin - policy.mean() * cost
                ),
            })
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "sensitivity_analysis.csv", index=False)

    pd.DataFrame({
        "row_id": test_ids, "treatment": t_test, "conversion": y_test,
        **{name: value for name, value in test_scores.items()},
    }).to_csv(OUT / "test_predictions.csv", index=False)
    environment = {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["pandas", "numpy", "scipy", "statsmodels", "scikit-learn", "scikit-uplift", "matplotlib"]
        },
    }
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print("POLICY LOCK", json.dumps(lock))
    print(pd.DataFrame(fixed_rows).to_string(index=False))
    return scores_table, lock
