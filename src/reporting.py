"""阶段七：从真实输出生成报告、图表和简历描述。"""
import json

import os
from pathlib import Path

_plot_cache = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
_plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_plot_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .config import ROOT, OUT, FIG, ARMS, ensure_directories

COLORS = {"No E-Mail": "#94a3b8", "Mens E-Mail": "#2563eb", "Womens E-Mail": "#16a34a"}
MODEL_COLORS = {
    "Response prediction": "#64748b", "S-Learner": "#2563eb",
    "T-Learner": "#16a34a", "X-Learner": "#9333ea",
}
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 160, "font.size": 10,
    "axes.grid": True, "grid.alpha": .2,
    "axes.spines.top": False, "axes.spines.right": False, "text.parse_math": False,
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)

def tidy_markdown(text):
    """标题前后留空行，代码块原样保留，统一 Markdown 阅读间距。"""
    result = []
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("#"):
            if result and result[-1].strip():
                result.append("")
            result.append(line)
            result.append("")
        else:
            if not line.strip() and result and not result[-1].strip():
                continue
            result.append(line)
    return "\n".join(result).rstrip() + "\n"

def markdown_table(frame):
    """不依赖 tabulate 的简洁 Markdown 表格。"""
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator] + rows)

def figure_groups(groups):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(
        axes, ["conversion", "visit", "spend"], ["Purchase rate", "Visit rate", "Mean spend per customer"]
    ):
        data = groups[groups.metric == metric].set_index("arm").loc[ARMS]
        scale = 1 if metric == "spend" else 100
        values = data["mean"].to_numpy() * scale
        errors = np.vstack([
            (data["mean"] - data.ci_low).to_numpy(),
            (data.ci_high - data["mean"]).to_numpy(),
        ]) * scale
        ax.bar(np.arange(3), values, yerr=errors, capsize=4, color=[COLORS[a] for a in ARMS], width=.6)
        for index, value in enumerate(values):
            ax.text(index, data.ci_high.iloc[index] * scale * 1.025, f"{value:.2f}", ha="center")
        ax.set_xticks(np.arange(3), ["Control", "Mens", "Womens"])
        ax.set_ylabel("USD" if metric == "spend" else "%")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(data.ci_high) * scale * 1.2)
    fig.suptitle("Randomized experiment | all assigned customers | individual 95% CI", fontsize=11)
    save(fig, "01-group-metrics.png")

def figure_balance(balance):
    data = balance.copy()
    data["label"] = data.feature + data.level.fillna("").map(lambda value: f": {value}" if value else "")
    pairs = data.groupby(["treatment", "control"], sort=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    for number, ((treated, control), group) in enumerate(pairs):
        ax.scatter(group.smd, group.label, s=25, label=f"{treated} - {control}", alpha=.75)
    ax.axvline(0, color="#334155", linewidth=.8)
    ax.axvline(.1, color="#dc2626", linestyle="--", linewidth=1)
    ax.axvline(-.1, color="#dc2626", linestyle="--", linewidth=1)
    ax.set_xlim(-.11, .11)
    ax.set_xlabel("Standardized mean difference")
    ax.set_title("Pre-treatment balance | reference threshold: ±0.1", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "02-balance.png")

def figure_ate(ab):
    data = ab[ab.control == "No E-Mail"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, ["conversion", "visit", "spend"], ["Purchase ATE", "Visit ATE", "Spend ATE"]):
        group = data[data.metric == metric]
        scale = 1 if metric == "spend" else 100
        point = group.effect.to_numpy() * scale
        error = np.vstack([
            (group.effect - group.ci_low).to_numpy(),
            (group.ci_high - group.effect).to_numpy(),
        ]) * scale
        ax.errorbar(point, np.arange(len(group)), xerr=error, fmt="o", capsize=4, color="#2563eb")
        ax.set_yticks(np.arange(len(group)), group.treatment)
        ax.axvline(0, color="#64748b", linestyle="--")
        ax.set_xlabel("USD per assigned customer" if metric == "spend" else "Percentage points")
        ax.set_title(title, fontweight="bold")
    fig.suptitle("Unadjusted ITT effects versus No E-Mail | individual 95% CI", fontsize=11)
    save(fig, "03-ate.png")

def figure_heterogeneity(subgroups):
    data = subgroups[(subgroups.treatment == "Any E-Mail") & (subgroups.metric == "conversion")]
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = data.feature + " = " + data.level.astype(str)
    errors = np.vstack([
        (data.effect - data.ci_low).to_numpy(), (data.ci_high - data.effect).to_numpy()
    ]) * 100
    ax.errorbar(data.effect * 100, np.arange(len(data)), xerr=errors, fmt="o", color="#2563eb", capsize=3)
    ax.set_yticks(np.arange(len(data)), labels)
    ax.axvline(0, linestyle="--", color="#64748b")
    ax.set_xlabel("Purchase probability change (percentage points)")
    ax.set_title("Exploratory subgroup effects | any email versus control", fontweight="bold")
    ax.text(0, -.14, "Overlapping groups; individual 95% CI. Effect differences require interaction tests.",
            transform=ax.transAxes, fontsize=9)
    save(fig, "04-heterogeneous-effects.png")

def figure_qini(curves):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, outcome, title in zip(axes, ["conversion", "visit"], ["Purchase uplift", "Visit uplift"]):
        data = curves[curves.outcome == outcome]
        for model, group in data.groupby("model", sort=False):
            ax.plot(group.fraction, group.incremental_conversions, label=model,
                    color=MODEL_COLORS[model], linewidth=1.8)
        group = data[data.model == "S-Learner"]
        ax.plot([0, 1], [0, group.incremental_conversions.iloc[-1]], "--", color="#94a3b8", label="Random expected")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Share of customers targeted")
        ax.set_ylabel("Estimated incremental events")
        ax.set_title(title + " | held-out test set", fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle("Qini-style IPW gain curves | endpoint = N × ATE | no test-set model selection", fontsize=10)
    save(fig, "05-qini.png")

def figure_deciles(deciles, lock):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, outcome in zip(axes, ["conversion", "visit"]):
        data = deciles[(deciles.outcome == outcome) & (deciles.model == lock["selected_model"])]
        error = np.vstack([
            (data.observed_uplift - data.ci_low).to_numpy(),
            (data.ci_high - data.observed_uplift).to_numpy(),
        ]) * 100
        ax.errorbar(data.decile, data.observed_uplift * 100, yerr=error, fmt="o-", capsize=3, color="#2563eb")
        ax.axhline(0, color="#94a3b8", linestyle="--")
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Predicted uplift decile (1 = highest)")
        ax.set_ylabel("Observed uplift (percentage points)")
        ax.set_title(outcome + " | " + lock["selected_model"], fontweight="bold")
    fig.suptitle("Held-out decile diagnostics | individual 95% CI; small purchase counts", fontsize=10)
    save(fig, "06-deciles.png")

def figure_profit(simulation, lock):
    fig, ax = plt.subplots(figsize=(10, 5))
    for strategy in [lock["selected_model"], "Response prediction", "Random expected"]:
        data = simulation[simulation.strategy == strategy]
        color = MODEL_COLORS.get(strategy, "#94a3b8")
        ax.plot(data.fraction, data.net_profit_usd, label=strategy, color=color, linewidth=2)
        if strategy == lock["selected_model"]:
            ax.fill_between(data.fraction, data.net_profit_ci_low, data.net_profit_ci_high, color=color, alpha=.12,
                            label="Selected model: pointwise 95% CI")
    ax.axhline(0, color="#64748b", linewidth=.8)
    ax.axvline(lock["selected_fraction"], linestyle="--", color="#334155", label="Validation-selected coverage")
    ax.axvline(.3, linestyle=":", color="#9333ea", label="30% budget example")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Share of candidate customers receiving an email")
    ax.set_ylabel("Estimated net profit per 10,000 candidates (USD)")
    ax.set_title("Offline profit scenarios | assumed cost $0.02, contribution $30", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "07-policy-profit.png")

def figure_sensitivity(sensitivity):
    fractions = sensitivity.pivot(index="email_cost_usd", columns="margin_usd", values="fraction_selected_on_validation")
    profits = sensitivity.pivot(index="email_cost_usd", columns="margin_usd", values="test_profit_per_10000")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, title, coverage in [
        (axes[0], fractions, "Coverage selected on validation", True),
        (axes[1], profits, "Test profit per 10,000 candidates", False),
    ]:
        values = data.to_numpy()
        image = ax.imshow(values, cmap="Blues" if coverage else "RdBu", aspect="auto")
        ax.set_xticks(range(len(data.columns)), [f"${x:.0f}" for x in data.columns])
        ax.set_yticks(range(len(data.index)), [f"${x:.3f}" for x in data.index])
        ax.set_xlabel("Assumed contribution per incremental purchase")
        ax.set_ylabel("Assumed cost per email")
        ax.set_title(title, fontsize=11, fontweight="bold")
        for row in range(len(data.index)):
            for col in range(len(data.columns)):
                label = f"{values[row, col]:.0%}" if coverage else f"${values[row, col]:,.0f}"
                ax.text(col, row, label, ha="center", va="center", color="#111827", fontsize=10,
                        bbox={"facecolor":"white", "alpha":.7, "edgecolor":"none", "pad":2})
        fig.colorbar(image, ax=ax, shrink=.8)
    save(fig, "08-sensitivity.png")

def create_report():
    ensure_directories()
    groups = pd.read_csv(OUT / "group_metrics.csv")
    ab = pd.read_csv(OUT / "ab_results.csv")
    adjusted = pd.read_csv(OUT / "ate_adjusted.csv")
    subgroups = pd.read_csv(OUT / "heterogeneous_effects.csv")
    heterogeneity = pd.read_csv(OUT / "heterogeneity_tests.csv")
    metrics = pd.read_csv(OUT / "uplift_metrics.csv")
    curves = pd.read_csv(OUT / "uplift_curves.csv")
    deciles = pd.read_csv(OUT / "uplift_deciles.csv")
    simulation = pd.read_csv(OUT / "business_simulation.csv")
    fixed = pd.read_csv(OUT / "locked_policy_test.csv")
    budget = pd.read_csv(OUT / "budget_locked_policies.csv")
    budget_contrasts = pd.read_csv(OUT / "budget_policy_contrasts.csv")
    lock = json.loads((OUT / "policy_lock.json").read_text())
    validation = json.loads((OUT / "experiment_validation.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())

    from .diagnostics import standard_qini
    standard_qini()
    figure_groups(groups)
    figure_balance(pd.read_csv(OUT / "balance_smd.csv"))
    figure_ate(ab)
    figure_heterogeneity(subgroups)
    figure_qini(curves)
    figure_deciles(deciles, lock)
    figure_profit(simulation, lock)
    figure_sensitivity(pd.read_csv(OUT / "sensitivity_analysis.csv"))

    group_rows = []
    for arm in ARMS:
        group = groups[groups.arm == arm].set_index("metric")
        group_rows.append({
            "实验组": arm, "客户数": f"{int(group.n.iloc[0]):,}",
            "访问率": f"{group.loc['visit','mean']:.2%}",
            "购买率": f"{group.loc['conversion','mean']:.3%}",
            "人均消费（美元）": f"{group.loc['spend','mean']:.3f}",
        })
    effect_rows = []
    for row in ab[ab.control == "No E-Mail"].itertuples():
        scale = 1 if row.metric == "spend" else 100
        unit = "美元" if row.metric == "spend" else "个百分点"
        effect_rows.append({
            "对比": row.treatment + " − No E-Mail", "指标": row.metric,
            "绝对提升": f"{row.effect * scale:.3f} {unit}",
            "相对提升": f"{row.relative_lift:.1%}",
            "95% CI": f"[{row.ci_low * scale:.3f}, {row.ci_high * scale:.3f}]",
            "Holm p": f"{row.p_holm:.3g}",
        })
    model_rows = []
    for row in metrics[(metrics.outcome == "conversion") & (metrics.split == "test")].itertuples():
        model_rows.append({
            "模型": row.model,
            "原始 AUUC": f"{row.auuc_raw_per_customer:.6f}",
            "AUUC 超过随机部分": f"{row.auuc_above_random_per_customer:.6f}",
            "标准化 Qini": f"{row.qini_normalized_sklift:.4f}",
            "Uplift@30%": f"{row.uplift_at_30 * 100:.3f} 个百分点",
        })
    cap30 = budget[np.isclose(budget.budget_cap, .3)]
    cap30_contrast = budget_contrasts[
        np.isclose(budget_contrasts.budget_cap, .3) & (budget_contrasts.comparator == "Random expected")
    ].iloc[0]
    budget_rows = []
    for row in cap30.itertuples():
        budget_rows.append({
            "策略": row.strategy, "覆盖比例": f"{row.selected_fraction:.0%}",
            "增量购买/万人": f"{row.incremental_conversions_per_10000:.2f}",
            "邮件成本": f"${row.email_cost_per_10000:.2f}",
            "假设净收益": f"${row.net_profit_per_10000:.2f}",
            "净收益 95% CI": f"[${row.profit_ci_low:.2f}, ${row.profit_ci_high:.2f}]",
        })
    pooled = ab[(ab.treatment == "Any E-Mail") & (ab.metric == "conversion")].iloc[0]
    full = fixed[fixed.policy == "All send"].iloc[0]
    significant_interactions = heterogeneity[
        (heterogeneity.metric == "conversion") & (heterogeneity.q_bh_exploratory < .05)
    ]
    hetero_text = (
        f"购买率交互检验中有 {len(significant_interactions)} 个结果通过全体交互检验的 BH 5% 修正，仍属于探索性证据。"
        if len(significant_interactions)
        else "购买率的分群交互检验经 BH 修正后没有达到 5% 水平；不能仅凭分群点估计不同就声称发现了稳定异质性。"
    )
    uncertainty_text = (
        "30% 预算场景中，增益投放相对随机投放的收益差区间跨越零，不能证明个性化策略优于随机。"
        if cap30_contrast.ci_low <= 0 <= cap30_contrast.ci_high
        else "30% 预算场景的收益差区间不跨零，方向和大小见下表；这是固定策略的离线估计，仍需要新的策略随机实验。"
    )
    readme = f"""# 邮件营销 A/B 测试与因果增益分析
## Email Campaign A/B Testing and Uplift Modeling

使用 Hillstrom 随机邮件实验，判断营销干预是否增加访问、购买和消费，并在有限预算下评价客户投放排序。项目补充 MySQL 电商购物漏斗作品集中的 Python、统计推断、因果分析和机器学习验证能力。

## 项目目标

1. 邮件是否真正带来增量，而不是把本来就会发生的购买归因于邮件？
2. 男装和女装邮件是否存在效果差异？
3. 实验前客户特征是否影响营销效果？
4. 在固定预算下，增益排序是否优于随机或购买概率排序？

## 数据概况

{markdown_table(pd.DataFrame(group_rows))}

共 **{audit['rows']:,} 客户、12 字段、{int(groups.loc[groups.metric == 'conversion', 'total'].sum()):,} 次购买**；无缺失或结果字段逻辑异常。结果观察窗口是实验后两周。{audit['identical_rows']:,} 行完全相同，但没有客户 ID，不能认定为重复客户；保留原始实验样本。消费金额的 {audit['spend_zero_share']:.2%} 为零，均值分析仍包含全部客户。

来源为 [Hillstrom 原始挑战](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)，下载脚本使用 [scikit-uplift 官方数据接口所列镜像](https://www.uplift-modeling.com/en/latest/_modules/sklift/datasets/datasets.html)。本次因运行环境限制，数据取自用户已有压缩包中的原始 CSV；只提取数据，未复制其他项目代码。实际取得方式和校验值记录于 reports/tables/data_provenance.json。没有声称本次成功联网校验镜像压缩文件。

## 分析口径

- 客户记录为分析单位；主要指标 conversion，辅助指标 visit、含零值的人均 spend。
- 处理定义为分配到邮件组，估计 ITT；未观测实际打开或阅读邮件。
- Mens、Womens 分别对 No E-Mail；合并邮件指男装/女装随机混合策略，不是个性化内容选择。
- 两比例 z 检验 / Newcombe 区间，spend 用 Welch + 2,000 次 bootstrap；12 个固定比较统一 Holm 修正。
- 1:1:1 为计划分配假设，SRM p={validation['srm_p']:.3f}；最大绝对 SMD={validation['max_abs_smd']:.3f}。分配日志缺失，检查通过不能证明来源实验一定没有设计问题。
- 仅使用 7 个实验前特征；训练/验证/测试=38,400/12,800/12,800。模型与覆盖比例只按验证集选择。

详细方法、假设和指标单位见 [分析方案](docs/01-analysis-plan.md)、[字段说明](docs/02-data-dictionary.md)。

## 核心结果：平均效应

{markdown_table(pd.DataFrame(effect_rows))}

合并邮件相对对照的购买率增加 **{pooled.effect * 100:.3f} 个百分点**，95% CI 为 [{pooled.ci_low * 100:.3f}, {pooled.ci_high * 100:.3f}] 个百分点，相对提升 {pooled.relative_lift:.1%}。绝对变化和相对变化单位不同，不能混淆。

男装邮件的购买率、访问率和人均消费高于女装邮件；三项组间比较在本项目固定的 12 个检验家族中通过 Holm 5% 修正。普通 95% 区间为逐项区间，Bonferroni 同时区间另存于 ab_results.csv。

![实验组指标](reports/figures/01-group-metrics.png)
![平均处理效应](reports/figures/03-ate.png)

Lin 完全交互回归的调整效应与均值差接近，见 reports/tables/ate_adjusted.csv。{hetero_text}

![实验前平衡](reports/figures/02-balance.png)
![分群效应](reports/figures/04-heterogeneous-effects.png)

## 增益模型：预测谁买与预测谁被邮件影响

响应预测基线使用 logistic；S/T-Learner 使用固定参数的浅层梯度提升树；X-Learner 使用三折交叉拟合伪效应与两组回归器。

下表为购买结果的独立测试集评价。原始 AUUC 与 scikit-uplift 标准化 Qini 的定义不同，不能直接比较数值大小。

{markdown_table(pd.DataFrame(model_rows))}

验证集在三个增益模型中选中了 **{lock['selected_model']}**。T-Learner 的测试指标更高属于测试诊断结果，不允许据此改选模型后再声称得到了无偏策略收益。S-Learner 测试排序优势较弱，项目不夸大增益模型有效性。响应预测的 ROC/PR AUC 作为分类诊断保存，不作为增益模型选择标准。

![标准Qini曲线](reports/figures/09-qini-standard.png)
![总体IPW增益曲线](reports/figures/05-qini.png)
![分位数组](reports/figures/06-deciles.png)

## 业务模拟与预算选择

假设每封邮件成本 **0.02 美元**，每次增量购买贡献利润 **30 美元**，候选客户 **10,000 人**。这些是教学场景，不是真实测得的客单价、营销成本或利润。

没有硬预算上限时，验证集选择 **{lock['selected_fraction']:.0%} 覆盖**；本次是全量发送。测试集估计全量发送增加 {full.incremental_conversions_per_10000:.2f} 次购买/万人，假设净收益为 \\$ {full.net_profit_per_10000:,.2f}，组内 bootstrap 95% CI [\\$ {full.bootstrap_profit_ci_low:,.2f}, \\$ {full.bootstrap_profit_ci_high:,.2f}]。低成本场景没有必要强行缩减到“个性化最佳人群”。

明确 30% 预算上限时，比例仍只按验证集选择：

{markdown_table(pd.DataFrame(budget_rows))}

增益策略相对同预算随机期望策略的净收益差为 \\$ {cap30_contrast.profit_difference_per_10000:.2f}/万人，95% CI [\\$ {cap30_contrast.ci_low:.2f}, \\$ {cap30_contrast.ci_high:.2f}]。**{uncertainty_text}**

同时保存 10%、50% 硬预算场景。全量发送对应 100% 预算，不能在 30% 预算下作为可执行的同预算选项。不同覆盖比例曲线仅作描述，未用测试峰值重新选择比例。

![策略利润](reports/figures/07-policy-profit.png)
![成本与购买价值敏感性](reports/figures/08-sensitivity.png)

成本 0.005/0.02/0.10/0.50 美元 × 贡献利润 10/30/60 美元的敏感性场景说明：成本足够高时，验证集会选择完全不发送。验证集选择也可能不能在测试集稳定复现收益，负点估计原样保留。

## 项目导航

目录与阅读顺序见 [项目首页](../README.md)。

## 运行方法

Windows PowerShell：

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt
.\\.venv\\Scripts\\python.exe run.py --stage all
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe src/tools/run_notebooks.py
```

不需要激活环境或更改系统执行策略。跨系统环境可使用 requirements.txt；本次实际依赖全量锁定在 requirements-lock.txt。详情见 [逐步执行指南](docs/03-execution-guide.md)。

## 项目成果

独立源码与随机实验审计、A/B/ATE 结果、探索性分群分析、响应预测基线、S/T/X-Learner、Qini/AUUC/Uplift@K 与分位数组、验证集锁定策略、三种硬预算场景、12 种成本价值场景、9 张图和编号教学 Notebook。方法测试覆盖不等组分配下的增量估计、分母定义、特征防泄漏、样本不重叠、排序方向及真实数据契约。

[项目总结](docs/project-summary.md) · [面试讲解](docs/04-interview-guide.md) · [中文简历四条](docs/resume-bullets.md)

## 局限性

- 历史公开实验，没有分配日志、客户 ID 或长期追踪，不能验证干扰、实际打开邮件、退订和长期客户价值。
- 购买事件仅 578 次，测试集更少，客户级异质性和策略差值区间较宽。机器学习不保证优于简单投放。
- 分群是探索分析；普通逐项区间不是同时区间。响应预测的关联能力不等于因果效应。
- bootstrap 条件于固定已训练策略，不包含重新训练和模型选择的不确定性。
- 预算、成本与贡献利润是场景假设；收益是离线估计，不是线上已实现利润，也不是直接用模型分数求和。
- 本模型只判断给谁发混合邮件；未建立选择男装/女装内容的多处理投放模型。
- 原始数据的公开可访问性不自动等于明确的再分发许可。GitHub 默认不提交原始数据；代码不把数据的许可问题覆盖为项目代码许可。

## 参考方法

[Meta-learners 原始论文](https://arxiv.org/abs/1706.03461)；
[scikit-uplift Qini 定义](https://www.uplift-modeling.com/en/latest/api/metrics/qini_auc_score.html)；
[scikit-uplift 标准化 uplift AUC](https://www.uplift-modeling.com/en/latest/api/metrics/uplift_auc_score.html)；
[statsmodels 两比例差区间](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.confint_proportions_2indep.html)。

源码及分析独立编写，官方文档用于核对接口、数据字段及指标定义。
"""
    detailed = readme.replace("](reports/figures/", "](figures/")
    detailed = detailed.replace("](docs/", "](../docs/")
    detailed = detailed.replace(
        "## Email Campaign A/B Testing and Uplift Modeling",
        "Email Campaign A/B Testing and Uplift Modeling",
    )
    detailed = detailed.replace(")\n![", ")\n\n![")
    (ROOT / "reports/analysis.md").write_text(tidy_markdown(detailed), encoding="utf-8")

    home = f"""# 邮件营销 A/B 测试与因果增益分析

Email Campaign A/B Testing and Uplift Modeling

使用 Hillstrom 随机邮件实验，回答“邮件是否带来增量购买”和“预算有限时应该给谁发送”。项目展示 Python、统计检验、因果推断和增益模型能力。

**第一次看这个项目：** 先读本页，再按编号阅读 notebooks。完整分析放在 reports，方法与操作说明放在 docs。

## 01 项目目标

- 判断邮件对访问率、购买率和人均消费的平均影响。
- 比较男装邮件、女装邮件与不发邮件的差异。
- 检验客户特征是否对应不同营销效果。
- 比较全量、随机、响应预测和增益模型投放的离线收益。

## 02 数据与分析口径

| 项目 | 说明 |
| --- | --- |
| 数据规模 | {audit['rows']:,} 名客户，12 个字段，{int(groups.loc[groups.metric == 'conversion', 'total'].sum()):,} 次购买 |
| 实验分组 | Mens E-Mail / Womens E-Mail / No E-Mail |
| 主要指标 | conversion：全部分配客户中的购买率 |
| 辅助指标 | visit：访问率；spend：包含零值的人均消费 |
| 模型特征 | 7 个实验前变量；不使用实验后的访问、购买和金额 |
| 数据划分 | 训练 60% / 验证 20% / 测试 20% |

无缺失；相同记录因没有客户 ID 而保留。SRM p={validation['srm_p']:.3f}，最大绝对 SMD={validation['max_abs_smd']:.3f}。
字段与来源见 [数据字典](docs/02-data-dictionary.md)，统计方法与假设见 [分析方案](docs/01-analysis-plan.md)。

## 03 核心结果

| 问题 | 本次代码运行得到的结论 |
| --- | --- |
| 邮件是否增加购买 | 混合邮件策略提高购买率 **{pooled.effect * 100:.3f} 个百分点**，95% CI [{pooled.ci_low * 100:.3f}, {pooled.ci_high * 100:.3f}] |
| 两种邮件是否相同 | 男装邮件的购买率、访问率、人均消费均高于女装邮件，固定比较通过 Holm 修正 |
| 选中了什么模型 | 验证集在 S/T/X 中选择 **{lock['selected_model']}**；测试集只做评价 |
| 默认成本下怎么投放 | 验证集选择 **{lock['selected_fraction']:.0%} 覆盖** |
| 个性化是否更好 | 30% 预算下，相对随机投放的收益差区间跨零，尚未证明更优 |

“提高 {pooled.effect * 100:.3f} 个百分点”是绝对概率差。收益模拟采用每封邮件 0.02 美元、每次增量购买贡献利润 30 美元的场景假设，并非线上实绩。

![三组实验指标与95%区间](reports/figures/01-group-metrics.png)

[完整分析报告](reports/analysis.md) · [标准 Qini 图](reports/figures/09-qini-standard.png) · [策略收益图](reports/figures/07-policy-profit.png)

## 04 按顺序阅读

| 顺序 | Notebook | 对应源码 |
| --- | --- | --- |
| 01 | [数据审计](notebooks/01-data-audit.ipynb) | src/data.py |
| 02 | [实验验证](notebooks/02-experiment-validation.ipynb) | src/statistics.py |
| 03 | [A/B 测试](notebooks/03-ab-test-analysis.ipynb) | src/statistics.py |
| 04 | [因果效应](notebooks/04-causal-effect-analysis.ipynb) | src/causal.py |
| 05 | [增益模型与预算](notebooks/05-uplift-modeling.ipynb) | src/modeling.py、src/evaluation.py、src/budget.py |

Notebook 按“问题 → 方法与假设 → 代码 → 结果”展开，计算统一调用 src 中的实现。
详细操作见 [执行指南](docs/03-execution-guide.md)。

## 05 文件放在哪里

```text
email-campaign-causal-analysis/
├── data/raw/          原始数据
├── notebooks/         按 01–05 顺序学习
├── src/               数据、统计、因果、模型与报告代码
│   └── tools/         导入、Notebook 执行和打包工具
├── reports/
│   ├── analysis.md    完整分析报告
│   ├── figures/       9 张结果图
│   ├── tables/        全部结果 CSV / JSON
│   └── logs/          运行与验证记录
├── docs/              方法、字段、运行、面试和简历说明
├── tests/             关键统计和建模口径检查
├── run.py             统一分析运行入口
├── requirements.txt
├── requirements-lock.txt
└── README.md          项目阅读入口
```

所有结果表统一保存一份。环境和缓存属于工具文件，不参与阅读。GitHub 发布规则见 [执行指南](docs/03-execution-guide.md)。

## 06 复现项目

在项目根目录执行：

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt
.\\.venv\\Scripts\\python.exe run.py --stage all
.\\.venv\\Scripts\\python.exe -m pytest -q
```

GitHub 仓库不包含原始 CSV。首次运行会自动下载并验证数据；已有通过校验的本地缓存时可离线读取。运行后，来源与获取方式记录在 reports/tables/data_provenance.json。
Notebook 可在 Jupyter 中打开，也可使用 `python src/tools/run_notebooks.py` 逐单元执行。

## 07 展示与面试

[项目总结](docs/project-summary.md) · [面试讲解](docs/04-interview-guide.md) · [中文简历四条](docs/resume-bullets.md)

局限性：历史实验只有 578 次购买，缺少客户 ID、分配日志和长期指标；分群与策略收益存在较大不确定性。模型只决定给谁发送混合邮件，没有实现男装/女装内容的个性化选择。完整假设、局限性与参考方法见 [分析报告](reports/analysis.md)。
"""
    (ROOT / "README.md").write_text(tidy_markdown(home), encoding="utf-8")

    summary = f"""# 项目总结
## 一、业务背景
随机营销实验用于区分邮件带来的增量和本来就会发生的购买，进一步检验预算有限时的客户排序。

## 二、实验与数据
{audit['rows']:,} 条客户记录，三组随机分配，结果观察两周；保留相同记录和消费零值。SRM p={validation['srm_p']:.3f}，最大 |SMD|={validation['max_abs_smd']:.3f}，同时说明来源随机化和日志缺失。

## 三、平均效果
{markdown_table(pd.DataFrame(effect_rows))}
总体邮件增加购买概率；调整 ATE 与均值差接近。{hetero_text}

## 四、模型验证
60/20/20 分组分层划分，预处理只拟合训练集。验证集选 {lock['selected_model']}，测试集仅评价。T 的测试指标更好不代表可以再次用测试集选模型。

## 五、预算与业务结论
低成本默认场景验证集选择 {lock['selected_fraction']:.0%} 覆盖。
{markdown_table(pd.DataFrame(budget_rows))}
{uncertainty_text} 不将正净收益误说成相对随机策略的显著改善。

## 六、后续实验
若要上线，在新客户批次中固定预算，随机分配“随机投放”和“预先固定的模型投放”策略，比较候选客户级增量贡献利润和退订率；样本量应按策略差的 MDE 规划，而非仅按邮件 vs 不发邮件的 ATE 规划。

## 七、局限性
数据较旧、购买稀少、没有客户 ID/长期指标、混合内容策略不能直接转为个性化内容。假设利润与成本不是实际观测。全部数值源于本项目代码运行。
"""
    (ROOT / "docs/project-summary.md").write_text(tidy_markdown(summary), encoding="utf-8")
    resume = f"""# 中文简历项目描述（四条）

项目名称：邮件营销 A/B 测试与因果增益分析 | Python、pandas、SciPy、statsmodels、scikit-learn、scikit-uplift

- 基于 Hillstrom 64,000 名客户随机邮件实验，建立可复现的数据审计和实验验证流程，完成 SRM、实验前特征平衡及防泄漏检查，最大绝对标准化差异为 {validation['max_abs_smd']:.3f}。
- 以购买率为主要指标，结合两比例检验、Newcombe 95% 区间、Welch/Bootstrap 和 Holm 多重修正估计 ITT/ATE；邮件混合策略较对照购买率提升 {pooled.effect * 100:.3f} 个百分点，并用协变量调整和分群交互检验验证结果。
- 使用实验前特征实现响应预测基线及 S/T/X-Learner，按 60%/20%/20% 划分训练、验证和测试集，以 Qini、AUUC、Uplift@K 和分位数组评价增益排序，避免以分类准确率或测试集峰值选择策略。
- 构建全量、随机、响应预测和增益投放的离线收益模拟，覆盖 3 种硬预算与 12 种成本价值场景，并用成对 Bootstrap 检验策略收益差；明确默认场景选择全量发送及个性化优势尚未得到稳定支持的业务边界。

使用提醒：这里展示的是离线项目成果，不是上线营销业绩。不要改写成“实际提高销售额”或“实现利润增长”。
"""
    (ROOT / "docs/resume-bullets.md").write_text(tidy_markdown(resume), encoding="utf-8")
    print("REPORT: nine figures, README, summary and four resume bullets generated.")
