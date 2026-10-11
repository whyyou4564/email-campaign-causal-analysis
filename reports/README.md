# 结果阅读指南

[完整分析报告](analysis.md) 按实验审计、平均效应、异质性、增益模型、业务模拟展开。先读报告，再按需要查图和表。

## 01 结果图

| 分析阶段 | 图表 |
| --- | --- |
| 实验分组 | [01 三组指标](figures/01-group-metrics.png) |
| 随机化检查 | [02 特征平衡](figures/02-balance.png) |
| 平均效果 | [03 ATE 与区间](figures/03-ate.png) |
| 客户分群 | [04 异质性效应](figures/04-heterogeneous-effects.png) |
| 增益排序 | [05 总体 IPW 增益曲线](figures/05-qini.png) |
| 分位数组 | [06 分位数组评价](figures/06-deciles.png) |
| 业务模拟 | [07 策略收益](figures/07-policy-profit.png) |
| 场景变化 | [08 成本与价值敏感性](figures/08-sensitivity.png) |
| 标准评价 | [09 标准 Qini 曲线](figures/09-qini-standard.png) |

图 05 的总体增量人数和图 09 的处理组尺度 Qini gain 单位不同，定义见 docs/01-analysis-plan.md。

## 02 关键结果表

| 要查的结果 | 文件 |
| --- | --- |
| 组间差值、相对提升、区间及多重修正 | [ab_results.csv](tables/ab_results.csv) |
| 协变量调整 ATE | [ate_adjusted.csv](tables/ate_adjusted.csv) |
| 各模型验证集与测试集指标 | [uplift_metrics.csv](tables/uplift_metrics.csv) |
| 固定策略测试收益 | [locked_policy_test.csv](tables/locked_policy_test.csv) |
| 固定策略的成对差值 | [policy_contrasts.csv](tables/policy_contrasts.csv) |
| 10%、30%、50% 预算场景 | [budget_locked_policies.csv](tables/budget_locked_policies.csv) |
| 成本与贡献利润变化 | [sensitivity_analysis.csv](tables/sensitivity_analysis.csv) |
| 未来实验样本量规划 | [power_sample_size.csv](tables/power_sample_size.csv) |

tables 中也保存审计、预测和策略锁定文件，按文件名查找即可；逐行数据不提交到 GitHub。
logs 保存真实运行及验证记录，delivery-manifest.json 保存交付文件校验值。
