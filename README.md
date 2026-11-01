# 邮件营销 A/B 测试与因果增益分析

本项目使用 Hillstrom 邮件营销随机实验数据，在 Python 中完成 A/B 测试、平均因果效应估计和增益建模，并通过独立测试集评价预算有限时的客户投放策略。

## 项目目标

- 发送营销邮件是否真正提高访问率、购买率和人均消费？
- 男装邮件与女装邮件的效果是否不同？
- 客户特征是否对应不同的营销增量？
- 预算有限时，应该优先给哪些客户发送邮件？
- 个性化投放是否优于全量发送或随机投放？

## 数据概况

| 指标 | 数量或说明 |
| --- | --- |
| 客户记录 | 64,000 |
| 数据字段 | 12 |
| 购买次数 | 578 |
| 实验分组 | Mens E-Mail / Womens E-Mail / No E-Mail |
| 结果观察期 | 实验后两周 |
| 模型数据划分 | 训练 60% / 验证 20% / 测试 20% |

原始数据来自公开的 Hillstrom 邮件营销实验。无缺失；由于没有客户 ID，完全相同的记录不能判定为重复客户，因此保留原始实验样本。

## 分析口径

1. 主要指标为购买率 conversion，分母包含全部分配客户；辅助指标为访问率 visit 与含零值的人均消费 spend。
2. 以 No E-Mail 为对照，分别估计男装、女装和混合邮件策略的 ITT / ATE。
3. 使用两比例检验、Newcombe 95% 区间、Welch / Bootstrap 和 Holm 多重检验修正，并检查 SRM、实验前平衡与未来样本量。
4. 模型仅使用 7 个实验前变量，禁止使用 visit、conversion、spend 等实验后结果作特征。
5. 验证集选择模型与投放比例，测试集仅评价；使用 Qini、AUUC、Uplift@K 和分位数组评价增益排序。

## 核心结果

| 问题 | 本次运行结果 |
| --- | --- |
| 邮件对购买的平均影响 | 购买率提高 **0.495 个百分点**，95% CI [0.351, 0.633] |
| 两种邮件是否相同 | 男装邮件的购买率、访问率与人均消费高于女装邮件，固定比较通过 Holm 修正 |
| 随机化诊断 | SRM p=0.904，最大绝对 SMD=0.017 |
| 验证集选择的模型 | S-Learner |
| 默认成本下的覆盖比例 | 100% |
| 个性化是否更优 | 30% 预算下，相对随机投放的收益差区间跨零，尚未证明更优 |

主要发现：

- 邮件对购买产生正向平均增量，但绝对百分点与相对百分比需要区分。
- 低成本默认场景下，验证集选择全量发送，不能预设个性化投放一定更好。
- 预测“谁会购买”和预测“谁会因邮件而增量购买”是不同任务。
- 收益模拟假设每封邮件成本 0.02 美元、每次增量购买贡献利润 30 美元；属于离线场景估计。

完整数值与图表见 [分析报告](4%20outputs/analysis.md)。

## 项目结构

```text
email-campaign-causal-analysis/
├── 1 data/
│   └── raw/                 原始 CSV（不上传 GitHub）
├── 2 python/
│   ├── notebooks/           01–05 教学分析
│   ├── src/                 数据、统计、因果和模型代码
│   │   └── tools/          数据导入、Notebook 执行和打包
│   └── tests/              关键口径检查
├── 3 docs/                 方法、字段、运行、面试与简历说明
├── 4 outputs/
│   ├── analysis.md         完整分析报告
│   ├── figures/            9 张结果图
│   ├── tables/             精选结果表及本地生成结果
│   └── logs/               本地运行记录（不上传 GitHub）
├── run.py                  统一运行入口
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

## 使用工具

- Python、pandas、NumPy：数据整理与审计
- SciPy、statsmodels：统计检验、置信区间与因果效应估计
- scikit-learn、scikit-uplift：响应预测、S/T/X-Learner 和增益评价
- Matplotlib、Seaborn：结果图表
- pytest：关键统计和数据泄漏检查

## 运行方法

详细步骤见 [Python 运行说明](3%20docs/03-execution-guide.md)。在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe run.py --stage all
.\.venv\Scripts\python.exe -m pytest -q
```

首次运行会自动下载并校验数据；已有通过校验的本地缓存时可离线读取。原始 CSV、环境、缓存和逐行预测不上传 GitHub。

按顺序阅读 [01 数据检查](2%20python/notebooks/01-data-audit.ipynb)、[02 实验验证](2%20python/notebooks/02-experiment-validation.ipynb)、[03 A/B 测试](2%20python/notebooks/03-ab-test-analysis.ipynb)、[04 因果效应](2%20python/notebooks/04-causal-effect-analysis.ipynb)、[05 增益模型](2%20python/notebooks/05-uplift-modeling.ipynb)。

## 项目成果

- 5 个有真实运行输出的教学 Notebook
- A/B 测试、ATE、异质性与增益建模的可复现 Python 实现
- 9 张结果图和精选聚合结果表
- 3 种硬预算与 12 种成本价值场景的离线模拟
- [项目总结](3%20docs/project-summary.md)、[面试讲解](3%20docs/04-interview-guide.md)和[中文简历四条](3%20docs/resume-bullets.md)

## 局限性

- 历史实验只有 578 次购买，细分群体和策略收益存在较大不确定性。
- 缺少客户 ID、分配日志、长期结果和退订指标。
- 随机化诊断通过不能单独证明随机分配成立；因果解释依赖实验设计及相关假设。
- 模型决定给谁发送混合邮件，没有建立男装与女装内容的个性化选择策略。
- 贡献利润与邮件成本为假设，不能将离线收益写成线上业绩。

## 数据来源

[Hillstrom 原始实验说明](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) · [scikit-uplift 数据接口](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html)

字段与数据来源详见 [数据字典](3%20docs/02-data-dictionary.md)，方法和假设详见 [分析方案](3%20docs/01-analysis-plan.md)。
