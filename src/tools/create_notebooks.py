"""创建按 01–05 编号的中文教学 Notebook；代码调用统一分析源码。"""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = '''
from pathlib import Path
import sys
root = Path.cwd()
if not (root / "src").exists():
    root = root.parent
sys.path.insert(0, str(root))
import pandas as pd
from src.data import download
from src.config import OUT
df = download()
print(f"客户数：{len(df):,}；字段数：{len(df.columns)}")
'''.strip()

def notebook(name, title, introduction, cells):
    content = [
        nbformat.v4.new_markdown_cell("# " + title + "\n\n" + introduction),
        nbformat.v4.new_code_cell(BOOTSTRAP),
    ]
    for kind, body in cells:
        creator = nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
        content.append(creator(body.strip()))
    document = nbformat.v4.new_notebook(cells=content)
    document.metadata["kernelspec"] = {"display_name": "Python 3 (project .venv)", "language": "python", "name": "python3"}
    document.metadata["language_info"] = {"name": "python", "version": "3.13"}
    target = ROOT / "notebooks" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(document, target)

notebook("01-data-audit.ipynb", "01 数据检查与分析单位",
         "先确定每行代表什么、哪些变量发生在实验前，再开展检验。详见 docs/02-data-dictionary.md。", [
    ("markdown", "## 一、字段与数据类型\n只允许实验前变量用于模型。visit 是实验后的结果，不能拿来预测 conversion 的因果增益。"),
    ("code", "from src.data import audit\nfindings = audit(df)\nprint(findings)\nprint(df.dtypes.to_string())"),
    ("markdown", "## 二、缺失与重复\n没有客户 ID，完全相同的特征和结果不能证明是同一个客户。保留记录，避免改变随机分配组的样本量。"),
    ("code", "print(df.isna().sum().to_string())\nprint('完全相同行：', df.duplicated().sum())\nprint('conversion > visit：', (df.conversion > df.visit).sum())\nprint('spend 最大值：', df.spend.max())"),
    ("markdown", "## 三、特征白名单\nhistory_segment 是 history 的分档，建模省略冗余分箱；这是实验前变量的取舍，不是结果筛选。"),
    ("code", "from src.modeling import feature_frame\nx = feature_frame(df)\nprint('建模字段：', x.columns.tolist())\nprint(x.head().to_string(index=False))"),
])
notebook("02-experiment-validation.ipynb", "02 实验分组与随机化检查",
         "检验是否有明显异常，但不能从检查通过直接推断随机化一定成立。公开分配日志不可用。", [
    ("markdown", "## 一、三组样本量与 SRM\n1:1:1 是本项目的计划比例假设。SRM 用卡方拟合优度检验，告警阈值 0.001。"),
    ("code", "from src.statistics import balance\nvalidation = balance(df)\nprint(df.segment.value_counts().to_string())\nprint(validation)"),
    ("markdown", "## 二、实验前平衡\nSMD 衡量组间变量分布差异。分类变量逐类别转为指示变量，0.1 是检查参考线。"),
    ("code", "smd = pd.read_csv(OUT / 'balance_smd.csv')\nprint(smd.assign(abs_smd=smd.smd.abs()).sort_values('abs_smd', ascending=False).head(10).to_string(index=False))"),
    ("markdown", "## 三、训练、验证、测试\n只按实验分组分层；验证集选择模型和预算；测试集不参与选择。"),
    ("code", "from src.modeling import split_data\ntrain, validation_ids, test = split_data(df)\nprint('样本量：', len(train), len(validation_ids), len(test))\nprint('训练与测试交集：', len(set(train) & set(test)))"),
])
notebook("03-ab-test-analysis.ipynb", "03 A/B 检验、区间与业务解释",
         "主要指标 conversion；辅助指标 visit、含零值的人均 spend。详细假设在 docs/01-analysis-plan.md。", [
    ("markdown", "## 一、绝对与相对提升\n概率差乘以 100 得到百分点；概率差除以对照率得到相对提升。这两个数不能混淆。"),
    ("code", "from src.statistics import rate_difference, ab_analysis, power_analysis\nif not (OUT / 'ab_results.csv').exists():\n    ab_analysis(df)\nresults = pd.read_csv(OUT / 'ab_results.csv')\nprint(results[['treatment','control','metric','effect','relative_lift','ci_low','ci_high','p_holm']].to_string(index=False))"),
    ("markdown", "## 二、直接复算购买率差\n二元结果使用两比例 z 检验和 Newcombe 区间。spend 使用 Welch，允许组间方差不同，并用 bootstrap 补充检查均值差区间。"),
    ("code", "treated = df.loc[df.segment == 'Mens E-Mail', 'conversion']\ncontrol = df.loc[df.segment == 'No E-Mail', 'conversion']\nprint(rate_difference(treated, control))\nprint('绝对百分点：', (treated.mean() - control.mean()) * 100)\nprint('相对提升：', (treated.mean() - control.mean()) / control.mean())"),
    ("markdown", "## 三、多重比较与功效\n12 项固定检验统一 Holm 修正；95% 普通区间为逐项覆盖。功效表是未来 MDE 场景规划，不是观测效应的事后功效。"),
    ("code", "power = power_analysis(df)\nprint(power.to_string(index=False))"),
])
notebook("04-causal-effect-analysis.ipynb", "04 平均因果效应与异质性",
         "随机分配支持 ITT 因果解释，但依赖处理一致性、无重要干扰、完整结果记录等假设。", [
    ("markdown", "## 一、ATE 的透明基准\n随机实验中，邮件组与对照组均值差是平均处理效应的直接估计。合并邮件指随机混合策略，不是总发男装。"),
    ("code", "email = df.segment != 'No E-Mail'\nfor outcome in ['conversion','visit','spend']:\n    print(outcome, '未调整 ATE =', df.loc[email,outcome].mean() - df.loc[~email,outcome].mean())"),
    ("markdown", "## 二、协变量调整\nLin 完全交互回归将实验前协变量中心化，并使用稳健标准误。线性概率模型系数是风险差，不把 odds ratio 当作 ATE。"),
    ("code", "from src.causal import causal_analysis\nif not (OUT / 'ate_adjusted.csv').exists():\n    causal_analysis(df)\nprint(pd.read_csv(OUT / 'ate_adjusted.csv').to_string(index=False))"),
    ("markdown", "## 三、分群效应差\n应直接检验 T×分群交互，而不是依据一组显著、另一组不显著判断不同；BH 修正后仍属于探索证据。"),
    ("code", "interaction = pd.read_csv(OUT / 'heterogeneity_tests.csv')\nprint(interaction[interaction.metric == 'conversion'].sort_values('q_bh_exploratory').to_string(index=False))"),
])
notebook("05-uplift-modeling.ipynb", "05 增益模型与预算模拟",
         "预测谁会买与预测谁因邮件而增量购买不同。模型参数固定，所有策略选择只在验证集进行。", [
    ("markdown", "## 一、S/T/X-Learner\nS 使用一个含处理变量的响应模型；T 分别拟合两组；X 通过交叉拟合响应模型生成伪效应。复用 src/modeling.py，避免第二套分析逻辑。"),
    ("code", "from src.modeling import run_models\nfrom src.budget import budget_analysis\nif not (OUT / 'uplift_metrics.csv').exists():\n    run_models(df)\nif not (OUT / 'budget_locked_policies.csv').exists():\n    budget_analysis()\nmetrics = pd.read_csv(OUT / 'uplift_metrics.csv')\nprint(metrics[(metrics.outcome == 'conversion') & (metrics.split == 'test')].to_string(index=False))"),
    ("markdown", "## 二、为什么不用准确率或 ROC AUC 作为增益标准\n它们评价响应预测，不评价干预的增量。Qini、AUUC、Uplift@K 和分位数组检查因果增益排序。"),
    ("code", "print(pd.read_csv(OUT / 'prediction_metrics.csv').to_string(index=False))\nprint(pd.read_csv(OUT / 'uplift_deciles.csv').query(\"outcome == 'conversion' and model == 'S-Learner'\").to_string(index=False))"),
    ("markdown", "## 三、验证集锁定策略\n低成本不一定要求缩减覆盖。不得看到测试集 T 更好，就改选 T 并声称策略得到了独立验证。"),
    ("code", "import json\nprint(json.loads((OUT / 'policy_lock.json').read_text()))\nprint(pd.read_csv(OUT / 'locked_policy_test.csv').to_string(index=False))"),
    ("markdown", "## 四、预算上限、收益差和敏感性\n假设贡献利润和邮件成本不是实际测得的数据。收益差区间跨零时不能证明模型优于随机投放。"),
    ("code", "print(pd.read_csv(OUT / 'budget_locked_policies.csv').to_string(index=False))\nprint(pd.read_csv(OUT / 'budget_policy_contrasts.csv').to_string(index=False))\nprint(pd.read_csv(OUT / 'sensitivity_analysis.csv').to_string(index=False))"),
    ("markdown", "## 五、推荐阅读顺序\n查看 reports/analysis.md 中的 Qini、分位数组、预算图和局限性，再阅读 docs/04-interview-guide.md。图表是 src/reporting.py 从结果表生成的，不手工填数字。"),
])
print("Five teaching notebooks generated.")
