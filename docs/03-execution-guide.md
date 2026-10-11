# 分阶段运行与学习说明

## 一、环境准备

推荐 Python 3.13（本次实际运行版本），Windows PowerShell 中进入项目文件夹：

```powershell
Set-Location -LiteralPath 'E:\项目2（AB）'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

虚拟环境无需激活，直接使用其 python.exe；这样不用修改 PowerShell 执行策略。跨系统安装若锁文件中 Windows 专属依赖不适用，使用 requirements.txt，再记录实际版本。

## 二、第一次运行

```powershell
.\.venv\Scripts\python.exe run.py --stage all
.\.venv\Scripts\python.exe -m pytest -q
```

all 完成取数、审计、实验检验、ATE、模型、模拟、图表及 README。结果覆盖项目内同名生成文件；不会删除原始客户行。GitHub 不附带原始 CSV，首次运行需要网络下载并校验数据；已有通过校验的本地原始 CSV 时可离线运行。

## 三、按阶段学习

| 编号 | 命令中的 stage | 需要学习什么 | 主要输出 |
|---|---|---|---|
| 01 | download / audit | 字段、分母、重复与异常 | audit.json、schema_audit.csv |
| 02 | experiment | 随机化检查、SRM、平衡 | experiment_validation.json、balance_smd.csv |
| 03 | experiment | 比例检验、均值检验、区间、多重修正与功效 | ab_results.csv、power_sample_size.csv |
| 04 | causal | ITT、ATE、协变量调整与组间效应差 | ate_adjusted.csv、heterogeneity_tests.csv |
| 05 | models | 预测概率和增量概率区别、S/T/X、独立评价 | uplift_metrics.csv、uplift_deciles.csv |
| 06 | models | 验证集选择预算、测试集估计收益 | locked_policy_test.csv、policy_contrasts.csv |
| 07 | report | 用结果解释业务及局限性 | README、reports/figures、docs/project-summary.md |

例子：

```powershell
.\.venv\Scripts\python.exe run.py --stage audit
.\.venv\Scripts\python.exe run.py --stage experiment
.\.venv\Scripts\python.exe run.py --stage causal
.\.venv\Scripts\python.exe run.py --stage models
.\.venv\Scripts\python.exe run.py --stage report
```

report 读取既有结果，需要先完成前面的分析阶段。只改文档/图表时无须重新下载或训练。notebooks/ 的 01–05 是同一源码的教学入口，避免维护第二套分析逻辑；使用本项目虚拟环境的 kernel。

## 四、如何理解结果

1. 先读 data-dictionary 和 analysis-plan，确认每个变量发生在实验前还是实验后。
2. 先看各组原始率，再看绝对百分点变化、相对百分比变化和区间。
3. 比较 Holm 修正 p 值；不要只看某个未经修正的 p<0.05。
4. 分群既要看效应区间，也要看交互检验；小样本效应波动大。
5. 模型主要看独立测试集 Qini、AUUC、Uplift@K 与分位数组。响应预测 AUC 只说明能否识别购买者。
6. 业务收益是离线估计加假设，优先看策略差值区间，不能把点估计当作线上收益保证。

## 五、GitHub 仓库发布

公开仓库：[whyyou4564/email-campaign-causal-analysis](https://github.com/whyyou4564/email-campaign-causal-analysis)，默认分支为 main。

在希望保存项目的父目录克隆，然后进入仓库目录执行前面的环境准备与运行命令：

```powershell
git clone https://github.com/whyyou4564/email-campaign-causal-analysis.git
Set-Location -LiteralPath 'email-campaign-causal-analysis'
```

如果使用克隆后的目录，不需要再切换到示例中的 E 盘路径；在当前仓库根目录创建 .venv 即可。

项目本身是独立 Git 仓库结构。原始数据、虚拟环境、模型预测与临时输出默认忽略；README、方法文档、测试、教学 notebooks、精选报告表和图表可发布。

新建独立仓库时，可以使用以下 Git 命令发布；已经克隆仓库时无需重复初始化或添加远程地址：

```powershell
git init -b main
git branch -M main
git add .
git commit -m "Build email experiment and uplift analysis"
git remote add origin <你的GitHub仓库地址>
git push -u origin main
```

发布分支使用 main。原始数据由下载脚本获取，完整计算结果可以重新运行生成；公开仓库只保留精选聚合表，避免提交逐行客户预测、环境与本地备份。

教学 Notebook 还可以用 python src/tools/run_notebooks.py 逐单元执行，结果保存到 Notebook。这个入口不启动 Jupyter kernel，便于在禁止本地内核通信的环境验证代码；本次使用该方式执行了所有代码单元。

已有 ZIP 数据的规范导入方式：python src/tools/import_local_data.py <压缩包路径>。该脚本只读取原始 CSV，核对 SHA-256，不读取或解压其他项目的代码。

本次 scikit-uplift 0.5.1 对 scikit-learn 的 stable_cumsum 调用会产生弃用警告，分析实际运行通过。requirements.txt 限制 scikit-learn <1.10，并锁定本次运行版本；未修改第三方库源文件。

测试临时文件保存在 tests/.cache，Git 和交付打包均忽略该目录，避免依赖系统临时目录的写入权限。
