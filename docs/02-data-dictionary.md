# 字段说明与防泄漏规则

| 字段 | 类型 | 含义 | 时间 | 建模用途 |
|---|---|---|---|---|
| recency | 整数 | 距上次购买的月份，1–12 | 实验前 | 特征 |
| history | 数值 | 过去一年消费金额，美元 | 实验前 | log1p 后使用 |
| history_segment | 类别 | history 对应的消费分档 | 实验前 | 审计；建模省略冗余分箱 |
| mens | 0/1 | 过去一年是否购买男装 | 实验前 | 特征；不是性别 |
| womens | 0/1 | 过去一年是否购买女装 | 实验前 | 特征；不是性别 |
| zip_code | 类别 | Urban/Suburban/Rural 地址地区类别 | 实验前 | 独热编码；不是实际邮政编码 |
| newbie | 0/1 | 过去一年是否为新客户 | 实验前 | 特征 |
| channel | 类别 | 历史购买渠道 Phone/Web/Multichannel | 实验前 | 独热编码 |
| segment | 类别 | 分配到 Mens/Womens/No E-Mail | 实验处理 | 定义 T，禁止作为客户特征 |
| visit | 0/1 | 实验后两周是否访问网站 | 实验后 | 辅助结果，禁止作为购买预测特征 |
| conversion | 0/1 | 实验后两周是否购买 | 实验后 | 主要结果 |
| spend | 数值 | 实验后两周消费金额，美元 | 实验后 | 辅助结果 |

本数据没有唯一客户 ID。本项目的 row_id 只是 CSV 行序号，用来证明训练/验证/测试不重叠，不能将它解释为真实客户身份。

相同特征与相同结果不必然是重复客户。删除这些行会改变实验组样本量和结果，因此原始数据全部保留。IQR 方法标出的高历史消费值只作为审计信号，不能自动认定为错误。

数据来源：[Hillstrom 原始挑战说明](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)、[scikit-uplift 官方数据说明](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html)。本次原始 CSV 来自用户已有压缩包，仅提取数据文件；来源说明和实际获取方式见 reports/tables/data_provenance.json。
