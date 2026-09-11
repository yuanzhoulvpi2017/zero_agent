# 数据处理

按业务建立子目录，放置轨迹清洗、过滤、标注、数据划分和 SFT 格式转换代码。

输入是原始轨迹和处理规则；输出是版本化的数据集与处理统计。原始轨迹单独保留，测试任务提前留出，避免进入训练数据。生成的数据和模型权重存入 `data/`，不提交 Git。本目录保存处理代码。

PaperTrail 蒸馏采集说明见 [docs/paper_trail/02_dataset.md](../docs/paper_trail/02_dataset.md)。虚拟人与链路格式在 `dataset/paper_trail_data/`，真正对小埋跑对话的入口在 `agent/paper_trail/collect.py`。
