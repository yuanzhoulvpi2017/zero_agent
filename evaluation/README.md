# 推理与评测

按业务建立子目录，放置推理服务入口、批量评测和报告生成代码。

对比商业模型、未训练基座模型和 SFT 模型接入 Agent 后的效果，尽量保持测试任务、工具、提示词和执行限制一致。记录任务成功率、工具调用正确率、耗时与成本。

PaperTrail 当前入口：[`paper_trail/README.md`](paper_trail/README.md)（基座 4B vs SFT 结果与运行方式）。脚本：[`paper_trail/compare_4b.py`](paper_trail/compare_4b.py)。运行 `python evaluation/paper_trail/run_paper_trail_eval.py`。

面向用户的推理交互和评测结果展示统一放在 `interface/<业务名>/`，本模块提供模型推理与评测能力。
