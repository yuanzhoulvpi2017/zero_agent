# 面向算法的agent开发实战（python）


## 结构

| 名称 | 介绍 | B站视频 |
| --- | --- | --- |
| [code01_intro.md](https://github.com/yuanzhoulvpi2017/zero_agent/blob/main/code01_intro.md) | 介绍比较好用的，可以对标openclaw的agent开源框架（python） | [面向python、算法工程师的agent开发教程 s1-介绍](https://www.bilibili.com/video/BV12AwDzgEwe) |
| [code02_agent.md](https://github.com/yuanzhoulvpi2017/zero_agent/blob/main/code02_agent.md) | 调用大模型（LLM）和Agent的案例，包含MCP工具调用示例及Agent消息流程图、基于qwen-agent进行介绍 | [面向python、算法工程师的agent开发教程 s2-调用大模型和Agent](https://www.bilibili.com/video/BV1XADnBBEMy) |
| [code03_agentscope.md](https://github.com/yuanzhoulvpi2017/zero_agent/blob/main/code03_agentscope.mdd) | 对agentscope框架的介绍和使用 | [面向python、算法工程师的agent开发教程 s3-简单介绍一下agentscope的闪光点～](https://www.bilibili.com/video/BV1ksd9BcEx3) |








## 常见问题解答（FAQ）

### 当前 Agent 开发是否有必要使用现有框架？

**结论：建议优先使用成熟的开源框架，而非从零构建。**

尽管借助 AI 辅助编程工具（如 Codex、Claude Opus 等）可以快速实现一套自定义 Agent 框架，但这并不意味着此举是合理的工程选择。具体理由如下：

1. **社区验证与可靠性**：以 AgentScope、Qwen-Agent 为代表的主流框架，均已经过广泛的社区验证，存在公开的 Issue 追踪与修复记录。其核心开发者的工程能力普遍较强，框架的设计质量与稳定性有一定保障。
2. **AI 生成代码的局限性**：AI 虽然能够生成框架代码，但其输出并不保证完全正确，潜在的边界问题与隐性缺陷往往不会被主动暴露，存在一定的工程风险。
3. **重复造轮子的成本**：自研框架本质上仍需实现 LLM 响应解析、工具调用检测、循环调度、输出整理等基础能力，工程收益有限，属于典型的低价值重复劳动。
4. **Agent 开发的核心投入方向应聚焦于以下几点：**
   - 4.1 持续优化 Agent 的系统提示词（System Prompt）及工具定义（Tool Schema）；
   - 4.2 建立完善的测试体系，尤其是安全性测试，涵盖对抗性攻击、越权访问等场景；
   - 4.3 强化 Agent 的运行管理机制，包括状态管理、异步调用、并发控制与协程调度等；
   - 4.4 构建统一的 Agent 管理平台，对接多种框架，统一管控输入输出规范、并发限制与调用策略，实现框架无关的标准化治理。

