# 介绍agentscope这个框架

在上一期的agent介绍中，那qwen-agent来做了开胃菜。这期是介绍agentscope框架，感觉是一个更加现代的框架，强烈推荐。
1. 这个框架支撑mcp、skill、等功能。
2. 支持外部的记忆读写（长期记忆）；这个绑定的角度也非常有意思，是通过转换成tool的形式。
3. 框架的定制化更高，对各种模型做适配（agent的message转换成大模型需要的message）
4. 支持对话压缩；支持结构化输出（这个主要是流程上的优化，可以大概研究一下）
5. 支持hook，这个实现比较复杂，但是很容易开发和使用。

相关流程图已经放在了[support_file/agentscope_reactagent.drawio](support_file/agentscope_reactagent.drawio)文件里面。

关于代码：
1. [code03_agentscope_001.py](code03_agentscope_001.py): 简单的agentscope代码体验。
2. [code03_agentscope_002_hook.py](code03_agentscope_002_hook.py): hook的使用体验。


