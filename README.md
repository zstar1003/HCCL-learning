# HCCL Learning

本仓库整理HCCL在AI_CPU场景下的常见集合通信算法选择资料，重点回答三个问题：

1. 给定算子、组网拓扑和数据规模，HCCL如何选择通信路径？
2. Algorithm、Executor和Template分别负责什么，三者如何衔接？
3. 面向方案估算和汇报时，应该用什么粒度描述算法选择？

> 本仓库是基于开源代码整理的学习与分析材料，不代表官方产品规格、性能承诺或完整Selector实现。

## 一页总览

![AI_CPU常见集合通信固定选择路线图](assets/hccl-aicpu-selector-one-slide-archify-tree-1920x1080.png)

核心判断顺序：

1. **先看覆盖条件**：`strict`、特殊归约等条件优先于普通性能规则。
2. **再看拓扑层数**：单层、两层、三层分别进入不同决策分支。
3. **再看层内形态**：Mesh、CLOS、连通性、规则性、对称性和UBoE能力决定可用通信路径。
4. **最后看数据规模**：数据档位进一步决定Sole、Parallel、Sequence、Pipeline或Concur等Executor。
5. **Template负责落地执行细节**：例如OneShot、Chunk、MultiLink等，不与算法或Executor混为一谈。

## 仓库内容

| 分类 | 文件 | 用途 |
| --- | --- | --- |
| 系统分析 | [算法选择目录](docs/analysis/hccl-algorithm-selection-catalog.md) | 算法、Executor、Template和Selector的整体关系 |
| 触发规则 | [AI_CPU常见算法触发条件](docs/analysis/hccl-aicpu-common-algorithm-triggers.md) | 按常见集合通信场景梳理触发条件 |
| 汇报材料 | [领导汇报版](docs/briefing/hccl-algorithm-selection-leadership-brief.md) | 面向非实现人员解释决策方法和估算口径 |
| 单页说明 | [一页PPT版说明](docs/briefing/hccl-aicpu-selector-one-slide.md) | 路线图、字段解释、阈值和演讲备注 |
| PPT图片 | [1920×1080路线图](assets/hccl-aicpu-selector-one-slide-archify-tree-1920x1080.png) | 可直接放入16:9汇报材料 |
| 交互图 | [Archify交互版](diagrams/hccl-aicpu-selector-one-slide-archify-tree.html) | 本地浏览完整分支和引导视图 |
| 图表源文件 | [Archify Workflow JSON](diagrams/hccl-aicpu-selector-one-slide-archify-tree.workflow.json) | 后续修改、校验和重新导出 |

## 推荐阅读顺序

1. 先看一页总览，建立“约束→拓扑→数据规模→Executor”的主线。
2. 再看领导汇报版，理解方案估算时应该采用的抽象粒度。
3. 然后阅读常见算法触发条件，掌握单层、两层和三层组网的典型出口。
4. 最后查算法选择目录，用于回到代码结构和细粒度实现。

## 分析范围

- 代码来源：[cann/hccl](https://gitcode.com/cann/hccl)
- 代码基线：提交`7696133a`
- 执行引擎：AI_CPU
- 重点算子：AllReduce、ReduceScatter、AllGather
- 重点内容：常见Selector路径、Executor组织方式与Template族

真实运行结果仍可能受到产品型号、通信域资源、数据类型、归约类型、确定性要求、Selector版本和运行时回退逻辑影响。

## 查看交互图

克隆仓库后，用浏览器打开：

```text
diagrams/hccl-aicpu-selector-one-slide-archify-tree.html
```

交互图支持浅色／深色主题、路径聚焦、章节引导和图表导出。
