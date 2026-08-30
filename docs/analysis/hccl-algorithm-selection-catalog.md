# HCCL AI_CPU算法选择条件与粗粒度决策表

## 1. 文档目标

本文不再逐个解释HCCL中的细粒度Executor名称，而是回答更实用的问题：

> 在AI_CPU展开模式下，具备什么前置条件时，HCCL大致会选择哪一类算法？

本文基于`master`分支提交`7696133a`，重点分析以下两层决策：

1. 用户通过`HCCL_ALGO`为Server间和超节点间通信指定什么算法。
2. AI_CPU Selector根据拓扑、数据特征和运行模式选择什么粗粒度执行方案。

源码中的OneShot、TwoShot、Chunk、MultiLink、2Die、SingleChannel等名称被视为同一算法族内部的实现变体，不作为本文的主要分类。

## 2. 一页结论

### 2.1 只需要记住五类执行方案

| 粗粒度算法族 | 典型含义 | 通常在什么条件下使用 |
| --- | --- | --- |
| **Mesh单平面** | 在Server内或一个全连接高速平面内完成通信 | 单层拓扑，并且Mesh可以覆盖全部rank |
| **NHR单平面** | 将通信域视为一个NHR平面 | CLOS、多节点小数据、拓扑退化或拓扑不对称 |
| **Mesh+NHR跨层** | Server内走Mesh，Server间走NHR | 两层拓扑、每个Server有多卡，并且数据量值得拆层并行 |
| **多层流水/并行** | 多个通信层并行或分块流水 | 大数据、两层或三层对称拓扑、PCIe/UBX/UBoE等混合链路 |
| **保序/特殊归约** | 保证浮点归约顺序，或使用AI_CPU完成特殊数据类型归约 | strict确定性、64位数据类型或PROD |

### 2.2 总决策流程

```text
产品和算子支持AI_CPU吗？
  └─ 否：不进入本文的AI_CPU选择流程
  └─ 是
      ↓
是否是AllReduce/ReduceScatter的strict保序场景？
  └─ 是：保序算法，优先级最高
  └─ 否
      ↓
是否是64位归约或PROD？
  └─ 是：跨非全连接通信域时，优先特殊归约/NHR路径
  └─ 否
      ↓
拓扑只有一层吗？
  ├─ 纯Mesh或Mesh能覆盖全部rank：Mesh单平面
  ├─ 纯CLOS：通常使用NHR/MultiLink
  └─ Mesh+CLOS混合且Mesh不能覆盖全部rank：跨层或流水
      ↓
拓扑有两层或三层吗？
  ├─ 层内组退化、Level1Nhr=true或拓扑不对称：NHR单平面
  ├─ 两层对称且数据较小：通常仍使用NHR单平面
  ├─ 两层对称且数据较大：Mesh+NHR并行
  └─ 三层对称或超大数据：多层Sequence/Pipeline
```

这里的“通常”很重要：`HCCL_ALGO`、产品支持范围、新旧Selector、资源回退和Tuner都可能进一步限制候选算法。

## 3. 两层选择不能混为一谈

### 3.1 HCCL_ALGO选择通信层算法

公开格式如下：

```bash
export HCCL_ALGO="allreduce=level0:NA;level1:NHR;level2:NHR"
```

按算子配置时共有7个公开分组：

| 配置名 | 覆盖算子 |
| --- | --- |
| `allreduce` | AllReduce |
| `reducescatter` | ReduceScatter、ReduceScatterV |
| `allgather` | AllGather、AllGatherV |
| `broadcast` | Broadcast |
| `reduce` | Reduce |
| `scatter` | Scatter |
| `alltoall` | AlltoAll、AlltoAllV、AlltoAllVC |

- `level0`是Server内通信，目前公开配置只能写`NA`，由HCCL自动选择。
- `level1`是Server间通信，可选择Ring、H-D_R、NHR、NB、AHC、Pipeline等。
- `level2`是超节点间通信，只有部分产品和AI_CPU场景支持配置。

### 3.2 AI_CPU Selector选择完整执行方案

AI_CPU Selector再根据以下条件决定是Sole、Parallel、Sequence、Pipeline，以及是否使用PCIe、UBX、UBoE、MultiLink等实现：

- 拓扑层数和每层的rank分组。
- Level0是Mesh、CLOS还是Mesh+CLOS。
- Mesh是否能够连接当前层的全部rank。
- 各层是否对称。
- 数据量、rank数和数据类型。
- strict确定性、PROD、原地计算和对称内存等能力。

因此，`HCCL_ALGO=NHR`并不等于直接指定一个名为`AicpuXXXNHR`的Executor；它配置的是通信层算法，完整执行方案仍需经过Selector。

## 4. 人工配置HCCL_ALGO时怎么选

如果不设置`HCCL_ALGO`，HCCL默认根据产品形态、节点数和数据量自适应选择。只有确实需要固定算法或做性能对比时，才建议手工配置。

### 4.1 Level1 Server间算法

| 算法 | 建议使用的前置条件 | 不适合或需要注意 |
| --- | --- | --- |
| `ring` | Server数较少、数据较小、网络存在明显拥塞，或者其他低时延算法不适用 | 通信步数随Server数线性增加，大规模时延较高 |
| `H-D_R` | Server数是2的整数次幂；或者不是2的整数次幂但数据较小 | 非2次幂规模会产生额外通信量 |
| `NHR` | Server数较多、拓扑不规则或非2次幂，并且不适合Pipeline | Ascend 950PR/Ascend 950DT当前只允许配置NHR |
| `NHR_V1` | 仅用于兼容历史版本，且Server数为非2次幂 | 理论性能低于新版NHR，后续会逐步停用 |
| `NB` | Server数较多、希望使用对数步数算法，且产品和算子支持 | 需要查看产品支持矩阵，不是所有算子都支持 |
| `AHC` | 多层、非对称NPU分布，或者层间存在明显带宽收敛 | 主要面向AllReduce、ReduceScatter、AllGather；Level1设为AHC后Level2自动使用AHC |
| `pipeline` | 数据量大、每个Server有多卡，可以同时利用Server内和Server间链路 | 小数据启动开销不划算；部分产品、模式或确定性场景不支持 |
| `pairwise` | AlltoAll系列大数据，希望避免网络“一打多”热点 | 仅用于AlltoAll/AlltoAllV/AlltoAllVC，步数较多并需要额外内存 |

可以把人工选择顺序粗略理解为：

```text
Ascend 950PR/950DT → NHR
AlltoAll大数据且要规避一打多 → Pairwise
多层非对称或层间带宽收敛 → AHC
大数据 + 每机多卡 + 支持流水 → Pipeline
节点数为2的整数次幂 → H-D_R
节点数很多或非规则规模 → NHR/NB
节点数少、小数据或网络易拥塞 → Ring
```

这只是选型建议，不代表跳过产品、算子、数据类型和运行模式支持检查。

### 4.2 Level2超节点间算法

| 前置条件 | 建议算法 |
| --- | --- |
| 超节点数较少且不是2的整数次幂 | `ring` |
| 超节点数是2的整数次幂，或者非2次幂但数据较小 | `H-D_R` |
| 超节点数较多 | `NHR`或`NB` |
| 大数据，并且每个超节点内包含多卡 | `pipeline` |
| Atlas A3产品的多层非对称场景，且层间带宽收敛 | 可评估`AHC`，由Level1 AHC联动 |

不设置Level2时，公开文档给出的默认规则是：超节点数小于8且不是2的整数次幂时使用Ring，其余场景使用H-D_R。产品自身的特殊限制仍然优先。

## 5. AI_CPU自动选择的前置条件

### 5.1 第零关：产品和算子必须支持AI_CPU

| 产品 | AI_CPU粗粒度范围 |
| --- | --- |
| Ascend 950PR/Ascend 950DT | `AICPU_TS`为默认模式，支持主要集合通信算子和P2P算子 |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | `AI_CPU`为默认模式，支持全量集合通信；归约类的数据类型和归约操作有限制 |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | 公开文档中AI_CPU主要支持AllGather和AlltoAll系列；其他算子通常使用HOST或AIV路径 |
| Atlas 300I Duo 推理卡 | 仅支持单机单通信域AllReduce，并且有运行模式限制 |

如果产品、算子或数据类型不支持AI_CPU，后续拓扑规则都不会生效。

### 5.2 选择优先级

AI_CPU旧选择器的规则不是“所有条件一起打分”，而是按代码分支顺序匹配。粗粒度优先级如下：

| 优先级 | 前置条件 | 结果 |
| ---: | --- | --- |
| 1 | AllReduce/ReduceScatter开启strict保序 | 直接选择保序族，不再走普通性能算法 |
| 2 | 多层或非全连接域中的INT64/UINT64/FP64，或归约操作为PROD | 特殊AI_CPU归约/NHR路径 |
| 3 | Level1分组GCD为1，导致`Level1Nhr=true`；或者Level0本地组只有1个rank | 退化为NHR单平面 |
| 4 | 三层拓扑是否对称、顶层是否为UBoE | 对称时使用多层并行/流水；不对称时通常回退NHR |
| 5 | Level0拓扑和链路是否全连通 | 全Mesh走Mesh；CLOS走NHR；混合拓扑走跨层方案 |
| 6 | 数据量和rank数 | 在Sole、Parallel、Sequence、Pipeline之间进一步选择 |

`Level1Nhr=true`不是环境变量直接设置的标志。源码在Level0各实例大小的最大公约数为1时设置该标志，表示继续拆分Mesh已经没有意义。

## 6. 按算子给出“条件 → 算法族”

以下用：

- `D`表示单rank数据量，即`count × sizeof(dataType)`。
- `R`表示通信域rank数。
- `T = D × R`表示Selector某些分支使用的总数据规模指标。

阈值来自当前源码快照，是实现参数，不是稳定API。后续版本可能调整。

### 6.1 AllReduce

| 前置条件 | 粗粒度选择 |
| --- | --- |
| strict保序 | 保序族；`R <= 8`使用直接保序，`R > 8`使用分组保序 |
| 多层拓扑，并且是64位数据类型或PROD | 特殊AI_CPU归约 + NHR |
| 单层纯Mesh，或Mesh可以覆盖全部rank | Mesh单平面；数据量只决定OneShot/TwoShot/Chunk，不改变大类 |
| 单层纯CLOS | NHR MultiLink |
| 单层Mesh+CLOS，Mesh不能全连通 | PCIe/UBX跨层方案；大数据更倾向Pipeline |
| 两层Mesh，`D <= 32MB` | NHR单平面，避免小数据拆层开销 |
| 两层Mesh，`32MB < D <= 4GB` | Mesh+NHR并行 |
| 两层Mesh，`D > 4GB` | Mesh+NHR Sequence |
| 三层对称 | Mesh+NHR+NHR；顶层UBoE时优先多层Pipeline/RSAG |
| 三层不对称、`Level1Nhr=true`或Level0组退化 | NHR单平面 |

### 6.2 ReduceScatter

| 前置条件 | 粗粒度选择 |
| --- | --- |
| strict保序 | 保序族；`R <= 8`直接保序，`R > 8`分组保序 |
| 多层拓扑，并且是64位数据类型或PROD | 特殊AI_CPU归约 + NHR |
| 单层纯Mesh或全Mesh连通 | Mesh单平面；大数据切Chunk |
| 单层纯CLOS | NHR MultiLink |
| 单层Mesh+CLOS且Mesh不全连通 | PCIe/UBX跨层；大数据倾向Pipeline |
| 两层Mesh，`D <= 4MB` | NHR单平面 |
| 两层Mesh，`D > 4MB`且`T <= 4GB` | Mesh+NHR并行 |
| 两层Mesh，`T > 4GB` | Mesh+NHR Sequence |
| 超大规模`R >= 256`且`T >= 1GB` | 即使单rank数据不大，也优先Mesh+NHR并行 |
| 三层对称 | Mesh+NHR+NHR；顶层UBoE时可走Pipeline |
| 三层不对称或拓扑退化 | NHR单平面 |

### 6.3 AllGather

| 前置条件 | 粗粒度选择 |
| --- | --- |
| 单层纯Mesh或全Mesh连通 | Mesh单平面 |
| 单层纯CLOS | NHR MultiLink |
| 单层Mesh+CLOS且Mesh不全连通 | PCIe/UBX跨层；大数据倾向Pipeline |
| 两层Mesh，`D <= 1MB` | NHR单平面 |
| 两层Mesh，`D > 1MB`且`T <= 4GB` | Mesh+NHR并行 |
| 两层Mesh，`T > 4GB` | Mesh+NHR Sequence |
| 超大规模`R >= 1024`且`T >= 1GB` | 优先Mesh+NHR并行 |
| 三层对称 | Mesh+NHR+NHR；顶层UBoE时可走Pipeline |
| 三层不对称或拓扑退化 | NHR单平面 |

AllGather没有归约计算，因此不存在PROD和特殊AI_CPU归约分支。

### 6.4 Broadcast、Reduce、Scatter

这三个算子在AI_CPU旧选择器中主要看拓扑，数据量的影响远小于AllReduce、ReduceScatter、AllGather。

| 前置条件 | Broadcast | Reduce | Scatter |
| --- | --- | --- | --- |
| 单层纯Mesh或全Mesh连通 | Mesh | Mesh；`D >= 8MB`仅切TwoShot | Mesh |
| 单层纯CLOS | NHR MultiLink | NHR；特殊归约使用AI_CPU Reduce | 通常为Mesh；PCIe混合时使用NHR |
| 单层Mesh+CLOS，Mesh不全连通 | Mesh+NHR跨层 | Mesh+NHR跨层 | Mesh+NHR跨层 |
| 两层Mesh且未退化 | Mesh+NHR | Mesh+NHR | Mesh+NHR |
| 两层CLOS、Level1Nhr或本地组大小为1 | NHR | NHR | NHR |
| 三层对称 | Mesh+NHR+NHR或UBoE并行 | Mesh+NHR+NHR或UBoE并行 | 顶层非UBoE时使用Mesh+NHR+NHR，否则回退NHR |
| 三层不对称 | NHR | NHR | NHR |

### 6.5 AllGatherV、ReduceScatterV

变长算子的外层选择非常简单：

| 算子 | 前置条件 | 粗粒度选择 |
| --- | --- | --- |
| AllGatherV | 拓扑层数为1～3 | 固定使用AI_CPU SoleMesh外层方案 |
| ReduceScatterV | 拓扑层数为1～3，且数据类型不是UINT64/FP64 | 固定使用AI_CPU SoleMesh外层方案 |

这里的SoleMesh是Executor外层组织方式，不代表Server间一定只能使用Mesh。变长数据分片和层间算法由Executor内部继续处理。

### 6.6 AlltoAll、AlltoAllV、AlltoAllVC

AlltoAll系列在当前AI_CPU Selector中统一归为Mesh外层执行族：

| 前置条件 | 粗粒度选择 |
| --- | --- |
| 纯Mesh或纯CLOS | Mesh；AlltoAll根据数据规模选择单通道或多通道 |
| Mesh+CLOS且属于PCIe混合拓扑 | Mesh基础方案 |
| Mesh+CLOS的UBX规则小规模组，通常`R <= 4` | Mesh Concurrent |
| 其他UBX场景 | Mesh UBX |
| AlltoAllVC | 固定AI_CPU SoleMesh外层方案 |

这不与`HCCL_ALGO=...pairwise`冲突：Pairwise/Pipeline是Level1通信算法配置，而这里的Mesh是完整Executor的外层组织方式。对于AlltoAll大数据和网络一打多问题，仍可在产品支持时配置Pairwise。

## 7. 三个典型例子

### 7.1 单Server全Mesh的AllReduce

前置条件：单层拓扑、Mesh可以覆盖全部8个rank、普通数据类型、未开启strict。

结果：选择 **Mesh单平面**。4MB和64MB可能使用不同的OneShot/TwoShot/Chunk实现，但从架构角度仍是同一类算法。

### 7.2 两层拓扑的AllGather

前置条件：每个Server多卡、Level0为Mesh、拓扑未退化、单rank数据2MB、总数据小于4GB。

结果：超过AllGather的1MB拆层阈值，选择 **Mesh+NHR并行**；如果总数据进一步超过4GB，则切为 **Sequence**。

### 7.3 三层非对称拓扑的ReduceScatter

前置条件：三层拓扑，但Level0/Level1分组不对称。

结果：不采用复杂的Mesh+NHR+NHR编排，保守地退回 **NHR单平面**。如果同时开启strict，则strict保序优先于这条拓扑规则。

## 8. 新旧Selector的差异

### 8.1 默认旧选择器

`HCCL_USE_NEW_SELECTOR`默认值为`0`。旧选择器按照各算子`SelectAicpuAlgo`中的`if/else`顺序匹配，前文的算子决策表主要描述这一流程。

多数AI_CPU `SelectAicpuAlgo`函数不直接读取`configAlgMap`，说明完整Executor的自动选择主要依赖拓扑和算子参数。`HCCL_ALGO`对通信层算法的约束不能简单理解为字符串匹配Executor名称。

### 8.2 新选择器

当`HCCL_USE_NEW_SELECTOR=1`时，当前仅AllReduce、ReduceScatter、AllGather进入成本模型流程：

```text
候选执行引擎
  → 通信域配置/HCCL_ALGO解析并约束候选
  → 拓扑属性过滤
  → 数据类型/PROD/In-place/保序能力过滤
  → 根据数据量、带宽、链路利用率计算cost
  → Tuner可选修正cost
  → 选择最小cost算法
```

因此新选择器没有一张完全固定的“阈值命中表”。在相同拓扑下，数据量、链路利用率、候选引擎和Tuner都可能改变最终结果。Broadcast、Reduce、Scatter、AlltoAll等算子仍使用旧规则。

`HCCL_USE_NEW_SELECTOR`是当前源码中的选择器开关，未列入公开环境变量文档，不应把它当作稳定的业务接口。
新选择器内部还存在面向成本模型的算法解析语法；业务配置仍应以公开`HCCL_ALGO`文档为准，不应依赖内部语法。

## 9. 配置与排查建议

1. 优先不设置`HCCL_ALGO`，先让HCCL自适应选择。
2. 需要固定算法时，先确认产品、算子、数据类型和运行模式支持，再配置Level1/Level2。
3. AI_CPU展开由`HCCL_OP_EXPANSION_MODE=AI_CPU`或`AICPU_TS`控制，具体取值取决于产品支持；它与`HCCL_ALGO`是两个不同维度。
4. `HcclCommConfig.hcclAlgo`的优先级高于环境变量`HCCL_ALGO`。
5. strict确定性可能覆盖普通性能算法；Atlas A2 strict保序场景不建议手工配置`HCCL_ALGO`。
6. CCU/AIV资源或能力不足时可能回退AI_CPU，因此即使未显式配置AI_CPU，也可能进入本文的AI_CPU选择分支。
7. 排查最终选择时，优先查看日志中的`Algo match`、`Level1Nhr`、`topoLevelNums`、`level0Topo`和`SelectMinCost`。

## 10. 源码入口

环境变量和产品支持：

- `docs/zh/user_guide/hccl_env/HCCL_ALGO.md`
- `docs/zh/user_guide/hccl_env/HCCL_OP_EXPANSION_MODE.md`
- `docs/zh/user_guide/hccl_env/inter_server_algo_support.md`
- `docs/zh/user_guide/hccl_env/inter_superpod_algo_support.md`

AI_CPU旧选择器：

- `src/ops/all_reduce/selector/all_reduce_auto_selector.cc::SelectAicpuAlgo`
- `src/ops/reduce_scatter/selector/reduce_scatter_auto_selector.cc::SelectAicpuAlgo`
- `src/ops/all_gather/selector/all_gather_auto_selector.cc::SelectAicpuAlgo`
- `src/ops/broadcast/selector/broadcast_auto_selector.cc::SelectAicpuAlgo`
- `src/ops/reduce/selector/reduce_auto_selector.cc::SelectAicpuAlgo`
- `src/ops/scatter/selector/scatter_auto_selector.cc::SelectAicpuAlgo`
- `src/ops/all_to_all_v/selector/*_auto_selector.cc::SelectAicpuAlgo`

公共选择与新成本模型：

- `src/ops/op_common/selector/auto_selector_base.cc`
- `src/ops/op_common/selector/selector_engine.cc`
- `src/ops/op_common/selector/cost_model.cc`
- `src/ops/op_common/selector/cost_table.cc`
- `src/ops/op_common/topo/topo_host.cc::CalcLevel1Nhr`
- `src/common/alg_parse.cc::FilterCmByHcclAlgo`
