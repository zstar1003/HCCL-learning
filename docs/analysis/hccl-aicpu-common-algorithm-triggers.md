# HCCL AI_CPU常见算法触发条件速查

## 1. 适用范围

本文用于快速判断：在AI_CPU展开模式下，HCCL旧Selector通常会在什么条件下选择SoleMesh、SoleNHR、Parallel、Sequence、Pipeline、Concur和保序算法。

本文基于`master`分支提交`7696133a`，只描述常见算法族，不展开OneShot、TwoShot、Chunk等全部实现细节。完整的环境变量和算子决策分析参见[HCCL AI_CPU算法选择条件与粗粒度决策表](hccl-algorithm-selection-catalog.md)。

本文沿用源码日志中的宽泛说法，将通信模板和Executor编排统称为“算法”。严格来说，Mesh/NHR属于通信模板，Parallel/Sequence/Pipeline/Concur属于编排方式。Ring、H-D_R、NB、AHC等`HCCL_ALGO`公开配置值的人工选型条件，见上述完整分析文档第4节。

前置假设：

- 产品、算子和数据类型支持AI_CPU。
- 已经通过`HCCL_OP_EXPANSION_MODE`进入AI_CPU，或者其他执行引擎失败后回退AI_CPU。
- `HCCL_USE_NEW_SELECTOR=0`。该开关默认值为`0`；新Selector的差异见第8节。

## 2. 最短判断路径

```text
strict保序？
  └─ 是 → StrictOrdered/OrderPreserved
  └─ 否
      ↓
64位归约或PROD，并且需要跨非全连接通信域？
  └─ 是 → NHRAicpuReduce类特殊归约
  └─ 否
      ↓
单层且全Mesh连通？
  └─ 是 → SoleMesh
  └─ 否
      ↓
拓扑退化、非对称、纯CLOS或多层小数据？
  └─ 是 → SoleNHR
  └─ 否
      ↓
两层拓扑且数据量值得拆层？
  └─ 是 → ParallelMeshNHR
      └─ 数据特别大 → Sequence或Pipeline
  └─ 否
      ↓
三层对称拓扑？
  └─ 是 → SequenceMeshNHRNHR或UBoE Pipeline
```

## 3. 触发条件中的几个变量

| 符号或字段 | 含义 |
| --- | --- |
| `D` | 单rank数据量，通常为`count × sizeof(dataType)` |
| `R` | 通信域rank数 |
| `T` | 总数据规模指标，`T = D × R` |
| `topoLevelNums` | 拓扑层数；1表示单层，2表示Server内+Server间，3表示再增加超节点间层次 |
| `level0Topo` | Level0拓扑，常见值为`MESH_1D`、`CLOS`、`MESH_1D_CLOS` |
| `Level1Nhr` | Level0各分组大小的最大公约数为1，无法继续形成有效Mesh层次，需要退化为NHR |
| `level0Symmetric/level1Symmetric` | 相应拓扑层的分组是否对称 |
| `level0PcieMix` | Level0是否包含PCIe混合链路 |
| `topLevelUboe` | 顶层链路是否为UBoE |

## 4. 常见算法族触发总表

| 常见算法族 | 主要触发条件 | 常见算法名示例 | 主要用途 |
| --- | --- | --- | --- |
| **SoleMesh** | 单层、Mesh能覆盖全部rank | `AicpuAllReduceSoleMeshOneShot`、`AicpuAllGatherSoleMesh` | 使用一个高速Mesh平面完成通信 |
| **SoleNHR** | 纯CLOS、拓扑退化、拓扑不对称，或者多层小数据 | `AicpuAllReduceSoleNHR`、`AicpuAllGatherSoleNHR` | 避免拆层和复杂编排开销 |
| **ParallelMeshNHR** | 两层拓扑、Level0为Mesh、每个Level0组包含多个rank，且数据超过拆层阈值 | `AicpuAllReduceParallelMeshNHR` | 同时利用Server内Mesh和Server间NHR |
| **SequenceMeshNHR** | 两层超大数据，或者三层对称拓扑 | `AicpuAllGatherSequenceMeshConcurNHR` | 分阶段处理，控制资源并支持更大数据 |
| **SequenceMeshNHRNHR** | 三层对称、非UBoE顶层 | `AicpuReduceScatterSequenceMeshConcurNHRNHR` | 按Mesh、Server间NHR、超节点间NHR依次通信 |
| **Pipeline** | 混合链路大数据，或者对称UBoE多层拓扑 | `AicpuAllReducePipeLinePcie`、`AicpuAllGatherPipeLineUBX` | 按chunk重叠不同通信层 |
| **Concur** | 小规模、规则且对称的UBX/Mesh+CLOS组合 | `AicpuAllGatherConcurMeshNHR` | 并发利用Mesh和CLOS链路 |
| **StrictOrdered/OrderPreserved** | AllReduce或ReduceScatter开启strict保序 | `AicpuAllReduceStrictOrderedMesh` | 保证浮点归约顺序 |
| **NHRAicpuReduce** | 64位归约或PROD跨越CLOS、多层或非全连接域 | `AicpuAllReduceSoleNHRAicpuReduce` | 使用AI_CPU执行不适合普通模板的归约计算 |

## 5. 每类算法什么时候触发

### 5.1 SoleMesh

常见前置条件：

1. `topoLevelNums == 1`。
2. `level0Topo == MESH_1D`，或者虽然是`MESH_1D_CLOS`，但Mesh已经能够连接该层全部rank。
3. 没有strict保序等更高优先级条件。

常见算子行为：

- AllReduce：小数据使用OneShot，中大数据切TwoShot或Chunk，但仍属于SoleMesh。
- ReduceScatter：普通数据使用Mesh，大数据切MeshChunk。
- AllGather/Broadcast/Reduce/Scatter：单层全Mesh时基本直接选择SoleMesh。
- AllGatherV/ReduceScatterV：当前AI_CPU Selector固定使用SoleMesh外层方案。
- AlltoAll系列：外层Executor基本都属于SoleMesh。

### 5.2 SoleNHR

满足以下任一条件时经常触发：

- `level0Topo == CLOS`。
- `Level1Nhr == true`。
- Level0本地分组大小为1，拆分Mesh没有收益。
- 三层拓扑不对称。
- 两层拓扑的数据量较小，不值得执行Mesh+NHR拆层。
- UBX/Mesh+CLOS拓扑不满足规则并发或流水条件。

例外：Scatter在单层纯CLOS且不是PCIe混合链路时选择Mesh；AlltoAll系列的外层Executor也统一归入Mesh，不能仅凭CLOS判断它们会选择SoleNHR。

三大算子的两层小数据边界：

| 算子 | 通常仍选择SoleNHR的范围 |
| --- | --- |
| AllReduce | `D <= 32MB` |
| ReduceScatter | `D <= 4MB` |
| AllGather | `D <= 1MB` |

这些阈值只有在前面的特殊类型、UBoE、拓扑退化等分支均未提前命中时才适用。

### 5.3 ParallelMeshNHR

典型前置条件：

1. 两层拓扑。
2. Level0为Mesh且每个Level0组包含多个rank。
3. 没有`Level1Nhr`退化。
4. 数据超过拆层并行阈值，但尚未达到Sequence阈值。

| 算子 | Parallel触发区间 |
| --- | --- |
| AllReduce | `32MB < D <= 4GB` |
| ReduceScatter | `D > 4MB`且`T <= 4GB` |
| AllGather | `D > 1MB`且`T <= 4GB` |
| Broadcast/Reduce/Scatter | 两层Mesh且未退化时通常直接采用跨层并行，不再按数据量细分 |

额外的大rank规则：

- ReduceScatter：`R >= 256`且`T >= 1GB`时优先Parallel。
- AllGather：`R >= 1024`且`T >= 1GB`时优先Parallel。

这两个大rank判断位于普通数据量分支之前，即使`T > 4GB`，也可能优先保持Parallel而不是切换Sequence。

### 5.4 Sequence

Sequence通常在两种场景触发：

| 场景 | 触发条件 | 典型结果 |
| --- | --- | --- |
| 两层超大数据 | AllReduce的`D > 4GB`；ReduceScatter/AllGather的`T > 4GB` | `SequenceMeshConcurNHR` |
| 三层对称拓扑 | Level0、Level1对称，且没有更高优先级的UBoE Pipeline条件 | `SequenceMeshConcurNHRNHR` |

Sequence的目标不是换一种基础拓扑算法，而是把多个通信阶段串行编排，降低一次性资源压力，并处理超大数据或更多通信层。

### 5.5 Pipeline

Pipeline最常见的触发来源是“大数据+不同层链路可以重叠”。

#### PCIe混合拓扑

前置条件：`level0Topo == MESH_1D_CLOS`、`level0PcieMix == true`，并且Mesh不能覆盖全部rank。

| 算子 | 小数据 | 达到流水阈值后 |
| --- | --- | --- |
| AllReduce | `D < 32MB`使用Parallel | `D >= 32MB`使用Pipeline |
| ReduceScatter | `D < 4MB`使用Parallel | `D >= 4MB`使用Pipeline |
| AllGather | `D < 4MB`使用Parallel | `D >= 4MB`使用Pipeline |

#### UBX或UBoE拓扑

- UBX矩形拓扑中，CLOS组数是Mesh组数的整数倍，并且数据不小时，可能选择Pipeline；部分归约算法还要求支持对称内存。
- 多层拓扑中，Level0/Level1对称、每模块8卡且顶层为UBoE时，AllReduce、ReduceScatter、AllGather优先选择Pipeline。
- 如果不满足Pipeline能力，通常退回Parallel/RSAG或SoleNHR。

### 5.6 Concur

Concur用于并发使用多个规则通信平面，常见于小规模、对称的UBX拓扑。

| 算子 | 典型触发条件 |
| --- | --- |
| AllReduce | Mesh组数等于CLOS组数、`R <= 4`、普通类型且`D > 8MB` |
| AllGather | Mesh组数等于CLOS组数、`R <= 4`且`D > 512KB` |
| AlltoAll | Mesh组数等于CLOS组数、`R <= 4`且`sendCounts[0] > 512` |
| AlltoAllV | Mesh组数等于CLOS组数且`R <= 4` |

`SoleMeshConcur`中的Concur只是Mesh内部实现变体；只有`ConcurMeshNHR`这类名称才表示主编排方式为Concur。

### 5.7 StrictOrdered和OrderPreserved

只在AllReduce和ReduceScatter的strict保序场景中优先触发：

| rank数 | 结果 |
| --- | --- |
| `R <= 8` | StrictOrdered Mesh |
| `R > 8` | OrderPreserved Group |

保序分支位于普通拓扑和数据量分支之前，因此一旦命中，普通Mesh/NHR/Pipeline性能规则不再参与选择。

### 5.8 NHRAicpuReduce

常见触发输入：

- 数据类型为INT64、UINT64或FP64。
- 归约操作为PROD。
- 通信需要跨越CLOS、多层拓扑或Mesh无法全连接的混合拓扑。

单层全Mesh中，AllReduce和ReduceScatter仍可以选择Mesh类实现；只有进入非全连接或多层通信域时，特殊归约才明显倾向`NHRAicpuReduce`。

## 6. 常见后缀是什么意思

这些后缀通常不会改变算法的大类，只表示同一算法族的实现变体：

| 后缀 | 常见触发条件 |
| --- | --- |
| `OneShot` | AllReduce全Mesh小数据，AI_CPU路径常见边界为`D <= 8MB` |
| `TwoShot` | AllReduce/Reduce的中大数据Mesh场景 |
| `Chunk` | 数据较大，需要分块避免一次性资源压力 |
| `MultiLink` | 纯CLOS或存在多条可用链路 |
| `SingleChannel` | AlltoAll中`sendCounts[0] × sizeof(sendType) × R <= 150MB`时减少通道开销 |
| `PCIe` | Mesh不能全连接，需要跨PCIe链路 |
| `UBX` | Mesh+CLOS的UBX链路拓扑 |
| `Uboe` | 顶层使用UBoE网络 |

## 7. 从算法名反查触发原因

| 日志中的关键词 | 优先检查的输入条件 |
| --- | --- |
| `SoleMesh` | 是否单层、是否全Mesh连通 |
| `SoleNHR` | 是否CLOS、`Level1Nhr`、本地组退化、小数据或拓扑不对称 |
| `ParallelMeshNHR` | 是否两层Mesh，数据是否刚超过拆层阈值 |
| `Sequence...NHR` | 数据是否超过4GB边界，或者是否为三层对称拓扑 |
| `PipeLinePcie` | 是否PCIe混合且Mesh不能全连通，数据是否达到4MB/32MB边界 |
| `PipeLineUBX` | 是否规则UBX矩形拓扑、数据是否足够大、是否支持对称内存 |
| `...NHRNHR` | 是否三层对称拓扑 |
| `StrictOrdered`或`OrderPreserved` | 是否开启strict确定性 |
| `NHRAicpuReduce` | 是否为64位归约或PROD |

建议同时查看日志字段：`topoLevelNums`、`level0Topo`、`Level1Nhr`、`level0Symmetric`、`level1Symmetric`、`topLevelUboe`、`dataSize`和`userRankSize`。

## 8. 例外和限制

1. `HCCL_ALGO`配置的是通信层算法，不是直接指定上述完整Executor名称。
2. strict保序、特殊数据类型、产品能力和不支持的数据类型会覆盖普通性能规则。
3. CCU或AIV资源不足后可以回退AI_CPU，所以AI_CPU算法不一定来自用户显式配置。
4. 阈值是提交`7696133a`的实现参数，不是稳定接口。
5. `HCCL_USE_NEW_SELECTOR=1`时，AllReduce、ReduceScatter、AllGather改为“候选过滤+成本模型+最小cost”选择，本文的固定阈值表不再是完整决策过程。
6. 新Selector之外的Broadcast、Reduce、Scatter、AlltoAll等仍使用旧Selector规则。

## 9. 主要源码入口

- `src/ops/all_reduce/selector/all_reduce_auto_selector.cc`
- `src/ops/reduce_scatter/selector/reduce_scatter_auto_selector.cc`
- `src/ops/all_gather/selector/all_gather_auto_selector.cc`
- `src/ops/broadcast/selector/broadcast_auto_selector.cc`
- `src/ops/reduce/selector/reduce_auto_selector.cc`
- `src/ops/scatter/selector/scatter_auto_selector.cc`
- `src/ops/all_to_all_v/selector/alltoall_auto_selector.cc`
- `src/ops/all_to_all_v/selector/alltoallv_auto_selector.cc`
- `src/ops/op_common/selector/auto_selector_base.cc`
- `src/ops/op_common/topo/topo_host.cc::CalcLevel1Nhr`
- `src/common/order_preserved_common.h`
