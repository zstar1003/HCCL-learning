# HCCL AI_CPU通信方案选择：一页PPT版

## PPT标题

**HCCL先由约束和拓扑确定通信路径，再由数据规模决定Executor**

## 左侧主图（约70%宽度）

```mermaid
flowchart LR
    A["输入<br/>Op、拓扑、S、R、类型、约束"] --> H{"高优先级约束"}
    H -->|strict| O["Ordered｜Mesh或<br/>AlltoAll+NHR"]
    H -->|特殊归约| SR["Sole/Sequence｜<br/>NHR+AICPU Reduce"]
    H -->|普通| L{"拓扑层数"}

    L -->|单层| L1{"Level0形态"}
    L1 -->|全Mesh| SM["Sole｜Mesh"]
    L1 -->|CLOS| SN["Sole｜NHR"]
    L1 -->|Mesh+CLOS| MX{"连通性与规则性"}
    MX -->|Mesh全连| SM
    MX -->|规则、小rank| CO["Concur｜Mesh+NHR"]
    MX -->|非全连、数据不大| PA["Parallel｜Mesh+NHR"]
    MX -->|大数据且可流水| PI["Pipeline｜Mesh+NHR"]
    MX -->|关系不规则| SN

    L -->|两层| L2{"层次退化或本地组为1?"}
    L2 -->|是| SN
    L2 -->|否| L2T{"Level0形态"}
    L2T -->|CLOS| SN
    L2T -->|规则Mesh| DS{"按Op判断数据档位"}
    DS -->|小| SN
    DS -->|中| PA
    DS -->|超大| SE["Sequence｜Mesh+NHR"]

    L -->|三层| L3{"Level0/1对称?"}
    L3 -->|否| SN
    L3 -->|是| UB{"顶层UBoE且能力满足?"}
    UB -->|是| UP["Pipeline/UBoE Parallel｜<br/>多层Mesh/NHR"]
    UB -->|否| S3["Sequence｜Mesh+NHR+NHR"]
```

## 右侧图例（约30%宽度）

### 输入和判断字段

- `Op/S/R`：算子、单rank数据量和rank数；部分分支使用`S×R`。
- `全Mesh/CLOS`：本层全部rank直连/经交换网络通信。
- `特殊域`：归约类Op的64位或PROD进入多层、CLOS或非全Mesh通信域。
- `退化`：Level1Nhr、Level0Nhr或本地组为1，无法建立有效层次。
- `规则混合`：Mesh/CLOS分组关系稳定，可以并发利用两个通信平面。
- `UBoE`：顶层专用网络，能力满足时可进入专用并行或流水路径。

### 输出格式

`Executor｜Template族`

- `Sole`单路；`Parallel`阶段并行；`Sequence`阶段串行。
- `Pipeline`切块重叠；`Concur`多平面并发；`Ordered`确定性优先。

## 底部阈值条

| 普通两层组网 | 小→中 | 中→超大 |
| --- | --- | --- |
| AllReduce | `S=32MB` | `S=4GB` |
| ReduceScatter | `S=4MB` | `S×R=4GB` |
| AllGather | `S=1MB` | `S×R=4GB` |

PCIe混合且Mesh不全连时：AllReduce达到`32MB`、ReduceScatter/AllGather达到`4MB`后转Pipeline。

页脚：`适用范围：提交7696133a、AI_CPU、AllReduce/ReduceScatter/AllGather常见路径；特殊产品能力、strict、资源回退和新Selector成本择优优先。`

## 演讲备注（不放在PPT正文）

1. 图的输出不是一个“算法名”，而是`Executor｜Template族`组合。
2. strict、64位类型和PROD属于高优先级覆盖条件，先于普通拓扑性能规则。
3. 单层主要看Level0是全Mesh、CLOS还是混合拓扑。
4. 两层规则Mesh才进入小/中/超大数据分档：小数据压平为Sole NHR，中等数据并行Mesh+NHR，超大数据切Sequence。
5. 三层先看对称性；不对称回退NHR，对称后再看顶层UBoE能力。
6. Template细节只需在答疑时说明：小数据常用OneShot，大数据可能使用Chunk，多端口使用MultiLink。
