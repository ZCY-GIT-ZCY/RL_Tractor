# Agent 模型与架构文档

本文档详细介绍了本项目中使用的强化学习 Agent 的模型架构、观测空间设计及动作选择机制。

## 1. 总体架构 (Architecture)

本项目采用**分布式 Actor-Learner 架构**（类似 IMPALA），支持自我对弈（Self-Play）训练。

*   **Learner (学习者)**: 单个进程，负责维护全局最新的模型参数，从 Replay Buffer 中采样数据，计算梯度并更新模型。
*   **Actor (演员)**: 多个并行进程（默认 4 个），每个 Actor 负责与环境交互。Actor 拉取最新的模型参数，生成对局数据，并将轨迹（Trajectory）推送到全局 Replay Buffer。
*   **Self-Play**: 在训练过程中，4 个玩家位置（Player 0-3）均使用同一个神经网络模型进行决策，从而实现自我进化。

## 2. 模型网络 (Neural Network)

模型定义在 `model.py` 中的 `CNNModel` 类。这是一个共享权重的卷积神经网络，同时输出策略（Policy）和价值（Value）。

### 网络结构
1.  **特征提取 (Backbone)**:
    *   Input: `[Batch, 128, 4, 14]` (详见下文观测空间)
    *   Conv2d: 128 -> 256 (3x3 kernel) + ReLU
    *   Conv2d: 256 -> 256 (3x3 kernel) + ReLU
    *   Conv2d: 256 -> 32  (3x3 kernel) + ReLU
    *   Flatten

2.  **策略分支 (Policy Head)**:
    *   Linear: `32 * 4 * 14` -> 256 + ReLU
    *   Linear: 256 -> **54** (输出 Logits)

3.  **价值分支 (Value Head)**:
    *   Linear: `32 * 4 * 14` -> 256 + ReLU
    *   Linear: 256 -> **1** (输出状态价值 V)

## 3. 观测空间 (Observation Space)

输入张量的形状为 `(128, 4, 14)`。这个设计将卡牌信息编码为类似图像的矩阵。
*   **4**: 代表花色 (Suit) - Spades, Hearts, Clubs, Diamonds。
*   **14**: 代表点数 (Rank) - 2, 3, ..., K, A, 大小王 (映射到特殊位置)。

**128 个通道 (Channels) 的构成**:
由 `wrapper.py` 中的 `obsWrap` 函数生成：

| 通道索引 | 含义 | 说明 |
| :--- | :--- | :--- |
| **0-1** | **主牌信息 (Major)** | 编码当前的主花色和级牌。使用 2 个通道是因为两副牌可能存在重复。 |
| **2-3** | **手牌 (Deck)** | Agent 当前持有的手牌。 |
| **4-19** | **历史出牌 (History)** | 过去一轮中其他玩家出的牌。共 4 个玩家，每人 2 个通道，共 8 个通道。这里分配了 8*2=16 个通道位置（代码中 `hist_mat` 大小为 8，实际只用前 4 个 move）。 |
| **20-35** | **已出牌 (Played)** | 记录之前的轮次中各家已经打出的牌。 |
| **36-143** | **动作选项 (Options)** | **核心设计**。环境生成的合法动作（单张、对子、拖拉机）被直接编码到输入中。系统预留了 108 个通道，每个动作占用 2 个通道，最多支持 **54** 个候选动作。 |

> **注意**: 输入的总通道数在 `wrapper.py` 中拼接后实际为 `2 + 2 + 8 + 8 + 108 = 128`。

## 4. 动作空间 (Action Space)

Agent 并不直接输出具体的卡牌（如 "出黑桃A"），而是采用**动作选择 (Action Selection)** 机制。

1.  **合法动作生成**: 环境（`env.py` 调用 `mvGen.py`）会根据当前手牌和规则，生成所有合法的出牌组合（Options），例如单张、对子、连对等。
2.  **动作编码**: 这些 Option 被编码进上述的 Observation 的最后 108 个通道中。
3.  **模型输出**: 模型输出大小为 **54** 的向量。
4.  **Masking**: 使用 `action_mask` 屏蔽掉无效的索引（即如果只有 10 个合法动作，则索引 10-53 被 Mask 掉）。
5.  **决策**: Agent 从有效的 logits 中采样一个索引 `i`。
6.  **执行**: 环境根据索引 `i` 查表找到对应的真实卡牌组合并执行。

## 5. 训练算法

*   **算法**: Actor-Critic 框架下的 PPO/IMPALA 变体。
*   **优势函数**: 使用 **GAE (Generalized Advantage Estimation)** 计算优势 (Advantage)。
*   **损失函数**:
    *   Policy Loss: 策略梯度。
    *   Value Loss: MSE 预测误差。
    *   Entropy Loss: 增加探索性。
*   **优化器**: 未在代码片段中显式展示，通常使用 Adam。

