# 数据流与接口文档 (Data Pipeline Documentation)

本文档详细描述了双生 RL Agent 项目中各个模块之间的数据交互流程、关键函数接口、传入传出数据的类型及形状。

## 1. Environment (env.py)

环境负责游戏逻辑，是数据流的起点。

### `TractorEnv.step` / `reset`
*   **输入 (Input)**: `response` (仅 `step` 需要)
    *   **类型**: `dict`
    *   **结构**: `{'player': int, 'action': list[int]}`
    *   **说明**: `action` 是卡牌的 ID 列表（0-107）。
*   **输出 (Output)**: `obs`, `action_options`, `reward`, `done`
    *   **`obs` (Observation)**:
        *   **类型**: `dict`
        *   **内容**:
            *   `'id'`: `int` (当前玩家 ID 0-3)
            *   `'deck'`: `list[str]` (手牌，如 `['sA', 'h2']`)
            *   `'history'`: `list[list[str]]` (本轮出牌记录)
            *   `'major'`: `list[str]` (当前主牌集合)
            *   `'played'`: `list[list[str]]` (已打出的牌)
    *   **`action_options`**:
        *   **类型**: `list[list[str]]`
        *   **说明**: 当前玩家合法的所有出牌组合。例如 `[['sA'], ['sK', 'sK']]`。
    *   **流向**: 数据返回给 `Actor`。

## 2. Feature Wrapper (wrapper.py)

负责将环境的原始数据转化为神经网络可接受的 Tensor/Matrix。

### `cardWrapper.obsWrap`
*   **调用位置**: `actor.py` -> `Actor.run`
*   **输入 (Input)**:
    *   `obs`: `dict` (来自 `env.step/reset`)
    *   `options`: `list[list[str]]` (来自 `env.step/reset`)
*   **输出 (Output)**: `obs_mat`, `action_mask`
    *   **`obs_mat`**:
        *   **类型**: `np.ndarray`
        *   **形状**: `(128, 4, 14)`
        *   **含义**: 128 个特征通道 (Major, Deck, History, Played, Options)。
    *   **`action_mask`**:
        *   **类型**: `np.ndarray`
        *   **形状**: `(54,)`
        *   **含义**: 1 表示对应索引的动作有效，0 表示无效。
    *   **流向**: 存入 `episode_data`，并转换为 Tensor 喂给 `model`。

## 3. Actor (actor.py)

负责数据采集、模型推理和轨迹存储。

### `Actor.run` (Inference Loop)
*   **输入 (Model Input)**: `state`
    *   **类型**: `dict` of `torch.Tensor`
    *   **形状**:
        *   `'observation'`: `(1, 128, 4, 14)` (Batch size = 1)
        *   `'action_mask'`: `(1, 54)`
*   **输出 (Model Output)**: `logits`, `value`
    *   **形状**: `(1, 54)`, `(1, 1)`
    *   **流向**:
        *   `logits` -> 用于采样动作索引 `action` (int)。
        *   `value` -> 存入 buffer 用于计算 Advantage。

### `env.action_intpt`
*   **输入**: `action` (索引 int), `player` (int)
*   **数据源**: `action_options[action]` (str 列表)
*   **输出**: `response` (dict, 见 Env 章节)
*   **流向**: 传入 `env.step`。

### Data Collection & Post-processing
*   **数据**: `episode_data`
    *   **结构**: `dict` (按 agent 分类) -> `dict` (state, action, reward...)
*   **GAE 计算**:
    *   计算 `td_target` 和 `advantages`。
*   **Push to Buffer**:
    *   **调用**: `self.replay_buffer.push(sample)`
    *   **Sample 结构**:
        *   `'state'`: `{'observation': (T, 128, 4, 14), 'action_mask': (T, 54)}` (T 为 episode 长度)
        *   `'action'`: `(T,)` (`np.int64`)
        *   `'adv'`: `(T,)` (`np.float32`)
        *   `'target'`: `(T,)` (`np.float32`)

## 4. Replay Buffer (replay_buffer.py)

负责存储和Batch采样。

### `ReplayBuffer.push`
*   **输入**: `samples` (见 Actor Push 结构)
*   **内部处理 (`_unpack`)**:
    *   将 **Dict of Arrays** (Structure of Arrays) 解包为 **List of Dicts** (Array of Structures)。
    *   存入 `self.buffer` (Deque)。

### `ReplayBuffer.sample`
*   **调用位置**: `learner.py` -> `Learner.run`
*   **输入**: `batch_size` (int)
*   **输出**: `batch`
    *   **内部处理 (`_pack`)**: 将采样出的 **List of Dicts** 重新堆叠为 **Dict of Arrays**。
    *   **形状**:
        *   `state['observation']`: `(B, 128, 4, 14)`
        *   `state['action_mask']`: `(B, 54)`
        *   `action`: `(B,)`
        *   `adv`: `(B,)`
        *   `target`: `(B,)`
    *   **流向**: 返回给 `Learner`。

## 5. Learner (learner.py)

负责模型训练。

### `Learner.run` (Training Loop)
*   **输入**: `batch` (来自 Buffer)
*   **Tensor 转换**:
    *   `obs`: `(B, 128, 4, 14)` -> device
    *   `actions`: `(B, 1)` -> unsqueeze 后
*   **Model Forward**:
    *   输入: `states` (`obs`, `mask`)
    *   输出: `logits` `(B, 54)`, `values` `(B, 1)`
*   **Loss Calculation**:
    *   使用 `logits`, `actions`, `advs`, `targets` 计算 PPO Loss。

## 6. Model (model.py)

### `CNNModel.forward`
*   **输入**: `input_dict`
    *   `'observation'`: `(B, 128, 4, 14)`
    *   `'action_mask'`: `(B, 54)`
*   **数据流**:
    1.  `obs` -> `_tower` (Conv Layers) -> `hidden` `(B, 1792)` (32*4*14)
    2.  `hidden` -> `_logits` -> `logits` `(B, 54)`
    3.  `action_mask` -> Log & Clamp -> `inf_mask`
    4.  `logits` + `inf_mask` -> `masked_logits` (无效动作 Logits 变为负无穷)
    5.  `hidden` -> `_value_branch` -> `value` `(B, 1)`
*   **输出**: `masked_logits`, `value`

## 7. Move Generator (mvGen.py)

### `move_generator.gen_*`
*   **调用位置**: `env.py` -> `_get_action_options`
*   **输入**: `deck` (List[str]), `tgt` (List[str], optional)
*   **输出**: `moves` (List[List[str]])
*   **流向**: 作为 `action_options` 返回给 Env，最终进入 Wrapper 编码为 Channel 36-143。

