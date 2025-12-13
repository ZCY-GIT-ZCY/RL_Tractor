# RULE_BASE 行动手册（2025 重构）

本版行动手册描述全新的庄/闲拆分策略：

- **阶段性 restrict**：早/中/晚期直接作用于行动集，先剪枝再决策；
- **局面估值**：通过手牌、得分、阶段以及特殊牌型 bonus 给出优势/平衡/劣势标签；
- **领牌双分叉**：庄/闲分别在“优势/劣势”与“主/副牌偏好”两个维度上选择策略；
- **跟牌三大行动类别**：管牌、贴分、跟小牌，由墩上归属 + 座位索引驱动，并附带微观过滤器确保合法、经济。

## 阶段性 restrict 与先验约束

| 阶段 | 约束要点 | 目标 |
| --- | --- | --- |
| 早期 (`cards_seen ≤ 20`) | 禁止 Joker 与级牌对子领出；副牌需保持原顺位。 | 隐藏杀手锏、避免早暴露底牌。 |
| 中期 (`20 < cards_seen ≤ 70`) | `_bias_off_suit_release` 将短门副牌排在高优先级；适度放开对子。 | 逐步释放副牌、为后期制造 void。 |
| 晚期 (`cards_seen > 70`) | 允许大部分组合，但 `_apply_micro_lead_filters` 自动保护剩余顶主。 | 进入收官，控制节奏并预留最后护主。 |

## 局面估值 (`_compute_position_valuation`)

1. **默认勇敢**：基础值 `1.0` + 早期额外 `+0.2`，确保开局无论庄家或闲家都被视为“优势”，敢于主动出牌。
2. **低值才触发保护**：当综合得分低于 `0.85` 时才落入“劣势”，才会启动保守策略；高于 `1.15` 认定为“优势”。
3. **信息来源**：
   - **手牌强度**：截取前 8 张按 `_card_strength` 取均值，并考虑主牌密度（庄家权重更高）。
   - **得分 + 阶段**：领先/落后状态、`score_margin`、以及“晚期仍低于 40 分”的惩罚；早期若已拿到 >20 分还会加成，符合“前期 >20 视为小优势”。
   - **特殊牌型 bonus**：双王、等级对等 premium 组合会额外 +0.1~0.2，确保拥有杀手锏时依旧视为优势。
4. **计划偏好 (`plan_bias`)**：根据 trump 密度、最短副牌长度与阶段推导出 `trump / offsuit / balanced`，驱动领牌第二层分叉。

## 领牌策略（庄上 / 闲家）

决策顺序：先看估值 `state ∈ {advantage, balanced, disadvantage}`，再看 `plan_bias ∈ {trump, offsuit, balanced}`，最终在 bucket 序列里选第一套可行动集。

### 庄家

- **优势 + 主牌偏好**：`[trump_control → points → void → safe → off_balance → minimal_trump]`。早期多张主直接控盘，若仍有分则切入贴分 bucket。
- **优势 + 副牌偏好**：`[void → safe → points → off_balance → trump_control → minimal_trump]`。主牌暂存，优先甩单张短门制造 void。
- **劣势**：`[safe → points → off_balance → minimal_trump → void → trump_control]`，强调保身，只有当安全牌耗尽才用最小主接牌。

### 闲家

- **优势 + 主牌偏好**：`[trump_control → off_balance → points → safe → void → minimal_trump]`，必要时协助队友拉主。
- **优势 + 副牌/平衡**：`[off_balance → void → points → safe → trump_control → minimal_trump]`，靠副牌取得信息后再视情况砸主。
- **劣势**：`[safe → void → off_balance → minimal_trump → points → trump_control]`，尽量让队友来收，自己保存 trump。

### 微观领牌过滤
- 主 bucket (`trump_control`) 在晚期会自动丢弃大于 `BIG_TRUMP_STRENGTH` 的牌，除非没有其它 trump。
- `points` bucket 仅允许单张 `10 > K > 5` 的顺序贴分。
- `void/safe` bucket 会剔除带分牌，真正做到“副牌甩牌”优先。

## 跟牌策略：三大行动类别

### 1. 管牌（Control）
- **触发**：队友未赢墩且存在可超越的组合。若台面有分则模式为“拿分式”，否则为“上台式”。
- **座位逻辑**：
  - **2 号位**（第二家）：除非台面分数 ≥ 阈值，否则会过滤掉最顶级 trump，优先保存实力。
  - **3 号位**：被要求“不遗余力”，候选集按强度降序排序。
  - **4 号位**：只取“能刚好压住”的最小牌，避免多余损耗。
- **实现**：`_filter_control_candidates` + `_select_winning_option` 组合完成。

### 2. 贴分（Stick）
- **确定性贴分**：队友已赢墩时启用，严格执行 `10 > K > 5`，且只允许单张，确保不拆对子/tractor。
- **试探式贴分**：仅 2 号位、且队友在自己之后时生效，只能给出单张 5 分并寄望队友回收。

### 3. 跟小牌（Follow Small）
- **宏观滤器**：`safe_fillers → fillers`，优先无分单张、其次考虑近绝门的副牌。
- **微观滤器**：通过 `MICRO_VALUE_ORDER (K > A > 10 > Q > J > 5 > 9 > 8 > …)` 控制相对大小；若场面已有 9，则 5 与 8 统统视为“过大”而被过滤，符合“5>8>7 不合法”的例子。
- **绝门式贴牌**：当副牌仅剩 1 张且无分时，优先将其送出，为后手创造切牌机会。

## 跟牌流程汇总

1. `_build_follow_buckets` 划分 `control / stick_sure / stick_probe / small`。所有候选集合都先经过阶段性 restrict、对牌保护等宏观层。
2. `_choose_follow_plan`：
   - 若队友赢墩 → `stick_sure`；
   - 若敌方赢且可赢 → `control`（points 或 stage）；
   - 若自己为 2 号位且队友在后 → `stick_probe`；
   - 否则默认 `small`。
3. `_apply_micro_follow_filters` 套用座位逻辑、贴分排序及相对价值过滤，最后通过 `_select_follow_option` 输出。

## Peer-Review 辅助信息

模型在 `_prepare_context` 阶段会“同行评审”手牌信息，推导多种信号：

- **void_map**：来自环境枚举的缺门集合，被拆成 `teammate_void_suits` 与 `enemy_void_suits`；
- **teammate_flags**：记录队友是否跟到首牌、是否已经贴分、是否通过切牌暴露 trump；
- **opponent_flags**：观察敌方是否丢分，以及“没有再见过该花色分牌”这类保守信号；
- **point_visibility**：同花色的 10/5 是否已经摊牌，用于判断是否允许提前拆对子。

这些信号会直接反馈到行动层：

- **领牌防呆**：`_filter_peer_incompatible_leads` 会在微观过滤之后再次审查 action。凡是非主牌且花色属于 `teammate_void_suits` 的选项一律丢弃；若花色落在 `enemy_void_suits`，则只有“可控 A 级牌”或完全不带分的弃牌才保留，`points` bucket 则直接禁止避免“给敌人送分”。
- **贴分刹车**：`stick_sure / stick_probe` 现在会检测 `enemy_void_suits`。一旦确认敌方缺门，哪怕满足贴分条件也会清空候选集，转而走 `control / small`，防止队友还没收分就被敌方切牌。
- **继承性信号**：`teammate_flags` 中的 `void_suit`、`dumped_points` 会调整 `_contextual_score`，让后续轮次自觉规避同一花色的分牌输出。

## 其他注意事项

- **文档中的 bucket 名称** 与代码保持一致，便于排查：`trump_control / off_balance / void / safe / points / minimal_trump` 与 `control / stick_* / small`。
- **估值信息透明**：`context["valuation"] = {score, state, plan_bias}` 可直接记录在日志，用于解释为何选择某一分支。
- **高牌保留**：只有在晚期且 action bucket 被耗尽时，才允许动用剩余王与级牌，防止“底牌被掏空”的极端情况。

以上框架保障了“庄上策略先行、庄闲分治、阶段先验→行动类别→微观 filter”这一完整决策链，也为后续扩展（如农民特化、记牌反馈）留出了钩子点。
