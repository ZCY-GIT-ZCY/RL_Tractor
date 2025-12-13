# RULE_BASE 行动手册

本文档汇总当前 rule-based 模型在对局中的核心规则，实现可解释的 3 种行动类别以及 2 (优势/劣势) × 3 (早/中/晚期) × 2 (庄/闲) 的 12 维度领牌逻辑。

## 三大行动类别

### 1. 控盘 / 管排 (`control`)
- 触发场景：抢在对手前领先、需要强行接管、或主动领主。代码路径：`_lead_policy` / `_select_trump_lead`、`_select_winning_option`。
- 关键信号：
  - `want_trump=True`（根据手中主牌、身份、优势判断）。
  - `_needs_minimal_control` 为真时尽量用最小成本（例如末家但台面有分）。
  - `force_take_override`、`_should_force_special_takeover`、`_should_force_future_control` 等逻辑强制接管。
- 行动特点：优先选择长度长、强度高的同花结构；若敌方花色已绝门则放宽强度要求；若主牌劣势则避免无谓主拖。

### 2. 跟小牌 (`follow_small`)
- 触发场景：既不需要管排，也没有贴分需求。
- 主要来源：
  - `safe_leads` / `safe_options`（无分且可拆 pair 的安全选项）。
  - `_filter_small_leads` 阻止无意义的小单张；`_prepare_void_shedding_leads` 放行“单张+近绝门”的弃牌。
  - 跟牌阶段使用 `_select_smallest`、`_select_low_risk_filler`，带有“非 5 分优先”逻辑。
- 行动特点：尽力释放无分牌、拆掉危险对子，同时遵守主花色/配对约束。

### 3. 贴分 (`stick`)
- 触发场景：
  - 队友已经赢墩 (`teammate_winning`) 且手里有 5/10/K。
  - 领牌但 `score_state` 领先且 `want_trump=False`，允许主动倒分。
  - 对手当前赢墩但判断其并非绝对最大（70% 随机 “小赌” 行为）。
- 关键策略：贴分前会检查 `_prepare_point_dump_candidates`，优先包含 10/K；若只剩 5 分则启用 70% 概率逻辑。

## 状态轴判定

| 维度 | 判定方法 | 影响点 |
| --- | --- | --- |
| 主牌优势 (`advantage` / `disadvantage`) | `_classify_trump_status`: 手牌主牌数量 >9 视为优势，<9 为劣势，其余为平衡。 | 决定 `want_trump`、`_determine_trump_drag_mode`、是否保守 trump。 |
| 阶段 (`early` / `mid` / `late`) | `_classify_game_stage`: 已知牌张数 ≤20 早期，≤70 中期，其余晚期。 | 控制 pair 领出、分牌优先级、拖主模式、得分倾向。 |
| 身份 (`role`) | `_infer_role` 区分庄家/闲家。 | 决定主攻/防守倾向、是否要保护底牌、是否更愿意抢主。 |

## 12 维领牌策略矩阵
下表描述在不同阶段 + 身份 + 主牌状态下的默认领牌思路。默认假设 `score_state` 中性，如有领先/落后会进一步在 `_contextual_score` 中微调（例如领先更倾向贴分）。

### 早期 (≤20 张牌已见)

#### 庄家 · 优势
- 目的：建立主花色统治，保护底牌。
- 行为：`want_trump=True`，`_determine_trump_drag_mode` 常进入 `assertive`；`_filter_trump_leads_by_mode` 允许 11 以上的强主，禁止双王/级牌过度拖。
- 配合：`_filter_pair_leads_by_stage` 阻止中小对牌领出，优先长主或大副牌 tractors。

#### 庄家 · 劣势
- 目的：稳住牌权，不暴露主。
- 行为：`want_trump=False`（除非 `meta.lead_trump` 强制），`_filter_small_leads` 让出安全副牌；当 suit 只剩 1 张时靠 `_prepare_void_shedding_leads` 尽早绝门为队友创造杀主机会。
- 配合：大主被 `SAFE_LEAD_MIN_RANK` 和 `_should_preserve_big_trump` 保护，避免早期亏本拖主。

#### 闲家 · 优势
- 目的：配合队友快速拖主、寻找得分窗口。
- 行为：若主牌≥阈值，`want_trump=True`；但 `_filter_trump_overkill` 防止盲目双王领出。优先处理非分副牌，避免给庄家见底。
- 配合：若侦测到队友 void，`_contextual_score` 会奖励继续打该花，便于贴分给队友。

#### 闲家 · 劣势
- 目的：拖延、找准贴分时机。
- 行为：`want_trump=False`，优先释放安全副牌；当敌方花色绝门时 `_needs_minimal_control=False`，避免抢主。
- 配合：`_prepare_safe_leads` 会按非分、可拆对子优先；若只剩 1 张某花且无分，则立即用 void-shedding。

### 中期 (21–70 张牌)

#### 庄家 · 优势
- 目的：筛掉敌方剩余主、为收尾铺垫。
- 行为：`drag_mode` 可能转为 `free`；允许更长的主拖，但 `_should_preserve_big_trump` 仍保护顶牌。若 `score_state leading`，在 `safe_leads` 耗尽后允许点数贴分。
- 配合：`_filter_pair_leads_by_stage` 逐渐放宽，允许 10 以上对牌领出。

#### 庄家 · 劣势
- 目的：找翻盘点，谨慎用主。
- 行为：`want_trump` 仍多为 False；不过若对手拉开分差，`_should_force_special_takeover` 只在发现特殊牌型（双王、AA tractor）时才接管。
- 配合：`_apply_trump_takeover_filters` 只有在台面>10 分或我方落后情况下才允许砸主救分。

#### 闲家 · 优势
- 目的：围剿庄家、扩大得分。
- 行为：优先打敌方 void 花色，`_contextual_score` 在敌 void + 含分的情况下直接加分，促使贴 K/10；适度打主逼庄家交顶。
- 配合：若队友 dump 过分，`teammate_flags.dumped_points` 提升继续贴分权重。

#### 闲家 · 劣势
- 目的：保存 trump，伺机让队友切。
- 行为：`_prepare_safe_leads` 居首，`void_leads` 仅在 trump ≤4 时触发，以免暴露短处；`_should_stick_against_opponent` 只有在判断对手非绝对最大且手里有 5 时才冒险。
- 配合：若 `teammate_void` 已知，`_teammate_can_cut` 激活“贴分给队友砍”逻辑。

### 晚期 (>70 张牌)

#### 庄家 · 优势
- 目的：收官控分。
- 行为：`_contextual_score` 对含分选项加 10；`_should_stick_on_lead` 常为真，即便不推主也会主动贴分锁分差。
- 配合：`_filter_pair_leads_by_stage` 全面放开，对牌、tractor 皆可用来扫尾。

#### 庄家 · 劣势
- 目的：孤注一掷抢回分。
- 行为：若 `points_on_table=0` 且 `score_state trailing`，`_should_force_future_control` 会要求提前控牌，为尾盘强杀做准备；但 `_needs_minimal_control` 在敌方 void 情况下会关闭以避免白送。
- 配合：`_select_winning_option` 在 `minimal=True` 时寻找 “够用即可” 的 trump 以节省资源。

#### 闲家 · 优势
- 目的：安全落袋。
- 行为：若 `want_trump=False`，先执行 void-shedding / safe lead，再贴分；若需要控盘，则优先使用剩余中等主（因为 `drag_mode` 多为 `light`）。
- 配合：`_should_stick_with_teammate` 在晚期只要队友赢墩几乎必定贴分。

#### 闲家 · 劣势
- 目的：最大化得分产出，寻找最后的砸主点。
- 行为：当敌方花色绝门且我方 trump 仍有厚度时，`_needs_minimal_control=False`，允许直接控盘砸主；若 trump 稀少则依赖 void-shedding + 贴 5 分。
- 配合：`_should_take_on_strength` 已被阶段逻辑限制，晚期除非卓越大牌不会再贸然接管，以免白送分。

## 额外守则
- **Pair 领出限制**：`_filter_pair_leads_by_stage` 视阶段和对子点数控制低点 pair 的出现；早期 10 以下对牌禁领，中期 5 以下需已见到对应得分牌才放行。
- **主拖模式**：`_determine_trump_drag_mode` 根据剩余主牌估算（见 `TRUMP_TOTAL_ESTIMATE`、`trump_drag_rounds`、`trump_half_spent` 等）动态切换 `assertive` / `light` / `free`，并在 `_filter_trump_leads_by_mode` 内过滤掉不符合节奏的主牌。
- **无主标记**：当手牌无主 (`no_trump_mark`) 时直接禁止主动拖主，优先走副牌或 void-shedding。
- **分牌优先级**：在领先或队友已 dump 的情况下，含分选项在 `_contextual_score` 会得到额外奖励，使 `stick` 更容易触发，反之若落后则权重下降。

以上规则共同构成 rule-based 引擎的决策骨架，可据此快速理解或调整不同阶段、身份与主牌状态下的出牌倾向。
