# 本地 Tractor 评测（摸牌阶段）

本仓库的 `env.py` 已支持在 `STAGE_SNATCH` 阶段逐张发牌（共 100 张），并在每次摸牌后给出“报主/反主/不报”的合法动作列表。

`local_simulation.py` 在此基础上提供一个**不依赖 botzone** 的本地模拟脚本，并按“事件流”在命令行输出。

## 运行

在项目根目录（`Tractor_Project`）下：

```powershell
# 随机庄位，固定级牌=2，固定随机种子
& "C:\Users\Hong Weijun\AppData\Local\Programs\Python\Python312\python.exe" .\local_simulation.py --seed 1 --banker random --level 2

# 指定庄位=0，级牌=J
& "C:\Users\Hong Weijun\AppData\Local\Programs\Python\Python312\python.exe" .\local_simulation.py --banker 0 --level J
```

## 输出格式（每轮）

- 轮次分隔符：`========== 摸牌阶段 Round i/100 ==========`
- 事件（事件间空行）：
  - `摸牌\tget=<card>`
  - `报主\tdeclare=[...]` 或 `反主\tsnatch=[...]` 或 `不报\tpass`
  - 每条事件后打印该玩家当前手牌：`P{seat} HAND (n): [...]`
- 轮末统一打印四人手牌：`[本轮截止] 四人手牌： ...`

说明：`env.step()` 在 `STAGE_SNATCH` 会在处理当前玩家动作后立即给下家发下一张牌；脚本会在 step 前截取手牌快照，避免“本轮截止手牌”把下一轮摸牌提前算进去。

## 更改四个 player / model 类型（接口预留）

`local_simulation.py` 顶部有 `DEFAULT_CONFIG`，其中 `players` 控制四个座位使用的策略：

- `type: "heuristic_snatch"`：默认，复用 `declaration.py` 的 `decide_declaration/decide_overcall`
- `type: "always_pass"`：永远不报/不反
- `type: "random"`：随机从合法动作中选一个

你也可以在某个 seat 上提供 `factory`（可调用对象），脚本会用它来构建你打包好的策略实例：

```python
# 伪代码：把 DEFAULT_CONFIG['players'][0] 改成：
{"factory": lambda seat, rng, config: MyPackedPolicy(...)}
```

当前脚本只跑摸牌阶段；如果你希望把后续“扣底/出牌”也接上，我可以按你现有的 `Actor` 结构继续扩展。
