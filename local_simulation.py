"""local_simulation.py

本地 Tractor(拖拉机)评测/模拟器（不依赖 botzone）。

目标：模拟完整对局流程（摸牌/报主/反主 -> 扣底 -> 出牌直到结束），并在命令行打印事件流。

输出格式（每轮）：
- 轮次分隔符
- 事件记录（摸牌/报主/反主/不报）：
  事件名\t事件内容\n事件所属玩家的手牌情况
  事件与事件之间空行
- 无论本轮内有几个事件，最后输出本轮截止后四个人手牌情况

你可以通过修改 DEFAULT_CONFIG 里的 players 配置，替换四个 player 的“决策器”。
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from env import TractorEnv
from declaration import decide_declaration, decide_overcall

try:
    from kitty import select_kitty_cards
except Exception:
    select_kitty_cards = None  # type: ignore

try:
    import torch

    from wrapper import cardWrapper
    from rule_based_model import RuleBasedModel

    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    cardWrapper = None  # type: ignore
    RuleBasedModel = None  # type: ignore
    _TORCH_OK = False


LEVELS: List[str] = ["2", "3", "4", "5", "6", "7", "8", "9", "0", "J", "Q", "K", "A"]


class PlayerPolicy(Protocol):
    def select_action(self, env: TractorEnv, obs: Dict[str, Any], action_options: List[List[str]]) -> int:
        """Return action index in action_options."""


@runtime_checkable
class BuryPolicy(Protocol):
    def select_bury_action(self, env: TractorEnv, obs: Dict[str, Any], action_options: List[List[str]]) -> int:
        """Return action index in action_options (BURY stage)."""


@runtime_checkable
class PlayPolicy(Protocol):
    def select_play_action(self, env: TractorEnv, obs: Dict[str, Any], action_options: List[List[str]]) -> int:
        """Return action index in action_options (PLAY stage)."""


def _find_option_index(action_options: List[List[str]], target_cards: List[str]) -> Optional[int]:
    for idx, option in enumerate(action_options or []):
        if option == target_cards:
            return idx
    return None


class AlwaysPassPolicy:
    def select_action(self, env: TractorEnv, obs: Dict[str, Any], action_options: List[List[str]]) -> int:
        return 0


class RandomPolicy:
    def __init__(self, rng: random.Random):
        self._rng = rng

    def select_action(self, env: TractorEnv, obs: Dict[str, Any], action_options: List[List[str]]) -> int:
        if not action_options:
            return 0
        return self._rng.randrange(0, len(action_options))


class HeuristicSnatchPolicy:
    """复用训练 actor 的报主/反主启发式（decide_declaration/decide_overcall）。"""

    def __init__(self, auto_snatch_on_level: bool = True):
        self.auto_snatch_on_level = auto_snatch_on_level

    def select_action(self, env: TractorEnv, obs: Dict[str, Any], action_options: List[List[str]]) -> int:
        level = env.level
        deck = obs.get("deck", [])

        # 未报主：尝试报主
        if env.reporter is None:
            candidate = decide_declaration(deck, level, force_on_level=self.auto_snatch_on_level)
            if not candidate:
                return 0
            target = [candidate + level]
            idx = _find_option_index(action_options, target)
            return idx if idx is not None else 0

        # 已报主且未反主：尝试反主
        candidate = decide_overcall(deck, level, env.major or "n")
        if not candidate:
            return 0
        if candidate == "n":
            for joker in ("Jo", "jo"):
                idx = _find_option_index(action_options, [joker, joker])
                if idx is not None:
                    return idx
            return 0
        target = [candidate + level, candidate + level]
        idx = _find_option_index(action_options, target)
        return idx if idx is not None else 0


class KittyBuryPolicy:
    """扣底策略：复用 kitty.select_kitty_cards，每次选 1 张（与 env._get_bury_options 对齐）。"""

    def select_bury_action(self, env: TractorEnv, obs: Dict[str, Any], action_options: List[List[str]]) -> int:
        if not action_options:
            return 0
        if select_kitty_cards is None:
            return 0

        banker = env.banker_pos
        if banker is None:
            return 0
        bury_count = int(getattr(env, "bury_left", 0) or 0)
        deck_ids = list(env.player_decks[banker])
        selected = select_kitty_cards(deck_ids, env.level, env.major or "n", bury_count)
        if not selected:
            return 0
        target_name = env._id2name(selected[0])
        idx = _find_option_index(action_options, [target_name])
        return idx if idx is not None else 0


class RuleBasedPlayPolicy:
    """出牌策略：用 rule_based_model.RuleBasedModel 生成 logits，选择 argmax。"""

    def __init__(self):
        if not _TORCH_OK:
            raise RuntimeError("torch/rule_based_model not available")
        self._wrapper = cardWrapper()
        self._policy = RuleBasedModel().eval()

    def select_play_action(self, env: TractorEnv, obs: Dict[str, Any], action_options: List[List[str]]) -> int:
        if not action_options:
            return 0
        obs_mat, action_mask = self._wrapper.obsWrap(obs, action_options)
        obs_tensor = torch.tensor(obs_mat, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self._policy({"observation": obs_tensor, "action_mask": mask_tensor})
        action = int(torch.argmax(logits, dim=-1).item())
        return max(0, min(action, len(action_options) - 1))


@dataclass(frozen=True)
class Event:
    name: str
    content: str
    player: int
    player_hand: List[str]


class CliPrinter:
    def __init__(self, show_sorted_hand: bool = True):
        self._show_sorted_hand = show_sorted_hand

    def print_round_header(self, round_idx: int, total_rounds: int) -> None:
        print(f"========== 摸牌阶段 Round {round_idx:03d}/{total_rounds} ==========")

    def print_bury_header(self, step_idx: int, total_steps: int, banker: int) -> None:
        print(f"========== 扣底阶段 Step {step_idx:02d}/{total_steps} (banker=P{banker}) ==========")

    def print_trick_header(self, trick_idx: int, leader: int) -> None:
        print(f"========== 出牌阶段 Trick {trick_idx:03d} (leader=P{leader}) ==========")

    def print_event(self, event: Event) -> None:
        print(f"{event.name}\t{event.content}")
        print(self._format_player_hand(event.player, event.player_hand))
        print("")

    def print_round_hands(self, hands: List[List[str]]) -> None:
        print("[本轮截止] 四人手牌：")
        for pid, hand in enumerate(hands):
            print(self._format_player_hand(pid, hand))
        print("")

    def _format_player_hand(self, player: int, hand: List[str]) -> str:
        shown = list(hand)
        if self._show_sorted_hand:
            shown = sorted(shown)
        return f"P{player} HAND ({len(hand)}): {shown}"


def _default_level_sequence(rng: random.Random) -> List[str]:
    seq = list(LEVELS)
    rng.shuffle(seq)
    return seq


def build_policies(config: Dict[str, Any], rng: random.Random) -> List[PlayerPolicy]:
    """通过 config 构建四个玩家策略。

    支持：
    - type == "always_pass"
    - type == "random"
    - type == "heuristic_snatch"（默认）

    你也可以直接把某个 seat 的配置写成：{"factory": callable}
    callable 签名建议：factory(seat:int, rng:random.Random, config:dict)->PlayerPolicy
    """

    players_cfg = (config.get("players") or {})
    policies: List[PlayerPolicy] = []
    for seat in range(4):
        seat_cfg = players_cfg.get(seat, players_cfg.get(str(seat), None))
        if seat_cfg is None:
            seat_cfg = {"type": "heuristic_snatch"}

        if isinstance(seat_cfg, dict) and callable(seat_cfg.get("factory")):
            policies.append(seat_cfg["factory"](seat=seat, rng=rng, config=config))
            continue

        ptype = (seat_cfg.get("type") if isinstance(seat_cfg, dict) else None) or "heuristic_snatch"
        if ptype == "always_pass":
            policies.append(AlwaysPassPolicy())
        elif ptype == "random":
            policies.append(RandomPolicy(rng))
        elif ptype == "heuristic_snatch":
            auto = True
            if isinstance(seat_cfg, dict):
                auto = bool(seat_cfg.get("auto_snatch_on_level", True))
            policies.append(HeuristicSnatchPolicy(auto_snatch_on_level=auto))
        else:
            raise ValueError(f"Unknown player policy type: {ptype!r} for seat {seat}")

    return policies


def _build_bury_policy(config: Dict[str, Any]) -> BuryPolicy:
    bury_cfg = config.get("bury") or {}
    btype = (bury_cfg.get("type") if isinstance(bury_cfg, dict) else None) or "kitty"
    if btype == "kitty":
        return KittyBuryPolicy()
    if btype == "pass":
        return KittyBuryPolicy()  # env 不允许真正的 pass，这里退化为 kitty
    raise ValueError(f"Unknown bury policy type: {btype!r}")


def _build_play_policy(config: Dict[str, Any], rng: random.Random) -> PlayPolicy:
    play_cfg = config.get("play") or {}
    ptype = (play_cfg.get("type") if isinstance(play_cfg, dict) else None) or "rule_based"
    if ptype == "rule_based":
        if _TORCH_OK:
            return RuleBasedPlayPolicy()
        return RandomPolicy(rng)  # type: ignore[return-value]
    if ptype == "random":
        return RandomPolicy(rng)  # type: ignore[return-value]
    raise ValueError(f"Unknown play policy type: {ptype!r}")


def _hand_names(env: TractorEnv, player: int) -> List[str]:
    return [env._id2name(cid) for cid in env.player_decks[player]]


def _all_hands(env: TractorEnv) -> List[List[str]]:
    return [_hand_names(env, pid) for pid in range(4)]


class LocalSimulator:
    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        seed = self.config.get("seed")
        self.rng = random.Random(seed)

        # TractorEnv 内部使用的是全局 random；这里同步设种子，保证可复现。
        if seed is not None:
            random.seed(int(seed))
            try:
                import numpy as np  # type: ignore

                np.random.seed(int(seed))
            except Exception:
                pass

        self.printer = CliPrinter(show_sorted_hand=bool(self.config.get("show_sorted_hand", True)))

        # 随机决定“级牌 sequence”；本文件只用到第一张 level
        self.level_sequence = list(self.config.get("level_sequence") or _default_level_sequence(self.rng))
        if not self.level_sequence:
            self.level_sequence = _default_level_sequence(self.rng)

        self.env = TractorEnv({"seed": seed} if seed is not None else {})
        self.policies = build_policies(self.config, self.rng)
        self.bury_policy = _build_bury_policy(self.config)
        self.play_policy = _build_play_policy(self.config, self.rng)

    def run_one_game(self) -> None:
        total_rounds = int(self.config.get("deal_rounds", 100))
        if total_rounds != 100:
            # env 固定 card_todeal=100；这里允许你以后扩展，但当前先强制一致
            raise ValueError("TractorEnv 当前实现固定摸牌 100 轮，请把 deal_rounds 设为 100")

        banker_pos = self._pick_banker_pos()
        level = self.level_sequence[0]

        # 初始 major 设为 'r'（随机）；后续会被报主/反主覆盖
        obs, action_options = self.env.reset(level=level, banker_pos=banker_pos, major="r")

        # 对局信息（初始化后立刻可见的状态）
        self.printer.print_event(
            Event(
                name="对局",
                content=f"start banker=P{getattr(self.env, 'banker_pos', None)} level={getattr(self.env, 'level', None)} major_init={getattr(self.env, 'major', None)}",
                player=int(getattr(self.env, "curr_player", 0)),
                player_hand=_hand_names(self.env, int(getattr(self.env, "curr_player", 0))),
            )
        )

        # ---------- STAGE_SNATCH：摸牌/报主/反主（100轮） ----------
        # Round 1：reset 已经给 curr_player 发了一张牌
        round_idx = 1
        while True:
            stage = obs.get("stage", TractorEnv.STAGE_PLAY)
            if stage != TractorEnv.STAGE_SNATCH:
                break

            player = int(obs["id"])
            dealt_card_name = self._latest_dealt_card_name(player)

            self.printer.print_round_header(round_idx, total_rounds)

            # 事件 1：摸牌
            self.printer.print_event(
                Event(
                    name="摸牌",
                    content=f"get={dealt_card_name}",
                    player=player,
                    player_hand=_hand_names(self.env, player),
                )
            )

            # 事件 2：报主/反主/不报
            action_idx = self.policies[player].select_action(self.env, obs, action_options)
            action_idx = max(0, min(int(action_idx), max(0, len(action_options) - 1)))
            chosen = (action_options[action_idx] if action_options else [])
            evt_name, evt_content = self._format_snatch_event(chosen)
            self.printer.print_event(
                Event(
                    name=evt_name,
                    content=evt_content,
                    player=player,
                    player_hand=_hand_names(self.env, player),
                )
            )

            # 注意：env.step() 在 SNATCH 阶段会“先处理本玩家报主/反主”，
            # 然后立刻给下家发下一张牌。
            # 为了让“本轮截止后四人手牌”不提前包含下一轮摸牌，
            # 这里先抓取快照，再 step，最后打印快照。
            hands_snapshot = _all_hands(self.env)

            # 推进一步（会自动给下家发牌，或在最后一轮进入 BURY 并给庄家发底牌）
            obs, action_options, _, _ = self.env.step({"player": player, "action": action_idx})

            # 本轮截止：四人手牌
            self.printer.print_round_hands(hands_snapshot)

            if round_idx >= total_rounds:
                break
            round_idx += 1

        # ---------- STAGE_BURY：庄家扣底（8步，env 默认一次扣 1 张） ----------
        bury_step = 1
        while True:
            stage = obs.get("stage", TractorEnv.STAGE_PLAY)
            if stage != TractorEnv.STAGE_BURY:
                break
            banker = int(obs["id"])

            # 固定总步数（一般为 8）
            if bury_step == 1:
                bury_total = int(getattr(self.env, "bury_left", 0) or 0)
                bury_total = max(bury_total, 1)
                public_cards = [self.env._id2name(cid) for cid in getattr(self.env, "card_public", [])]
                self.printer.print_event(
                    Event(
                        name="底牌入庄",
                        content=f"public={public_cards} (now banker_hand={len(self.env.player_decks[banker])})",
                        player=banker,
                        player_hand=_hand_names(self.env, banker),
                    )
                )
            self.printer.print_bury_header(bury_step, bury_total, banker)

            action_idx = 0
            if isinstance(self.bury_policy, BuryPolicy):
                action_idx = self.bury_policy.select_bury_action(self.env, obs, action_options)
            action_idx = max(0, min(int(action_idx), max(0, len(action_options) - 1)))
            chosen = (action_options[action_idx] if action_options else [])
            bury_left_before = getattr(self.env, "bury_left", None)

            obs, action_options, _, _ = self.env.step({"player": banker, "action": action_idx})

            self.printer.print_event(
                Event(
                    name="扣底",
                    content=f"bury={list(chosen)} bury_left_before={bury_left_before}",
                    player=banker,
                    player_hand=_hand_names(self.env, banker),
                )
            )
            self.printer.print_round_hands(_all_hands(self.env))
            bury_step += 1

        # ---------- STAGE_PLAY：出牌直到结束 ----------
        trick_idx = 0
        current_trick_leader = int(getattr(self.env, "curr_player", 0))
        while True:
            stage = obs.get("stage", TractorEnv.STAGE_PLAY)
            if stage != TractorEnv.STAGE_PLAY:
                break

            # 新的一墩开始：history 为空或上一墩刚结束（len==4，尚未被 _play 清空）
            if len(getattr(self.env, "history", [])) in (0, 4):
                trick_idx += 1
                current_trick_leader = int(obs["id"])
                self.printer.print_trick_header(trick_idx, current_trick_leader)
                if trick_idx == 1:
                    self.printer.print_event(
                        Event(
                            name="定主",
                            content=f"banker=P{getattr(self.env, 'banker_pos', None)} major={getattr(self.env, 'major', None)} level={getattr(self.env, 'level', None)}",
                            player=current_trick_leader,
                            player_hand=_hand_names(self.env, current_trick_leader),
                        )
                    )

            player = int(obs["id"])
            action_idx = 0
            if isinstance(self.policies[player], PlayPolicy):
                action_idx = self.policies[player].select_play_action(self.env, obs, action_options)  # type: ignore[attr-defined]
            else:
                # 默认：使用全局 play_policy
                action_idx = self.play_policy.select_play_action(self.env, obs, action_options)
            action_idx = max(0, min(int(action_idx), max(0, len(action_options) - 1)))
            chosen = (action_options[action_idx] if action_options else [])

            before_score = int(getattr(self.env, "score", 0) or 0)
            before_history_len = len(getattr(self.env, "history", []))

            obs, action_options, rewards, done = self.env.step({"player": player, "action": action_idx})

            # 出牌事件：显示出牌后该玩家的真实手牌
            self.printer.print_event(
                Event(
                    name="出牌",
                    content=f"play={list(chosen)}",
                    player=player,
                    player_hand=_hand_names(self.env, player),
                )
            )

            # 一墩结束：env.step 内部会在第 4 家出完后调用 _checkWinner 并产生 rewards
            after_history_len = len(getattr(self.env, "history", []))
            after_score = int(getattr(self.env, "score", 0) or 0)
            if rewards is not None and (before_history_len == 3):
                # 此时上一手是第 4 手；winner 就是 env.curr_player（已被设为 winner）
                winner = int(getattr(self.env, "curr_player", -1))
                delta = after_score - before_score
                self.printer.print_event(
                    Event(
                        name="墩结算",
                        content=f"winner=P{winner} farmer_score_delta={delta} farmer_score_total={after_score}",
                        player=winner if winner >= 0 else player,
                        player_hand=_hand_names(self.env, winner if winner >= 0 else player),
                    )
                )
                self.printer.print_round_hands(_all_hands(self.env))

            if done:
                final_score = int(getattr(self.env, "score", 0) or 0)
                self.printer.print_event(
                    Event(
                        name="结束",
                        content=f"done=True farmer_score_total={final_score} banker=P{getattr(self.env, 'banker_pos', None)} major={getattr(self.env, 'major', None)} level={getattr(self.env, 'level', None)}",
                        player=int(getattr(self.env, "curr_player", 0)),
                        player_hand=_hand_names(self.env, int(getattr(self.env, "curr_player", 0))),
                    )
                )
                self.printer.print_round_hands(_all_hands(self.env))
                break

    def _pick_banker_pos(self) -> int:
        banker = self.config.get("banker_pos")
        if banker is None or banker == "random":
            return self.rng.randrange(0, 4)
        return int(banker) % 4

    def _latest_dealt_card_name(self, player: int) -> str:
        if not self.env.player_decks[player]:
            return "<none>"
        return self.env._id2name(self.env.player_decks[player][-1])

    @staticmethod
    def _format_snatch_event(chosen_names: Sequence[str]) -> Tuple[str, str]:
        if not chosen_names:
            return "不报", "pass"
        if len(chosen_names) == 1:
            return "报主", f"declare={list(chosen_names)}"
        if len(chosen_names) == 2:
            return "反主", f"snatch={list(chosen_names)}"
        return "异常", f"invalid_action={list(chosen_names)}"


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": None,
    "banker_pos": "random",
    "deal_rounds": 100,
    "show_sorted_hand": True,
    # 你可以手动指定级牌序列；不指定则每局随机 shuffle 一次。
    # "level_sequence": ["2","3",...],
    "players": {
        0: {"type": "heuristic_snatch", "auto_snatch_on_level": True},
        1: {"type": "heuristic_snatch", "auto_snatch_on_level": True},
        2: {"type": "heuristic_snatch", "auto_snatch_on_level": True},
        3: {"type": "heuristic_snatch", "auto_snatch_on_level": True},
    },
    # 扣底与出牌的默认策略（全局）。
    "bury": {"type": "kitty"},
    "play": {"type": "rule_based"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Tractor deal-stage simulator (100 rounds).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    parser.add_argument("--banker", type=str, default="random", help="Banker seat: 0/1/2/3 or 'random'.")
    parser.add_argument("--level", type=str, default=None, help="Force level for this game (e.g. '2','A','0','J').")
    args = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    cfg["banker_pos"] = args.banker
    if args.level is not None:
        if args.level not in LEVELS:
            raise ValueError(f"Invalid level: {args.level!r}, must be one of {LEVELS}")
        cfg["level_sequence"] = [args.level] + [lv for lv in LEVELS if lv != args.level]

    sim = LocalSimulator(cfg)
    sim.run_one_game()


if __name__ == "__main__":
    main()
