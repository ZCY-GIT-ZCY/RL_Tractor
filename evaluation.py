import json
import os
import random
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Tuple

import torch

from declaration import decide_declaration, decide_overcall
from env import TractorEnv
from kitty import select_kitty_cards
from model import CNNModel
from rule_based_model import RuleBasedModel
from wrapper import cardWrapper

LEVELS = ["2", "3", "4", "5", "6", "7", "8", "9", "0", "J", "Q", "K", "A"]


@dataclass
class EvalTask:
    iterations: int
    checkpoint_path: str


def _find_option_index(action_options: List[List[str]], target_cards: List[str]) -> Optional[int]:
    if not action_options:
        return None
    for idx, option in enumerate(action_options):
        if option == target_cards:
            return idx
    return None


def _select_declaration_action(env: TractorEnv, obs, action_options, auto_snatch_on_level: bool) -> int:
    level = env.level
    deck = obs.get("deck", [])
    if env.reporter is None:
        candidate = decide_declaration(deck, level, force_on_level=auto_snatch_on_level)
        if not candidate:
            return 0
        target = [candidate + level]
        idx = _find_option_index(action_options, target)
        return idx if idx is not None else 0

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


def _select_bury_action(env: TractorEnv, action_options) -> int:
    banker = env.banker_pos
    if banker is None:
        return 0
    bury_count = env.bury_left
    deck_ids = list(env.player_decks[banker])
    selected = select_kitty_cards(deck_ids, env.level, env.major or "n", bury_count)
    if not selected:
        return 0
    target_name = env._id2name(selected[0])
    idx = _find_option_index(action_options, [target_name])
    return idx if idx is not None else 0


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(mask <= 0, -1e9)


def _play_one_game(
    model: CNNModel,
    rule_model: RuleBasedModel,
    wrapper: cardWrapper,
    rng: random.Random,
    rl_team: int,
    auto_snatch_on_level: bool,
) -> float:
    env = TractorEnv()
    level = rng.choice(LEVELS)
    banker_pos = rng.randrange(0, 4)
    obs, action_options = env.reset(level=level, banker_pos=banker_pos, major="r")

    rl_players = {pid for pid in range(4) if (pid % 2) == (rl_team % 2)}
    done = False
    while not done:
        stage = obs.get("stage", TractorEnv.STAGE_PLAY)
        player = obs["id"]

        if stage == TractorEnv.STAGE_SNATCH:
            action = _select_declaration_action(env, obs, action_options, auto_snatch_on_level)
            obs, action_options, _, done = env.step({"player": player, "action": action})
            continue
        if stage == TractorEnv.STAGE_BURY:
            action = _select_bury_action(env, action_options)
            obs, action_options, _, done = env.step({"player": player, "action": action})
            continue

        obs_mat, action_mask = wrapper.obsWrap(obs, action_options)
        state = {
            "observation": torch.tensor(obs_mat, dtype=torch.float32).unsqueeze(0),
            "action_mask": torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0),
        }
        mask_tensor = state["action_mask"].squeeze(0)

        if player in rl_players:
            model.eval()
            with torch.no_grad():
                logits, _ = model(state)
                masked = _masked_logits(logits.squeeze(0), mask_tensor)
                action_dist = torch.distributions.Categorical(logits=masked)
                action = int(action_dist.sample().item())
        else:
            rule_model.eval()
            with torch.no_grad():
                logits, _ = rule_model(state)
                masked = _masked_logits(logits.squeeze(0), mask_tensor)
                action = int(torch.argmax(masked).item())

        obs, action_options, _, done = env.step({"player": player, "action": action})

    banker_parity = env.banker_pos % 2
    farmer_parity = (banker_parity + 1) % 2
    score = float(getattr(env, "score", 0))
    if (rl_team % 2) == farmer_parity:
        if score > 0:
            return 1.0
        if score < 0:
            return 0.0
    else:
        if score < 0:
            return 1.0
        if score > 0:
            return 0.0
    return 0.5


def _load_history(path: str) -> List[Dict[str, float]]:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def _save_history(path: str, history: List[Dict[str, float]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def _plot_history(history: List[Dict[str, float]], plot_path: str) -> None:
    import matplotlib.pyplot as plt

    if not history:
        return
    history_sorted = sorted(history, key=lambda x: x.get("iterations", 0))
    x_vals = [entry.get("iterations", 0) for entry in history_sorted]
    y_vals = [entry.get("win_rate", 0.0) for entry in history_sorted]

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, marker="o", linewidth=1.5)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Iterations")
    plt.ylabel("Win Rate")
    plt.title("Checkpoint Win Rate vs Rule-Based")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def evaluate_checkpoint(task: EvalTask, config: Dict) -> None:
    ckpt_path = task.checkpoint_path
    ckpt_dir = os.path.dirname(ckpt_path) or "."
    os.makedirs(ckpt_dir, exist_ok=True)

    eval_games = int(config.get("eval_games", 300))
    rl_team = int(config.get("rl_team", 0))
    auto_snatch_on_level = bool(config.get("auto_snatch_on_level", True))

    device = torch.device("cpu")
    model = CNNModel()
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    rule_model = RuleBasedModel().to(device)
    wrapper = cardWrapper()

    rng = random.Random()
    wins = 0.0
    for _ in range(eval_games):
        wins += _play_one_game(model, rule_model, wrapper, rng, rl_team, auto_snatch_on_level)

    win_rate = wins / float(eval_games)
    history_path = os.path.join(ckpt_dir, "eval_results.json")
    history = _load_history(history_path)
    history.append(
        {
            "iterations": int(task.iterations),
            "win_rate": float(win_rate),
            "games": int(eval_games),
            "timestamp": float(time.time()),
            "checkpoint": ckpt_path,
        }
    )
    _save_history(history_path, history)

    plot_name = config.get("eval_plot_name", "eval_winrate.png")
    plot_path = os.path.join(ckpt_dir, plot_name)
    _plot_history(history, plot_path)


class EvaluationWorker(Process):
    def __init__(self, queue: Queue, config: Dict):
        super().__init__()
        self.queue = queue
        self.config = dict(config)

    def run(self) -> None:
        torch.set_num_threads(1)
        while True:
            task = self.queue.get()
            if task is None:
                break
            if isinstance(task, EvalTask):
                try:
                    evaluate_checkpoint(task, self.config)
                except Exception as exc:
                    print(f"[Eval] Failed for {task.checkpoint_path}: {exc}")
