import json
import os
import random
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

import torch

from declaration import decide_declaration, decide_overcall
from env import TractorEnv
from kitty import select_kitty_cards
from model import CNNModel
from rule_based_model import RuleBasedModel
from wrapper import cardWrapper

LEVELS = ["2", "3", "4", "5", "6", "7", "8", "9", "0", "J", "Q", "K", "A"]
WIN_THRESHOLD = 80


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


def _select_play_action(
    policy: str,
    state: Dict[str, torch.Tensor],
    mask_tensor: torch.Tensor,
    rng: random.Random,
    model: Optional[CNNModel],
    rule_model: Optional[RuleBasedModel],
) -> int:
    if policy == "model":
        if model is None:
            return 0
        model.eval()
        with torch.no_grad():
            logits, _ = model(state)
            masked = _masked_logits(logits.squeeze(0), mask_tensor)
            action_dist = torch.distributions.Categorical(logits=masked)
            return int(action_dist.sample().item())
    if policy == "rule":
        if rule_model is None:
            return 0
        rule_model.eval()
        with torch.no_grad():
            logits, _ = rule_model(state)
            masked = _masked_logits(logits.squeeze(0), mask_tensor)
            return int(torch.argmax(masked).item())
    if policy == "random":
        valid_indices = torch.nonzero(mask_tensor > 0, as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return 0
        idx = rng.randrange(0, int(valid_indices.numel()))
        return int(valid_indices[idx].item())
    return 0


def _play_one_game_score(
    team0_policy: str,
    team1_policy: str,
    model: Optional[CNNModel],
    rule_model: Optional[RuleBasedModel],
    wrapper: cardWrapper,
    rng: random.Random,
    auto_snatch_on_level: bool,
) -> Tuple[float, int]:
    env = TractorEnv()
    level = rng.choice(LEVELS)
    banker_pos = rng.randrange(0, 4)
    obs, action_options = env.reset(level=level, banker_pos=banker_pos, major="r")

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

        policy = team0_policy if (player % 2) == 0 else team1_policy
        action = _select_play_action(policy, state, mask_tensor, rng, model, rule_model)

        obs, action_options, _, done = env.step({"player": player, "action": action})

    score = float(getattr(env, "score", 0))
    banker_parity = env.banker_pos % 2
    return score, banker_parity


def _team_win(score: float, banker_parity: int, team_parity: int) -> bool:
    farmer_parity = (banker_parity + 1) % 2
    farmers_win = score >= WIN_THRESHOLD
    team_is_farmer = (team_parity % 2) == farmer_parity
    return farmers_win if team_is_farmer else (not farmers_win)


def _play_one_game(
    model: CNNModel,
    rule_model: RuleBasedModel,
    wrapper: cardWrapper,
    rng: random.Random,
    rl_team: int,
    auto_snatch_on_level: bool,
) -> float:
    if (rl_team % 2) == 0:
        team0_policy = "model"
        team1_policy = "rule"
    else:
        team0_policy = "rule"
        team1_policy = "model"
    score, banker_parity = _play_one_game_score(
        team0_policy,
        team1_policy,
        model,
        rule_model,
        wrapper,
        rng,
        auto_snatch_on_level,
    )
    rl_win = _team_win(score, banker_parity, rl_team)
    return 1.0 if rl_win else 0.0


def evaluate_rulebase_vs_random(num_games: int = 1000) -> float:
    rng = random.Random()
    wrapper = cardWrapper()
    rule_model = RuleBasedModel().eval()
    wins = 0
    for _ in tqdm(range(num_games), desc="RuleBased vs Random", leave=False):
        score, banker_parity = _play_one_game_score(
            "rule",
            "random",
            None,
            rule_model,
            wrapper,
            rng,
            auto_snatch_on_level=True,
        )
        if _team_win(score, banker_parity, 0):
            wins += 1
    return wins / float(num_games)


def evaluate_init_model_vs_rulebase(
    checkpoint_path: str = "init_model.pt",
    num_games: int = 1000,
) -> float:
    rng = random.Random()
    wrapper = cardWrapper()
    rule_model = RuleBasedModel().eval()
    device = torch.device("cpu")
    model = CNNModel()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    wins = 0
    for _ in tqdm(range(num_games), desc="InitModel vs RuleBased", leave=False):
        score, banker_parity = _play_one_game_score(
            "model",
            "rule",
            model,
            rule_model,
            wrapper,
            rng,
            auto_snatch_on_level=True,
        )
        if _team_win(score, banker_parity, 0):
            wins += 1
    return wins / float(num_games)


def evaluate_init_model_vs_random(
    checkpoint_path: str = "init_model.pt",
    num_games: int = 1000,
) -> float:
    rng = random.Random()
    wrapper = cardWrapper()
    rule_model = None
    device = torch.device("cpu")
    model = CNNModel()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    wins = 0
    for _ in tqdm(range(num_games), desc="InitModel vs Random", leave=False):
        score, banker_parity = _play_one_game_score(
            "model",
            "random",
            model,
            rule_model,
            wrapper,
            rng,
            auto_snatch_on_level=True,
        )
        if _team_win(score, banker_parity, 0):
            wins += 1
    return wins / float(num_games)


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
    for _ in tqdm(range(eval_games), desc="Eval checkpoint", leave=False):
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


if __name__ == "__main__":
    rb_vs_random = evaluate_rulebase_vs_random(num_games=3000)
    print(f"[Baseline] RuleBased vs Random win_rate={rb_vs_random:.4f}")
    init_vs_rule = evaluate_init_model_vs_rulebase(checkpoint_path="init_model.pt", num_games=1000)
    print(f"[Baseline] init_model.pt vs RuleBased win_rate={init_vs_rule:.4f}")
    init_vs_random = evaluate_init_model_vs_random(checkpoint_path="init_model.pt", num_games=1000)
    print(f"[Baseline] init_model.pt vs Random win_rate={init_vs_random:.4f}")
