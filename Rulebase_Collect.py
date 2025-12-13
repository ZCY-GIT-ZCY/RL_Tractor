import argparse
import os
from typing import List

import numpy as np
import torch
from tqdm import trange

from env import TractorEnv
from wrapper import cardWrapper
from rule_based_model import RuleBasedModel
from kitty import select_kitty_cards


def cover_Pub(old_public, deck, level, major):
    full_hand = list(deck) + list(old_public)
    bury_count = len(old_public)
    selected = select_kitty_cards(full_hand, level, major, bury_count)
    remaining = list(full_hand)
    for card in selected:
        if card in remaining:
            remaining.remove(card)
    return selected


def _find_option_index(options: List[List[str]], target: List[str]) -> int:
    for idx, option in enumerate(options):
        if len(option) != len(target):
            continue
        if option == target:
            return idx
    return 0


def collect_rulebase_games(num_games: int, out_path: str):
    env = TractorEnv()
    wrapper = cardWrapper()
    policy = RuleBasedModel().eval()

    observations = []
    masks = []
    labels = []

    for game_idx in trange(num_games, desc="Rulebase self-play"):
        banker = game_idx % 4
        obs, action_options = env.reset(major='r', banker_pos=banker)
        done = False
        while not done:
            stage = obs.get("stage", env.STAGE_PLAY)
            player = obs["id"]
            if stage == env.STAGE_SNATCH:
                action = 0
            elif stage == env.STAGE_BURY:
                selected = cover_Pub(env.card_public, env.player_decks[player], env.level, env.major or "n")
                if selected:
                    target = [env._id2name(card) for card in selected]
                    action = _find_option_index(action_options, target)
                else:
                    action = 0
            else:
                obs_mat, action_mask = wrapper.obsWrap(obs, action_options)
                obs_tensor = torch.tensor(obs_mat, dtype=torch.float32).unsqueeze(0)
                mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = policy({"observation": obs_tensor, "action_mask": mask_tensor})
                action = torch.argmax(logits, dim=-1).item()
                action = max(0, min(action, len(action_options) - 1))
                observations.append(obs_mat.astype(np.float32))
                masks.append(action_mask.astype(np.float32))
                labels.append(action)

            response = {"player": player, "action": action}
            obs, action_options, _, done = env.step(response)

    if not observations:
        raise RuntimeError("No samples were generated.")

    obs_arr = np.stack(observations)
    mask_arr = np.stack(masks)
    label_arr = np.asarray(labels, dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(out_path, observation=obs_arr, action_mask=mask_arr, action=label_arr)
    print(f"Saved {len(observations)} samples to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000, help="Number of self-play games to run.")
    parser.add_argument("--out", type=str, default="Pre_trained_Data/Rulebase_Data.npz", help="Path to save dataset.")
    args = parser.parse_args()
    collect_rulebase_games(args.games, args.out)


if __name__ == "__main__":
    main()
