import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from data.wrapper import cardWrapper

POINT_CARD_VALUES = {"5": 5, "0": 10, "K": 10}
RANK_ORDER = {rank: idx for idx, rank in enumerate(["2", "3", "4", "5", "6", "7", "8", "9", "0", "J", "Q", "K", "A"])}
EARLY_CARD_LIMIT = 20
MID_CARD_LIMIT = 70
BIG_TRUMP_STRENGTH = 950


class RuleBasedModel(nn.Module):
    """Drop-in replacement for CNNModel that produces rule-based logits on CPU."""

    def __init__(self, take_points_threshold: int = 20, trump_push_threshold: int = 8):
        super().__init__()
        self.wrapper = cardWrapper()
        self.take_points_threshold = take_points_threshold
        self.trump_push_threshold = trump_push_threshold

    def forward(self, input_dict: Dict[str, torch.Tensor]):
        observations = input_dict["observation"].detach()
        masks = input_dict["action_mask"].detach()
        device = observations.device
        batch_meta = input_dict.get("meta")

        logits_batch: List[torch.Tensor] = []
        values_batch: List[torch.Tensor] = []
        batch_size = observations.shape[0]

        for idx in range(batch_size):
            sample_meta = self._select_meta(batch_meta, idx)
            logits = self._compute_sample_logits(observations[idx], masks[idx], sample_meta)
            logits_batch.append(logits.to(device))
            values_batch.append(torch.zeros(1, dtype=observations.dtype, device=device))

        return torch.stack(logits_batch, dim=0), torch.stack(values_batch, dim=0)

    def _select_meta(self, meta: Any, index: int) -> Optional[Dict[str, Any]]:
        if meta is None:
            return None
        if isinstance(meta, (list, tuple)):
            if 0 <= index < len(meta):
                entry = meta[index]
                return entry if isinstance(entry, dict) else None
            return None
        if isinstance(meta, dict):
            return meta
        return None

    def _compute_sample_logits(self, obs_sample: torch.Tensor, mask_sample: torch.Tensor, meta: Optional[Dict[str, Any]]) -> torch.Tensor:
        obs_np = obs_sample.cpu().numpy()
        mask_np = mask_sample.cpu().numpy()

        major_block = obs_np[0:2]
        deck_block = obs_np[2:4]
        history_block = obs_np[4:12]
        played_block = obs_np[12:20]
        options_block = obs_np[20:]

        major_cards = self.wrapper.Unwrap(major_block.copy())
        deck_cards = self.wrapper.Unwrap(deck_block.copy())
        history_moves = self._decode_sequence(history_block)
        played_piles = self._decode_played(played_block)
        played_cards = [card for pile in played_piles for card in pile]
        history_cards_flat = [card for move in history_moves for card in move]

        major_suit, level_rank = self._infer_major_info(major_cards)
        valid_indices = [int(i) for i, flag in enumerate(mask_np) if flag > 0.5]
        option_slices = self._decode_options(options_block, valid_indices)

        option_features = [
            self._build_option_feature(idx, cards, major_suit, level_rank)
            for idx, cards in zip(valid_indices, option_slices)
        ]

        total_points_played = self._total_points_played(played_cards)
        role = self._infer_role(meta)
        score_state = self._infer_score_state(meta)
        void_map = self._normalize_void_map(meta)
        stage = self._classify_game_stage(len(played_cards) + len(history_cards_flat))
        context = {
            "major": major_suit,
            "level": level_rank,
            "deck": deck_cards,
            "history": history_moves,
            "played": played_cards,
            "points_on_table": self._count_points(history_moves),
            "options": option_features,
            "meta": meta or {},
            "stage": stage,
            "point_visibility": self._point_visibility_tracker(played_cards, history_moves),
            "role": role,
            "score_state": score_state,
            "scenario_key": f"{role}:{stage}:{score_state}",
            "void_map": void_map,
            "player_id": (meta or {}).get("player_id"),
            "teammate_id": (meta or {}).get("teammate_id"),
            "remaining_cards": (meta or {}).get("remaining_cards", len(deck_cards)),
            "kitty_points": (meta or {}).get("kitty_points", 0),
            "kitty_high_risk": bool((meta or {}).get("kitty_high_risk", False)),
            "score_margin": (meta or {}).get("score_margin", 0),
            "current_trick_order": (meta or {}).get("current_trick_order", []),
            "total_points_played": total_points_played,
            "last_completed_points": (meta or {}).get("last_completed_points", 0),
        }
        context["teammate_flags"] = self._build_teammate_flags(context)
        context["teammate_void_suits"], context["enemy_void_suits"] = self._split_void_map(context)

        if not option_features:
            logits = torch.full((mask_np.shape[0],), -1e9, dtype=obs_sample.dtype)
            logits[0] = 0.0
            return logits

        if not history_moves:
            chosen = self._lead_policy(context)
        else:
            chosen = self._follow_policy(context)

        if chosen is None:
            chosen = option_features[0]["index"]

        logits = torch.full((mask_np.shape[0],), -1e9, dtype=obs_sample.dtype)
        logits[chosen] = 0.0
        return logits

    def _decode_sequence(self, block: Sequence[Sequence[Sequence[float]]]) -> List[List[str]]:
        moves: List[List[str]] = []
        for offset in range(0, len(block), 2):
            chunk = block[offset : offset + 2]
            cards = self.wrapper.Unwrap(chunk.copy())
            if not cards:
                break
            moves.append(cards)
        return moves

    def _decode_options(self, option_block, indices: List[int]) -> List[List[str]]:
        slices: List[List[str]] = []
        for idx in indices:
            start = idx * 2
            end = start + 2
            if end > option_block.shape[0]:
                slices.append([])
                continue
            chunk = option_block[start:end]
            slices.append(self.wrapper.Unwrap(chunk.copy()))
        return slices

    def _decode_played(self, block: Sequence[Sequence[Sequence[float]]]) -> List[List[str]]:
        piles: List[List[str]] = []
        for offset in range(0, len(block), 2):
            chunk = block[offset : offset + 2]
            piles.append(self.wrapper.Unwrap(chunk.copy()))
        return piles

    def _infer_major_info(self, major_cards: List[str]) -> Tuple[str, str]:
        major_suit = "s"
        level_rank = "2"
        for card in major_cards:
            if len(card) == 2:
                major_suit = card[0]
                break
        rank_counts: Dict[str, int] = {}
        for card in major_cards:
            if len(card) == 2:
                rank_counts[card[1]] = rank_counts.get(card[1], 0) + 1
        level_candidates = [rank for rank, cnt in rank_counts.items() if cnt > 1]
        if level_candidates:
            level_rank = level_candidates[0]
        return major_suit, level_rank

    def _build_option_feature(self, index: int, cards: List[str], major: str, level: str) -> Dict[str, Any]:
        strengths = [self._card_strength(card, major, level) for card in cards]
        contains_points = any(len(card) == 2 and card[1] in POINT_CARD_VALUES for card in cards)
        premium = any(card in ("jo", "Jo") for card in cards)
        premium |= any(len(card) == 2 and card[1] == level for card in cards)
        is_trump = all(self._is_trump(card, major, level) for card in cards) if cards else False
        points_value = sum(POINT_CARD_VALUES.get(card[1], 0) for card in cards if len(card) == 2)
        is_pair = len(cards) == 2 and len(set(cards)) == 1
        pair_rank: Optional[str] = None
        pair_suit: Optional[str] = None
        if is_pair:
            sample = cards[0]
            if len(sample) == 2 and sample[0] in ("s", "h", "c", "d"):
                pair_suit = sample[0]
                pair_rank = sample[1]
            else:
                is_pair = False
        primary_suit = self._option_primary_suit(cards, major, level)
        feature = {
            "index": index,
            "cards": cards,
            "length": len(cards),
            "strengths": sorted(strengths, reverse=True),
            "max_strength": max(strengths) if strengths else -math.inf,
            "min_strength": min(strengths) if strengths else math.inf,
            "is_trump": is_trump,
            "contains_points": contains_points,
            "points_value": points_value,
            "premium_cost": self._premium_cost(cards, major, level),
            "is_pair": is_pair,
            "pair_rank": pair_rank,
            "pair_suit": pair_suit,
            "primary_suit": primary_suit,
        }
        return feature

    def _count_points(self, moves: List[List[str]]) -> int:
        total = 0
        for move in moves:
            for card in move:
                if len(card) == 2:
                    total += POINT_CARD_VALUES.get(card[1], 0)
        return total

    def _card_strength(self, card: str, major: str, level: str) -> int:
        if card == "Jo":
            return 1000
        if card == "jo":
            return 999
        if len(card) != 2:
            return -1
        suit, rank = card[0], card[1]
        base = RANK_ORDER.get(rank, 0)
        strength = base
        if rank == level:
            strength += 200
        if suit == major:
            strength += 120
        if rank == level and suit == major:
            strength += 40
        return strength

    def _is_trump(self, card: str, major: str, level: str) -> bool:
        if card in ("jo", "Jo"):
            return True
        if len(card) != 2:
            return False
        return card[0] == major or card[1] == level

    def _premium_cost(self, cards: List[str], major: str, level: str) -> int:
        cost = 0
        for card in cards:
            if card == "Jo":
                cost += 30
            elif card == "jo":
                cost += 20
            elif len(card) == 2 and card[1] == level:
                cost += 15
            elif len(card) == 2 and card[0] == major:
                cost += 5
        return cost

    def _beats(self, option: Dict[str, Any], current_best: Optional[Dict[str, Any]]) -> bool:
        if current_best is None:
            return True
        if option["is_trump"] and not current_best["is_trump"]:
            return True
        if not option["is_trump"] and current_best["is_trump"]:
            return False
        if option["length"] != current_best["length"]:
            return option["length"] > current_best["length"]
        return option["strengths"] > current_best["strengths"]

    def _current_best(self, history: List[List[str]], major: str, level: str) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        for move in history:
            candidate = self._build_option_feature(-1, move, major, level)
            if best is None or self._beats(candidate, best):
                best = candidate
        return best

    def _follow_policy(self, context: Dict[str, Any]) -> Optional[int]:
        options = context["options"]
        major = context["major"]
        level = context["level"]
        history = context["history"]
        current_best = self._current_best(history, major, level)
        if current_best is None:
            return self._lead_policy(context)

        winning = [opt for opt in options if self._beats(opt, current_best)]
        fillers = [opt for opt in options if opt not in winning]
        winning = self._prioritize_pair_release(winning, context)
        fillers = self._prioritize_pair_release(fillers, context)
        winning = self._enforce_primary_constraints(winning, context)
        fillers = self._enforce_primary_constraints(fillers, context)
        winning = self._refine_options(winning, context, priority="winning")
        fillers = self._refine_options(fillers, context, priority="filler")

        points_on_table = context["points_on_table"]
        meta = context["meta"]
        need_takeover = points_on_table >= self.take_points_threshold or bool(meta.get("force_take", False))
        if meta.get("protect_points", False):
            need_takeover = True

        if winning:
            if need_takeover:
                return self._select_winning_option(winning, prefer_preserve=True)
            if points_on_table == 0 and not meta.get("apply_pressure", False):
                return self._select_minimal(winning)
            return self._select_winning_option(winning, prefer_preserve=False)

        if fillers:
            safe = [opt for opt in fillers if not opt["contains_points"]]
            safe = self._prioritize_pair_release(safe, context)
            safe = self._enforce_primary_constraints(safe, context)
            safe = self._refine_options(safe, context, priority="safe")
            if safe:
                return self._select_smallest(safe)
            if meta.get("allow_point_dump", False) or points_on_table == 0:
                return self._select_smallest(fillers)
            return self._select_smallest(fillers)

        return None

    def _lead_policy(self, context: Dict[str, Any]) -> Optional[int]:
        options = context["options"]
        deck_cards = context["deck"]
        major = context["major"]
        level = context["level"]
        meta = context["meta"]

        trump_count = sum(1 for card in deck_cards if self._is_trump(card, major, level))
        is_banker = bool(meta.get("is_banker", False))
        has_advantage = bool(meta.get("advantage", False))

        want_trump = False
        if is_banker and (has_advantage or trump_count >= self.trump_push_threshold):
            want_trump = True
        elif not is_banker and not has_advantage and trump_count >= self.trump_push_threshold:
            want_trump = True
        elif meta.get("lead_trump", False):
            want_trump = True

        trump_opts = [opt for opt in options if opt["is_trump"]]
        off_opts = [opt for opt in options if not opt["is_trump"]]
        trump_opts = self._prioritize_pair_release(trump_opts, context)
        off_opts = self._prioritize_pair_release(off_opts, context)
        trump_opts = self._enforce_primary_constraints(trump_opts, context, is_lead=True)
        off_opts = self._enforce_primary_constraints(off_opts, context, is_lead=True)
        trump_opts = self._refine_options(trump_opts, context, priority="lead-trump")
        off_opts = self._refine_options(off_opts, context, priority="lead-off")

        if want_trump and trump_opts:
            return self._select_trump_lead(trump_opts)

        if off_opts:
            safe = [opt for opt in off_opts if not opt["contains_points"]]
            safe = self._prioritize_pair_release(safe, context)
            safe = self._enforce_primary_constraints(safe, context, is_lead=True)
            safe = self._refine_options(safe, context, priority="lead-safe")
            if safe:
                return self._select_offensive_lead(safe)
            if meta.get("allow_point_dump", False):
                return self._select_offensive_lead(off_opts)
            return self._select_minimum_point_dump(off_opts)

        if trump_opts:
            return self._select_trump_lead(trump_opts)

        return options[0]["index"] if options else None

    def _select_winning_option(self, options: List[Dict[str, Any]], prefer_preserve: bool) -> int:
        def key(opt):
            priority = opt["premium_cost"] if prefer_preserve else 0
            return (priority, -opt["max_strength"], -opt["length"])

        best = min(options, key=key)
        return best["index"]

    def _select_minimal(self, options: List[Dict[str, Any]]) -> int:
        best = min(options, key=lambda opt: (opt["max_strength"], opt["length"]))
        return best["index"]

    def _select_smallest(self, options: List[Dict[str, Any]]) -> int:
        return self._select_minimal(options)

    def _select_trump_lead(self, options: List[Dict[str, Any]]) -> int:
        best = max(options, key=lambda opt: (opt["length"], opt["max_strength"] - opt["premium_cost"]))
        return best["index"]

    def _select_offensive_lead(self, options: List[Dict[str, Any]]) -> int:
        best = max(options, key=lambda opt: (opt["length"], -opt["points_value"], -opt["premium_cost"]))
        return best["index"]

    def _select_minimum_point_dump(self, options: List[Dict[str, Any]]) -> int:
        best = min(options, key=lambda opt: (opt["points_value"], opt["max_strength"]))
        return best["index"]

    def _classify_game_stage(self, cards_seen: int) -> str:
        if cards_seen <= EARLY_CARD_LIMIT:
            return "early"
        if cards_seen <= MID_CARD_LIMIT:
            return "mid"
        return "late"

    def _point_visibility_tracker(self, played_cards: List[str], history: List[List[str]]) -> Dict[str, Dict[str, bool]]:
        tracker: Dict[str, Dict[str, bool]] = {suit: {"0": False, "5": False} for suit in ("s", "h", "c", "d")}

        def mark(card: str):
            if len(card) != 2:
                return
            suit, rank = card[0], card[1]
            if suit in tracker and rank in tracker[suit]:
                tracker[suit][rank] = True

        for pile_card in played_cards:
            mark(pile_card)
        for move in history:
            for card in move:
                mark(card)
        return tracker

    def _prioritize_pair_release(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not options:
            return []
        allowed, restricted = self._partition_pair_options(options, context)
        return allowed if allowed else restricted

    def _partition_pair_options(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        allowed: List[Dict[str, Any]] = []
        restricted: List[Dict[str, Any]] = []
        for opt in options:
            if self._is_pair_restricted(opt, context):
                restricted.append(opt)
            else:
                allowed.append(opt)
        return allowed, restricted

    def _is_pair_restricted(self, option: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not option.get("is_pair"):
            return False
        if option.get("is_trump"):
            return False
        rank = option.get("pair_rank")
        suit = option.get("pair_suit")
        if rank is None or suit is None:
            return False
        stage = context.get("stage", "early")
        rank_value = RANK_ORDER.get(rank, -1)
        ten_value = RANK_ORDER["0"]
        five_value = RANK_ORDER["5"]
        if stage == "late":
            return False
        if stage == "early":
            return rank_value <= ten_value

        visibility = context.get("point_visibility", {})
        suit_vis = visibility.get(suit, {"0": False, "5": False})
        if rank_value < ten_value and not suit_vis.get("0", False):
            return True
        if rank_value <= five_value and not suit_vis.get("5", False):
            return True
        return False

    def _infer_role(self, meta: Optional[Dict[str, Any]]) -> str:
        if not meta:
            return "farmer"
        if meta.get("is_banker") is True:
            return "banker"
        if meta.get("is_banker") is False:
            return "farmer"
        banker_id = meta.get("banker_id")
        player_id = meta.get("player_id")
        if banker_id is None or player_id is None:
            return "farmer"
        return "banker" if player_id % 4 == banker_id % 4 else "farmer"

    def _infer_score_state(self, meta: Optional[Dict[str, Any]]) -> str:
        if not meta:
            return "balanced"
        state = meta.get("score_state")
        if state in {"leading", "trailing", "balanced"}:
            return state
        margin = meta.get("score_margin", 0)
        if margin >= 20:
            return "leading"
        if margin <= -20:
            return "trailing"
        return "balanced"

    def _normalize_void_map(self, meta: Optional[Dict[str, Any]]) -> Dict[int, set]:
        norm: Dict[int, set] = {}
        if not meta:
            return norm
        raw_map = meta.get("void_map") or {}
        for seat, suits in raw_map.items():
            try:
                seat_idx = int(seat)
            except (ValueError, TypeError):
                continue
            if isinstance(suits, (list, tuple, set)):
                norm[seat_idx] = set(suits)
        return norm

    def _split_void_map(self, context: Dict[str, Any]) -> Tuple[set, set]:
        void_map = context.get("void_map") or {}
        teammate_id = context.get("teammate_id")
        player_id = context.get("player_id")
        teammate_void = set(void_map.get(teammate_id, set())) if teammate_id is not None else set()
        enemy_void: set = set()
        for seat, suits in void_map.items():
            if seat in (teammate_id, player_id):
                continue
            enemy_void.update(suits)
        return teammate_void, enemy_void

    def _build_teammate_flags(self, context: Dict[str, Any]) -> Dict[str, Any]:
        flags = {
            "follow_state": "unknown",
            "dumped_points": False,
            "played_trump": False,
            "void_suit": None,
        }
        order = context.get("current_trick_order") or []
        history = context.get("history") or []
        teammate_id = context.get("teammate_id")
        if not order or not history or teammate_id is None:
            return flags
        major = context.get("major", "s")
        level = context.get("level", "2")
        lead_suit = self._lead_suit_from_history(history, major, level)
        for idx, move in enumerate(history):
            if idx >= len(order):
                break
            if order[idx] != teammate_id:
                continue
            if not move:
                flags["follow_state"] = "void"
                flags["void_suit"] = lead_suit
                return flags
            names = move
            if any(self._card_suit(card, major, level) == lead_suit for card in names):
                flags["follow_state"] = "followed"
            else:
                flags["follow_state"] = "void"
                flags["void_suit"] = lead_suit
            flags["dumped_points"] = any(len(card) == 2 and card[1] in POINT_CARD_VALUES for card in names)
            flags["played_trump"] = any(self._is_trump(card, major, level) for card in names)
            break
        return flags

    def _lead_suit_from_history(self, history: List[List[str]], major: str, level: str) -> Optional[str]:
        if not history or not history[0]:
            return None
        return self._card_suit(history[0][0], major, level)

    def _card_suit(self, card: str, major: str, level: str) -> Optional[str]:
        if card in ("jo", "Jo"):
            return "trump"
        if len(card) != 2:
            return None
        suit, rank = card[0], card[1]
        if rank == level or suit == major:
            return "trump"
        return suit

    def _option_primary_suit(self, cards: List[str], major: str, level: str) -> Optional[str]:
        for card in cards:
            suit = self._card_suit(card, major, level)
            if suit:
                return suit
        return None

    def _enforce_primary_constraints(self, options: List[Dict[str, Any]], context: Dict[str, Any], is_lead: bool = False) -> List[Dict[str, Any]]:
        if not options:
            return []
        history = context.get("history") or []
        if is_lead or not history:
            return options
        major = context.get("major", "s")
        level = context.get("level", "2")
        lead_suit = self._lead_suit_from_history(history, major, level)
        if not lead_suit or lead_suit == "trump":
            return options
        singles = [
            opt
            for opt in options
            if opt["length"] == 1 and not opt["contains_points"] and not opt["is_trump"] and opt.get("primary_suit") == lead_suit
        ]
        if len(singles) <= 1:
            return options
        weakest = min(singles, key=lambda opt: opt["max_strength"])
        filtered = [opt for opt in options if opt not in singles or opt is weakest]
        return filtered or options

    def _refine_options(self, options: List[Dict[str, Any]], context: Dict[str, Any], priority: str) -> List[Dict[str, Any]]:
        if not options:
            return []
        preferred, deferred = self._apply_record_adjustments(options, context, priority)
        return preferred if preferred else deferred

    def _apply_record_adjustments(self, options: List[Dict[str, Any]], context: Dict[str, Any], priority: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        preferred: List[Dict[str, Any]] = []
        deferred: List[Dict[str, Any]] = []
        for opt in options:
            if self._should_preserve_big_trump(opt, context):
                deferred.append(opt)
            else:
                preferred.append(opt)
        preferred.sort(key=lambda opt: (self._contextual_score(opt, context, priority), opt["max_strength"], opt["length"]))
        deferred.sort(key=lambda opt: (-opt["max_strength"], opt["length"]))
        return preferred, deferred

    def _contextual_score(self, option: Dict[str, Any], context: Dict[str, Any], priority: str) -> float:
        score = 0.0
        stage = context.get("stage", "early")
        score_state = context.get("score_state", "balanced")
        teammate_flags = context.get("teammate_flags", {})
        teammate_void = context.get("teammate_void_suits", set())
        enemy_void = context.get("enemy_void_suits", set())
        primary_suit = option.get("primary_suit")
        if score_state == "leading" and option.get("contains_points"):
            score += 25
        if score_state == "trailing" and priority == "winning":
            score -= option.get("max_strength", 0) * 0.1
        if teammate_flags.get("follow_state") == "void" and primary_suit == teammate_flags.get("void_suit"):
            score -= 15 if option.get("contains_points") else 0
        if teammate_void and primary_suit in teammate_void and option.get("contains_points"):
            score -= 5
        if enemy_void and primary_suit in enemy_void and option.get("contains_points"):
            score += 20
        if stage == "late" and option.get("contains_points") and priority.startswith("lead"):
            score += 10
        if teammate_flags.get("dumped_points") and option.get("contains_points"):
            score += 5
        return score

    def _should_preserve_big_trump(self, option: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not option.get("is_trump"):
            return False
        max_strength = option.get("max_strength", 0)
        if max_strength < BIG_TRUMP_STRENGTH:
            return False
        remaining = context.get("remaining_cards", 0)
        stage = context.get("stage", "early")
        role = context.get("role", "farmer")
        kitty_high_risk = context.get("kitty_high_risk", False)
        total_points_played = context.get("total_points_played", 0)
        if role == "banker" and kitty_high_risk and remaining <= 4:
            return True
        if role == "farmer" and stage == "late" and total_points_played < 180 and remaining <= 5:
            return True
        return False

    def _total_points_played(self, played_cards: List[str]) -> int:
        total = 0
        for card in played_cards:
            if len(card) == 2:
                total += POINT_CARD_VALUES.get(card[1], 0)
        return total
