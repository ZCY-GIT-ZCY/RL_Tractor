import hashlib
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
TRUMP_TOTAL_ESTIMATE = 24
ASSERTIVE_MIN_RANK = "J"
LIGHT_MIN_RANK = "7"


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
        hand_trump_count = sum(1 for card in deck_cards if self._is_trump(card, major_suit, level_rank))
        lead_suit = self._lead_suit_from_history(history_moves, major_suit, level_rank)
        trump_drag_rounds = self._estimate_trump_drag_rounds(played_piles, history_moves, major_suit, level_rank)
        seen_trump_cards = played_cards + history_cards_flat
        total_trump_seen = self._count_trump_cards(seen_trump_cards, major_suit, level_rank)
        high_trump_seen = self._count_high_trump_seen(seen_trump_cards, major_suit, level_rank)
        trump_remaining_estimate = max(0, TRUMP_TOTAL_ESTIMATE - total_trump_seen)
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
            "lead_suit": lead_suit,
            "hand_trump_count": hand_trump_count,
            "trump_status": self._classify_trump_status(hand_trump_count),
            "trump_drag_rounds": trump_drag_rounds,
            "trump_remaining_estimate": trump_remaining_estimate,
            "trump_half_spent": trump_remaining_estimate <= TRUMP_TOTAL_ESTIMATE / 2,
            "trump_low_rounds_left": trump_remaining_estimate <= 8,
            "high_trump_seen": high_trump_seen,
            "no_trump_mark": hand_trump_count == 0,
        }
        context["teammate_flags"] = self._build_teammate_flags(context)
        context["teammate_void_suits"], context["enemy_void_suits"] = self._split_void_map(context)
        context["position_info"] = self._build_position_info(context)
        context["opponent_flags"] = self._build_opponent_flags(context)

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
        best, _ = self._current_best_with_owner(history, major, level, None)
        return best

    def _current_best_with_owner(
        self,
        history: List[List[str]],
        major: str,
        level: str,
        order: Optional[List[int]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        best: Optional[Dict[str, Any]] = None
        owner: Optional[int] = None
        for idx, move in enumerate(history):
            candidate = self._build_option_feature(-1, move, major, level)
            if best is None or self._beats(candidate, best):
                best = candidate
                owner = order[idx] if order and idx < len(order) else None
        return best, owner

    def _follow_policy(self, context: Dict[str, Any]) -> Optional[int]:
        options = context["options"]
        major = context["major"]
        level = context["level"]
        history = context["history"]
        order = context.get("current_trick_order")
        current_best, best_owner = self._current_best_with_owner(history, major, level, order)
        if current_best is None:
            return self._lead_policy(context)

        teammate_id = context.get("teammate_id")
        teammate_winning = best_owner is not None and teammate_id is not None and best_owner == teammate_id
        context["teammate_winning"] = teammate_winning
        position_info = context.get("position_info", {})
        enemy_flags = context.get("opponent_flags", {})
        enemy_dumped_points = enemy_flags.get("dumped_points", False)
        enemy_no_points = enemy_flags.get("no_points_seen", False)

        winning = [opt for opt in options if self._beats(opt, current_best)]
        fillers = [opt for opt in options if opt not in winning]
        winning = self._apply_trump_takeover_filters(winning, context)
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
        if enemy_dumped_points:
            need_takeover = True
        teammate_dumped = context.get("teammate_flags", {}).get("dumped_points")
        if teammate_dumped and winning:
            if not self._can_ignore_teammate_points(context, winning, points_on_table):
                need_takeover = True
        opponent_winning = not teammate_winning
        if opponent_winning and points_on_table > 0:
            need_takeover = True
        if enemy_no_points and not meta.get("force_take", False) and points_on_table <= 5:
            need_takeover = False
        if position_info.get("is_last") and points_on_table > 0:
            need_takeover = True

        force_take_override = False
        teammate_trump_push = self._teammate_trump_lead_pressure(context)
        if teammate_trump_push and winning:
            need_takeover = True
            force_take_override = True
        special_takeover = self._should_force_special_takeover(context, winning)
        if special_takeover:
            need_takeover = True
            force_take_override = True
        if self._should_take_on_strength(context, winning, opponent_winning):
            need_takeover = True

        dump_candidates = self._prepare_point_dump_candidates(fillers, context)
        safe_options = self._prepare_safe_fillers(fillers, context)
        decision = self._evaluate_follow_decision(
            context=context,
            winning=winning,
            fillers=fillers,
            safe_options=safe_options,
            dump_candidates=dump_candidates,
            need_takeover=need_takeover,
            teammate_winning=teammate_winning,
            opponent_winning=opponent_winning,
            current_best=current_best,
            force_take_override=force_take_override,
        )
        if decision:
            decision_type, choice = decision
            context["decision_type"] = decision_type
            return choice

        if winning:
            return self._select_winning_option(winning, prefer_preserve=False, context=context, current_best=current_best)
        if fillers:
            return self._select_low_risk_filler(fillers)
        return None

    def _lead_policy(self, context: Dict[str, Any]) -> Optional[int]:
        options = context["options"]
        deck_cards = context["deck"]
        major = context["major"]
        level = context["level"]
        meta = context["meta"]

        trump_count = context.get("hand_trump_count")
        if trump_count is None:
            trump_count = sum(1 for card in deck_cards if self._is_trump(card, major, level))
        is_banker = bool(meta.get("is_banker", False))
        has_advantage = bool(meta.get("advantage", False))
        trump_status = context.get("trump_status", "balanced")

        want_trump = False
        if trump_status == "advantage":
            want_trump = True
        elif is_banker and (has_advantage or trump_count >= self.trump_push_threshold):
            want_trump = True
        elif not is_banker and not has_advantage and trump_count >= self.trump_push_threshold:
            want_trump = True
        elif meta.get("lead_trump", False):
            want_trump = True
        if trump_status == "disadvantage" and not meta.get("lead_trump", False):
            want_trump = False

        drag_mode = self._determine_trump_drag_mode(context, want_trump)

        trump_opts = [opt for opt in options if opt["is_trump"]]
        off_opts = [opt for opt in options if not opt["is_trump"]]
        trump_opts = self._prioritize_pair_release(trump_opts, context)
        off_opts = self._prioritize_pair_release(off_opts, context)
        trump_opts = self._enforce_primary_constraints(trump_opts, context, is_lead=True)
        off_opts = self._enforce_primary_constraints(off_opts, context, is_lead=True)
        trump_opts = self._refine_options(trump_opts, context, priority="lead-trump")
        off_opts = self._refine_options(off_opts, context, priority="lead-off")
        trump_opts = self._filter_trump_overkill(trump_opts, context, is_lead=True)
        trump_opts = self._filter_trump_leads_by_mode(trump_opts, context, drag_mode)
        context["trump_drag_mode"] = drag_mode
        trump_opts = self._filter_pair_leads_by_stage(trump_opts, context)
        off_opts = self._filter_pair_leads_by_stage(off_opts, context)

        safe_leads = self._prepare_safe_leads(off_opts, context)
        point_leads = self._prepare_point_dump_candidates(off_opts, context)
        decision = self._evaluate_lead_decision(
            context=context,
            trump_opts=trump_opts,
            off_opts=off_opts,
            safe_leads=safe_leads,
            point_leads=point_leads,
            want_trump=want_trump,
        )
        if decision:
            decision_type, choice = decision
            context["decision_type"] = decision_type
            return choice

        if options:
            return options[0]["index"]
        return None

    def _select_winning_option(
        self,
        options: List[Dict[str, Any]],
        prefer_preserve: bool,
        context: Optional[Dict[str, Any]] = None,
        current_best: Optional[Dict[str, Any]] = None,
        minimal: bool = False,
    ) -> int:
        if not options:
            raise ValueError("_select_winning_option requires options")
        lead_suit = context.get("lead_suit") if context else None
        enemy_void = context.get("enemy_void_suits", set()) if context else set()
        trump_status = context.get("trump_status") if context else None
        if lead_suit in enemy_void:
            minimal = False
        if minimal and current_best is not None:
            base_strength = current_best.get("max_strength", -math.inf)
            best = min(
                options,
                key=lambda opt: (
                    max(opt["max_strength"] - base_strength, 0),
                    self._point_penalty(opt),
                    opt["premium_cost"],
                    opt["length"],
                ),
            )
            return best["index"]

        def key(opt):
            penalty = opt["premium_cost"] if prefer_preserve else 0
            penalty += self._point_penalty(opt)
            if lead_suit and lead_suit != "trump" and opt.get("is_trump") and trump_status == "disadvantage":
                penalty += 30
            if lead_suit and lead_suit in enemy_void and opt.get("is_trump"):
                penalty += 10
            return (penalty, -opt["max_strength"], -opt["length"])

        best = min(options, key=key)
        return best["index"]

    def _select_minimal(self, options: List[Dict[str, Any]]) -> int:
        best = min(options, key=lambda opt: (self._point_penalty(opt), opt["max_strength"], opt["length"]))
        return best["index"]

    def _select_smallest(self, options: List[Dict[str, Any]]) -> int:
        return self._select_minimal(options)

    def _select_trump_lead(self, options: List[Dict[str, Any]]) -> int:
        best = max(options, key=lambda opt: (opt["length"], opt["max_strength"] - opt["premium_cost"] - self._point_penalty(opt)))
        return best["index"]

    def _select_offensive_lead(self, options: List[Dict[str, Any]]) -> int:
        best = max(options, key=lambda opt: (opt["length"], -self._point_penalty(opt), -opt["premium_cost"]))
        return best["index"]

    def _select_minimum_point_dump(self, options: List[Dict[str, Any]]) -> int:
        best = min(options, key=lambda opt: (self._point_penalty(opt), opt["max_strength"]))
        return best["index"]

    def _select_low_risk_filler(self, options: List[Dict[str, Any]]) -> int:
        non_point = [opt for opt in options if not opt.get("contains_points")]
        if non_point:
            return self._select_smallest(non_point)
        five_only = [opt for opt in options if self._option_points_subset(opt, {"5"})]
        if five_only:
            return self._select_smallest(five_only)
        return self._select_smallest(options)

    def _point_penalty(self, option: Dict[str, Any]) -> int:
        ranks = self._option_point_ranks(option)
        penalty = option.get("points_value", 0)
        if "0" in ranks or "K" in ranks:
            penalty += 200
        if "5" in ranks:
            penalty += 40
        return penalty

    def _option_point_ranks(self, option: Dict[str, Any]) -> set:
        ranks: set = set()
        for card in option.get("cards", []):
            if len(card) == 2 and card[1] in POINT_CARD_VALUES:
                ranks.add(card[1])
        return ranks

    def _option_points_subset(self, option: Dict[str, Any], allowed: set) -> bool:
        ranks = self._option_point_ranks(option)
        return bool(ranks) and ranks.issubset(allowed)

    def _classify_game_stage(self, cards_seen: int) -> str:
        if cards_seen <= EARLY_CARD_LIMIT:
            return "early"
        if cards_seen <= MID_CARD_LIMIT:
            return "mid"
        return "late"

    def _classify_trump_status(self, trump_count: int) -> str:
        if trump_count > 9:
            return "advantage"
        if trump_count < 9:
            return "disadvantage"
        return "balanced"

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

    def _build_position_info(self, context: Dict[str, Any]) -> Dict[str, Any]:
        order = context.get("current_trick_order") or []
        history = context.get("history") or []
        player_id = context.get("player_id")
        info = {
            "turn_index": len(history),
            "is_last": len(history) >= 3,
            "is_first": len(history) == 0,
        }
        if order and player_id is not None and player_id in order:
            seat_index = order.index(player_id)
            info["seat_index"] = seat_index
            info["is_last"] = seat_index == len(order) - 1 and len(history) >= seat_index
            info["is_first"] = seat_index == 0 and len(history) == 0
        return info

    def _build_opponent_flags(self, context: Dict[str, Any]) -> Dict[str, Any]:
        flags = {
            "void_suits": context.get("enemy_void_suits", set()),
            "dumped_points": False,
            "no_points_seen": True,
        }
        order = context.get("current_trick_order") or []
        history = context.get("history") or []
        player_id = context.get("player_id")
        teammate_id = context.get("teammate_id")
        enemy_ids = set(order)
        if player_id is not None:
            enemy_ids.discard(player_id)
        if teammate_id is not None:
            enemy_ids.discard(teammate_id)
        major = context.get("major", "s")
        level = context.get("level", "2")
        for idx, move in enumerate(history):
            if idx >= len(order):
                break
            seat = order[idx]
            if seat not in enemy_ids:
                continue
            if any(len(card) == 2 and card[1] in POINT_CARD_VALUES for card in move):
                flags["dumped_points"] = True
                flags["no_points_seen"] = False
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
        lead_suit = context.get("lead_suit")
        point_ranks = self._option_point_ranks(option)
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
        if option.get("is_trump") and lead_suit != "trump" and context.get("trump_status") == "disadvantage":
            score += 15
        if (
            option.get("is_trump")
            and context.get("trump_status") == "advantage"
            and lead_suit
            and lead_suit != "trump"
            and lead_suit not in enemy_void
        ):
            score -= 5
        if point_ranks:
            if {"0", "K"} & point_ranks:
                score += 30
            elif "5" in point_ranks:
                score += 10
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

    def _can_ignore_teammate_points(
        self,
        context: Dict[str, Any],
        winning_options: List[Dict[str, Any]],
        points_on_table: int,
    ) -> bool:
        if not winning_options:
            return False
        if context.get("score_state") != "leading":
            return False
        if points_on_table > 5:
            return False
        min_strength = min(opt.get("max_strength", 0) for opt in winning_options)
        return min_strength >= BIG_TRUMP_STRENGTH

    def _prepare_point_dump_candidates(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not options:
            return []
        candidates = [opt for opt in options if opt.get("contains_points")]
        return candidates

    def _select_point_dump(self, options: List[Dict[str, Any]]) -> int:
        if not options:
            raise ValueError("_select_point_dump requires options")
        best = max(
            options,
            key=lambda opt: (
                opt.get("points_value", 0),
                -opt.get("premium_cost", 0),
                -opt.get("max_strength", 0),
            ),
        )
        return best["index"]

    def _filter_trump_overkill(
        self,
        options: List[Dict[str, Any]],
        context: Dict[str, Any],
        is_lead: bool = False,
    ) -> List[Dict[str, Any]]:
        if not options or not is_lead:
            return options
        filtered = [opt for opt in options if not self._is_overkill_trump_lead(opt, context)]
        return filtered if filtered else options

    def _is_overkill_trump_lead(self, option: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not option.get("is_trump"):
            return False
        cards = option.get("cards", [])
        if len(cards) < 2:
            return False
        # Pair of jokers (any combination of jo / Jo)
        joker_count = sum(1 for card in cards if card in ("jo", "Jo"))
        if joker_count >= 2 and len(cards) == 2:
            return True
        # Pair of level cards when leading trump
        level = context.get("level", "2")
        ranks = [card[1] for card in cards if isinstance(card, str) and len(card) == 2]
        if len(cards) == 2 and len(ranks) == 2 and all(rank == level for rank in ranks):
            return True
        return False

    def _estimate_trump_drag_rounds(
        self,
        played_piles: List[List[str]],
        history: List[List[str]],
        major: str,
        level: str,
    ) -> int:
        rounds = 0
        for pile in played_piles:
            if not pile:
                continue
            if self._card_suit(pile[0], major, level) == "trump":
                rounds += 1
        if history and history[0]:
            if self._card_suit(history[0][0], major, level) == "trump":
                rounds += 1
        return rounds

    def _count_trump_cards(self, cards: List[str], major: str, level: str) -> int:
        return sum(1 for card in cards if self._is_trump(card, major, level))

    def _count_high_trump_seen(self, cards: List[str], major: str, level: str) -> int:
        return sum(1 for card in cards if self._is_high_trump(card, major, level))

    def _is_high_trump(self, card: str, major: str, level: str) -> bool:
        if card in ("jo", "Jo"):
            return True
        if len(card) != 2:
            return False
        return card[1] == level

    def _prepare_safe_fillers(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not options:
            return []
        safe = [opt for opt in options if not opt.get("contains_points")]
        safe = self._prioritize_pair_release(safe, context)
        safe = self._enforce_primary_constraints(safe, context)
        safe = self._refine_options(safe, context, priority="safe")
        return safe

    def _prepare_safe_leads(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not options:
            return []
        safe = [opt for opt in options if not opt.get("contains_points")]
        safe = self._prioritize_pair_release(safe, context)
        safe = self._enforce_primary_constraints(safe, context, is_lead=True)
        safe = self._refine_options(safe, context, priority="lead-safe")
        return safe

    def _determine_trump_drag_mode(self, context: Dict[str, Any], want_trump: bool) -> str:
        if context.get("no_trump_mark"):
            return "none"
        rounds = context.get("trump_drag_rounds", 0)
        half_spent = context.get("trump_half_spent", False)
        low_rounds = context.get("trump_low_rounds_left", False)
        high_seen = context.get("high_trump_seen", 0)
        if want_trump and rounds < 2:
            return "assertive"
        if want_trump and not half_spent:
            return "assertive"
        if rounds > 5 and high_seen >= 4:
            return "light"
        if low_rounds or half_spent:
            return "light"
        return "free"

    def _filter_trump_leads_by_mode(
        self,
        options: List[Dict[str, Any]],
        context: Dict[str, Any],
        mode: str,
    ) -> List[Dict[str, Any]]:
        if not options or mode in {"none", "free"}:
            return options
        major = context.get("major", "s")
        level = context.get("level", "2")
        filtered: List[Dict[str, Any]] = []
        for opt in options:
            if not opt.get("is_trump"):
                filtered.append(opt)
                continue
            if mode == "assertive":
                if self._contains_joker(opt):
                    continue
                if not opt.get("is_pair") and self._option_min_rank_value(opt, major, level) < RANK_ORDER[ASSERTIVE_MIN_RANK]:
                    continue
            if mode == "light":
                if not opt.get("is_pair") and self._option_min_rank_value(opt, major, level) < RANK_ORDER[LIGHT_MIN_RANK]:
                    continue
            filtered.append(opt)
        return filtered if filtered else options

    def _contains_joker(self, option: Dict[str, Any]) -> bool:
        return any(card in ("jo", "Jo") for card in option.get("cards", []))

    def _option_min_rank_value(self, option: Dict[str, Any], major: str, level: str) -> int:
        min_value = math.inf
        for card in option.get("cards", []):
            if card in ("jo", "Jo"):
                min_value = min(min_value, 1000)
                continue
            if len(card) != 2:
                continue
            if self._is_trump(card, major, level):
                min_value = min(min_value, RANK_ORDER.get(card[1], -1))
        return min_value if min_value != math.inf else -1

    def _filter_pair_leads_by_stage(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not options:
            return []
        position = context.get("position_info", {})
        if position.get("is_last"):
            return options
        stage = context.get("stage", "early")
        filtered: List[Dict[str, Any]] = []
        for opt in options:
            if not self._is_pair_lead_blocked(opt, stage):
                filtered.append(opt)
        return filtered if filtered else options

    def _is_pair_lead_blocked(self, option: Dict[str, Any], stage: str) -> bool:
        if not option.get("is_pair"):
            return False
        rank = option.get("pair_rank")
        if rank is None:
            return False
        rank_value = RANK_ORDER.get(rank, -1)
        five_value = RANK_ORDER["5"]
        ten_value = RANK_ORDER["0"]
        if rank_value < five_value:
            return True
        if stage == "early" and rank_value < ten_value:
            return True
        return False

    def _apply_trump_takeover_filters(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not options:
            return []
        lead_suit = context.get("lead_suit")
        if not lead_suit or lead_suit == "trump":
            return options
        trump_status = context.get("trump_status", "balanced")
        points_on_table = context.get("points_on_table", 0)
        score_state = context.get("score_state", "balanced")
        filtered: List[Dict[str, Any]] = []
        for opt in options:
            if not self._is_trump_takeover(opt, lead_suit):
                filtered.append(opt)
                continue
            if trump_status == "disadvantage":
                allow = points_on_table > 10 or (points_on_table > 5 and score_state == "trailing")
                if allow:
                    filtered.append(opt)
                continue
            filtered.append(opt)
        return filtered

    def _is_trump_takeover(self, option: Dict[str, Any], lead_suit: Optional[str]) -> bool:
        if not option.get("is_trump"):
            return False
        if not lead_suit:
            return False
        return lead_suit != "trump"

    def _needs_minimal_control(self, context: Dict[str, Any]) -> bool:
        points_on_table = context.get("points_on_table", 0)
        position_info = context.get("position_info", {})
        enemy_void = context.get("enemy_void_suits", set())
        lead_suit = context.get("lead_suit")
        minimal = False
        if position_info.get("is_last") and points_on_table > 0:
            minimal = True
        if (
            context.get("trump_status") == "advantage"
            and lead_suit
            and lead_suit != "trump"
            and lead_suit not in enemy_void
        ):
            minimal = True
        if lead_suit in enemy_void:
            minimal = False
        return minimal

    def _teammate_trump_lead_pressure(self, context: Dict[str, Any]) -> bool:
        order = context.get("current_trick_order") or []
        history = context.get("history") or []
        teammate_id = context.get("teammate_id")
        if not order or not history or teammate_id is None:
            return False
        if order[0] != teammate_id:
            return False
        lead_move = history[0]
        if not lead_move:
            return False
        if not self._is_normal_trump_play(lead_move, context):
            return False
        position_info = context.get("position_info", {})
        if position_info.get("turn_index") != 2:
            return False
        return context.get("hand_trump_count", 0) > 5

    def _is_normal_trump_play(self, cards: List[str], context: Dict[str, Any]) -> bool:
        major = context.get("major", "s")
        level = context.get("level", "2")
        for card in cards:
            if card in ("jo", "Jo"):
                return False
            if len(card) == 2 and card[1] == level:
                return False
            if not self._is_trump(card, major, level):
                return False
        return True

    def _should_take_on_strength(
        self,
        context: Dict[str, Any],
        winning: List[Dict[str, Any]],
        opponent_winning: bool,
    ) -> bool:
        if not winning or not opponent_winning:
            return False
        stage = context.get("stage", "early")
        if stage == "late":
            return False
        if context.get("points_on_table", 0) > 20:
            return False
        threshold = self._takeover_strength_threshold(context)
        best_strength = max(opt.get("max_strength", -math.inf) for opt in winning)
        return best_strength >= threshold

    def _takeover_strength_threshold(self, context: Dict[str, Any]) -> float:
        lead_suit = context.get("lead_suit")
        if lead_suit == "trump":
            return RANK_ORDER["J"] + 120  # roughly weaker trump strength
        return RANK_ORDER["A"]

    def _should_force_special_takeover(self, context: Dict[str, Any], winning: List[Dict[str, Any]]) -> bool:
        if not winning:
            return False
        stage = context.get("stage", "early")
        specials = [opt for opt in winning if self._is_special_take_option(opt, context, stage)]
        if not specials:
            return False
        if self._teammate_led_high_guard(context):
            return False
        return True

    def _is_special_take_option(self, option: Dict[str, Any], context: Dict[str, Any], stage: str) -> bool:
        length = option.get("length", 0)
        cards = option.get("cards", [])
        level = context.get("level", "2")
        if length >= 4:
            return True
        if length == 2 and set(cards) == {"jo", "Jo"}:
            return stage != "early"
        if option.get("is_pair"):
            rank = option.get("pair_rank")
            if rank in {"A", level}:
                return True
        return False

    def _teammate_led_high_guard(self, context: Dict[str, Any]) -> bool:
        order = context.get("current_trick_order") or []
        history = context.get("history") or []
        teammate_id = context.get("teammate_id")
        if not order or not history or teammate_id is None:
            return False
        if order[0] != teammate_id:
            return False
        lead_move = history[0]
        if not lead_move:
            return False
        major = context.get("major", "s")
        level = context.get("level", "2")
        ranks = [card[1] for card in lead_move if isinstance(card, str) and len(card) == 2]
        if ranks.count("A") >= 2:
            return True
        return any(self._is_guard_card(card, major, level) for card in lead_move)

    def _is_guard_card(self, card: str, major: str, level: str) -> bool:
        if card in ("jo", "Jo"):
            return True
        if len(card) != 2:
            return False
        suit, rank = card[0], card[1]
        if rank == level:
            return True
        if rank == "A" and suit == major:
            return True
        return False

    def _evaluate_follow_decision(
        self,
        context: Dict[str, Any],
        winning: List[Dict[str, Any]],
        fillers: List[Dict[str, Any]],
        safe_options: List[Dict[str, Any]],
        dump_candidates: List[Dict[str, Any]],
        need_takeover: bool,
        teammate_winning: bool,
        opponent_winning: bool,
        current_best: Optional[Dict[str, Any]],
        force_take_override: bool = False,
    ) -> Optional[Tuple[str, int]]:
        minimal_control = self._needs_minimal_control(context)
        if force_take_override:
            teammate_winning = False
        if winning and (need_takeover or self._should_force_future_control(context, winning)):
            prefer_preserve = need_takeover and not context.get("meta", {}).get("apply_pressure", False)
            return "control", self._select_winning_option(
                winning,
                prefer_preserve=prefer_preserve,
                context=context,
                current_best=current_best,
                minimal=minimal_control,
            )

        if teammate_winning and self._should_stick_with_teammate(context, winning, dump_candidates):
            return "stick", self._select_point_dump(dump_candidates)

        if opponent_winning and self._should_stick_against_opponent(context, dump_candidates, current_best):
            return "stick", self._select_point_dump(dump_candidates)

        if safe_options:
            return "follow_small", self._select_smallest(safe_options)

        if fillers:
            if teammate_winning and dump_candidates:
                return "stick", self._select_point_dump(dump_candidates)
            return "follow_small", self._select_low_risk_filler(fillers)

        if winning:
            return "control", self._select_winning_option(
                winning,
                prefer_preserve=False,
                context=context,
                current_best=current_best,
                minimal=minimal_control,
            )

        return None

    def _evaluate_lead_decision(
        self,
        context: Dict[str, Any],
        trump_opts: List[Dict[str, Any]],
        off_opts: List[Dict[str, Any]],
        safe_leads: List[Dict[str, Any]],
        point_leads: List[Dict[str, Any]],
        want_trump: bool,
    ) -> Optional[Tuple[str, int]]:
        if want_trump and trump_opts:
            return "control", self._select_trump_lead(trump_opts)

        if point_leads and self._should_stick_on_lead(context, want_trump):
            return "stick", self._select_point_dump(point_leads)

        if safe_leads:
            return "follow_small", self._select_offensive_lead(safe_leads)

        if off_opts:
            if context.get("meta", {}).get("allow_point_dump", False):
                return "stick", self._select_point_dump(off_opts)
            return "follow_small", self._select_offensive_lead(off_opts)

        if trump_opts:
            return "control", self._select_trump_lead(trump_opts)

        return None

    def _should_force_future_control(self, context: Dict[str, Any], winning: List[Dict[str, Any]]) -> bool:
        if not winning:
            return False
        if context.get("points_on_table", 0) > 0:
            return False
        score_state = context.get("score_state", "balanced")
        if score_state != "trailing":
            return False
        if context.get("meta", {}).get("advantage", False):
            return False
        return self._has_powerful_future_play(context)

    def _has_powerful_future_play(self, context: Dict[str, Any]) -> bool:
        deck_cards = context.get("deck", [])
        if not deck_cards:
            return False
        major = context.get("major", "s")
        level = context.get("level", "2")
        strengths = sorted((self._card_strength(card, major, level) for card in deck_cards), reverse=True)
        if strengths and strengths[0] >= BIG_TRUMP_STRENGTH:
            return True
        pair_tracker: Dict[Tuple[str, str], int] = {}
        for card in deck_cards:
            if len(card) != 2:
                continue
            suit, rank = card[0], card[1]
            key = (suit, rank)
            pair_tracker[key] = pair_tracker.get(key, 0) + 1
            if pair_tracker[key] >= 2 and (suit == major or rank == level or rank in {"A", "K"}):
                return True
        return False

    def _should_stick_with_teammate(
        self,
        context: Dict[str, Any],
        winning: List[Dict[str, Any]],
        dump_candidates: List[Dict[str, Any]],
    ) -> bool:
        if not dump_candidates:
            return False
        teammate_supreme = not winning
        if teammate_supreme:
            return True
        has_five = any(self._option_contains_rank(opt, "5") for opt in dump_candidates)
        if not has_five:
            return False
        if context.get("score_state") == "leading":
            # 70%概率在领先且持有5分时贴给领先中的队友
            return self._pseudo_random_chance(context, "stick_teammate") < 0.7
        return False

    def _should_stick_against_opponent(
        self,
        context: Dict[str, Any],
        dump_candidates: List[Dict[str, Any]],
        current_best: Optional[Dict[str, Any]],
    ) -> bool:
        if not dump_candidates:
            return False
        if self._teammate_can_cut(context):
            return True
        opponent_not_supreme = self._opponent_play_not_supreme(context, current_best)
        if not opponent_not_supreme:
            return False
        has_five = any(self._option_contains_rank(opt, "5") for opt in dump_candidates)
        if not has_five:
            return False
        if context.get("score_state") == "leading":
            # 70%概率在领先且仅有5分时小赌一把
            return self._pseudo_random_chance(context, "stick_opponent") < 0.7
        return False

    def _should_stick_on_lead(self, context: Dict[str, Any], want_trump: bool) -> bool:
        meta = context.get("meta", {})
        if meta.get("allow_point_dump", False):
            return True
        score_state = context.get("score_state", "balanced")
        if score_state == "leading" and not want_trump:
            return True
        return False

    def _teammate_can_cut(self, context: Dict[str, Any]) -> bool:
        flags = context.get("teammate_flags", {})
        if flags.get("follow_state") == "void":
            return True
        history = context.get("history") or []
        major = context.get("major", "s")
        level = context.get("level", "2")
        lead_suit = self._lead_suit_from_history(history, major, level)
        teammate_voids = context.get("teammate_void_suits", set())
        return lead_suit in teammate_voids if lead_suit else False

    def _opponent_play_not_supreme(self, context: Dict[str, Any], current_best: Optional[Dict[str, Any]]) -> bool:
        if not current_best:
            return False
        deck_cards = context.get("deck", [])
        if not deck_cards:
            return False
        major = context.get("major", "s")
        level = context.get("level", "2")
        strongest_hand = max((self._card_strength(card, major, level) for card in deck_cards), default=-math.inf)
        return strongest_hand > current_best.get("max_strength", -math.inf)

    def _option_contains_rank(self, option: Dict[str, Any], rank: str) -> bool:
        for card in option.get("cards", []):
            if len(card) == 2 and card[1] == rank:
                return True
        return False

    def _pseudo_random_chance(self, context: Dict[str, Any], salt: str) -> float:
        seed_parts = (
            context.get("player_id"),
            context.get("total_points_played", 0),
            context.get("points_on_table", 0),
            context.get("scenario_key"),
            salt,
        )
        seed_str = "|".join(str(part) for part in seed_parts)
        digest = hashlib.md5(seed_str.encode("ascii", "ignore")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF
