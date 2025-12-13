import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

try:
    from data.wrapper import cardWrapper
except ImportError:  # Fallback when package-style import is unavailable
    from wrapper import cardWrapper

POINT_CARD_VALUES = {"5": 5, "0": 10, "K": 10}
RANK_ORDER = {rank: idx for idx, rank in enumerate(["2", "3", "4", "5", "6", "7", "8", "9", "0", "J", "Q", "K", "A"])}
EARLY_CARD_LIMIT = 20
MID_CARD_LIMIT = 70
BIG_TRUMP_STRENGTH = 950
TRUMP_TOTAL_ESTIMATE = 24
ASSERTIVE_MIN_RANK = "J"
LIGHT_MIN_RANK = "7"
SAFE_LEAD_MIN_RANK = "Q"
VALUATION_ADV_THRESHOLD = 1.15
VALUATION_DISADV_THRESHOLD = 0.85
LATE_LOW_POINT_THRESHOLD = 40
MICRO_VALUE_ORDER = {"K": 9, "A": 8, "0": 7, "Q": 6, "J": 5, "5": 4, "9": 3, "8": 2, "7": 1, "6": 0, "4": 0, "3": 0, "2": 0}
STICK_PRIORITY = ("0", "K", "5")
PEER_SAFE_LEAD_MIN_VALUE = RANK_ORDER["A"]
SMALL_TRUMP_SIGNAL_THRESHOLD = BIG_TRUMP_STRENGTH - 150
STRONG_OFFSUIT_PRESSURE = RANK_ORDER["Q"]


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
        hand_suit_counts = self._count_suit_distribution(deck_cards, major_suit, level_rank)
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
            "hand_suit_counts": hand_suit_counts,
        }
        context = self._prepare_context(context)
        option_features = context["options"]
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

    def _prepare_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["options"] = self._apply_stage_constraints(context)
        context["teammate_flags"] = self._build_teammate_flags(context)
        context["teammate_void_suits"], context["enemy_void_suits"] = self._split_void_map(context)
        context["position_info"] = self._build_position_info(context)
        context["opponent_flags"] = self._build_opponent_flags(context)
        context["valuation"] = self._compute_position_valuation(context)
        context["advantage_state"] = context["valuation"]["state"]
        context["lead_bias"] = context["valuation"]["plan_bias"]
        context["teammate_small_trump_signal"] = self._detect_teammate_small_trump_signal(context)
        context["strong_peer_mark"] = self._resolve_strong_peer_mark(context)
        context["teammate_point_trump_lead"] = self._detect_teammate_point_trump_lead(context)
        context["should_force_trump_open"] = self._should_force_trump_open(context)
        return context

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

    def _count_suit_distribution(self, cards: List[str], major: str, level: str) -> Dict[str, int]:
        counts = {"s": 0, "h": 0, "c": 0, "d": 0}
        for card in cards:
            suit = self._card_suit(card, major, level)
            if suit and suit != "trump":
                counts[suit] = counts.get(suit, 0) + 1
        return counts

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
        options = context.get("options") or []
        if not options:
            return None
        major = context.get("major")
        level = context.get("level")
        history = context.get("history") or []
        order = context.get("current_trick_order")
        current_best, best_owner = self._current_best_with_owner(history, major, level, order)
        if current_best is None:
            return self._lead_policy(context)

        buckets = self._build_follow_buckets(context, current_best)
        plan = self._choose_follow_plan(context, buckets, best_owner)
        if plan is None:
            plan = {"bucket": "small", "mode": "safety"}

        candidates = buckets.get(plan["bucket"], [])
        candidates = self._apply_micro_follow_filters(plan, candidates, context, current_best)
        if not candidates:
            for fallback in ("control", "stick_sure", "small"):
                fallback_plan = {"bucket": fallback, "mode": "safety"}
                cand = buckets.get(fallback, [])
                cand = self._apply_micro_follow_filters(fallback_plan, cand, context, current_best)
                if cand:
                    plan = fallback_plan
                    candidates = cand
                    break
        if not candidates:
            return options[0]["index"]
        return self._select_follow_option(plan, candidates, context, current_best)

    def _build_follow_buckets(self, context: Dict[str, Any], current_best: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        options = context.get("options") or []
        winning = [opt for opt in options if self._beats(opt, current_best)]
        fillers = [opt for opt in options if opt not in winning]
        winning = self._apply_trump_takeover_filters(winning, context)
        winning = self._prioritize_pair_release(winning, context)
        winning = self._enforce_primary_constraints(winning, context)
        winning = self._refine_options(winning, context, priority="winning")
        fillers = self._prioritize_pair_release(fillers, context)
        fillers = self._enforce_primary_constraints(fillers, context)
        fillers = self._refine_options(fillers, context, priority="filler")
        safe_fillers = self._prepare_safe_fillers(fillers, context)
        dump_candidates = self._prepare_point_dump_candidates(fillers, context)
        probing = [opt for opt in dump_candidates if opt.get("length") == 1 and self._option_point_ranks(opt) == {"5"}]
        small = safe_fillers or fillers
        return {
            "control": winning,
            "stick_sure": dump_candidates,
            "stick_probe": probing,
            "small": small,
        }

    def _choose_follow_plan(
        self,
        context: Dict[str, Any],
        buckets: Dict[str, List[Dict[str, Any]]],
        best_owner: Optional[int],
    ) -> Optional[Dict[str, str]]:
        teammate_id = context.get("teammate_id")
        teammate_winning = teammate_id is not None and best_owner == teammate_id
        context["teammate_winning"] = teammate_winning
        seat_index = self._seat_index(context)
        points_on_table = context.get("points_on_table", 0)
        teammate_signal = context.get("teammate_small_trump_signal", False)

        if (
            seat_index == 2
            and context.get("teammate_point_trump_lead")
            and buckets.get("control")
        ):
            context["needs_collab_point_trump"] = True
            return {"bucket": "control", "mode": "collab"}

        if teammate_signal and buckets.get("control"):
            context["needs_aggressive_contest"] = True
            return {"bucket": "control", "mode": "stage"}

        if teammate_winning and buckets.get("stick_sure"):
            return {"bucket": "stick_sure", "mode": "deterministic"}
        if teammate_winning and self._should_reinforce_teammate_trick(context) and buckets.get("control"):
            context["needs_reinforce_teammate"] = True
            return {"bucket": "control", "mode": "stage"}
        context["needs_aggressive_contest"] = not teammate_winning and seat_index == 1
        if not teammate_winning and buckets.get("control"):
            mode = "points" if points_on_table > 0 else "stage"
            return {"bucket": "control", "mode": mode}
        if (
            not teammate_winning
            and seat_index == 1
            and buckets.get("stick_probe")
            and self._teammate_behind(context)
        ):
            return {"bucket": "stick_probe", "mode": "probing"}
        if buckets.get("small"):
            return {"bucket": "small", "mode": "safety"}
        if buckets.get("control"):
            return {"bucket": "control", "mode": "stage"}
        if buckets.get("stick_sure"):
            return {"bucket": "stick_sure", "mode": "deterministic"}
        return None

    def _apply_micro_follow_filters(
        self,
        plan: Dict[str, str],
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any],
        current_best: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        bucket = plan["bucket"]
        if bucket == "control":
            return self._filter_control_candidates(candidates, context, plan.get("mode"), current_best)
        if bucket == "stick_sure":
            return self._filter_stick_candidates(candidates, context, deterministic=True)
        if bucket == "stick_probe":
            return self._filter_stick_candidates(candidates, context, deterministic=False)
        return self._filter_small_follow(candidates, context)

    def _select_follow_option(
        self,
        plan: Dict[str, str],
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any],
        current_best: Optional[Dict[str, Any]],
    ) -> int:
        bucket = plan["bucket"]
        if bucket == "control":
            mode = plan.get("mode")
            if mode == "collab":
                return self._select_collab_trump_kill(candidates, context)
            prefer_preserve = mode == "stage"
            minimal = mode == "points" or self._seat_index(context) == 3
            return self._select_winning_option(
                candidates,
                prefer_preserve=prefer_preserve,
                context=context,
                current_best=current_best,
                minimal=minimal,
            )
        if bucket in {"stick_sure", "stick_probe"}:
            return self._select_point_dump(candidates)
        return self._select_low_risk_filler(candidates, context)

    def _filter_control_candidates(
        self,
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any],
        mode: Optional[str],
        current_best: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        if mode == "collab":
            return candidates
        aggressive = context.get("needs_aggressive_contest") or context.get("teammate_small_trump_signal")
        if aggressive:
            low_cost = [opt for opt in candidates if opt.get("premium_cost", 0) < 15 and not self._contains_joker(opt)]
            if low_cost:
                candidates = low_cost
        seat_index = self._seat_index(context)
        if seat_index == 3:
            return sorted(candidates, key=lambda opt: (opt.get("max_strength", 0), self._point_penalty(opt)))
        if seat_index == 2:
            return sorted(candidates, key=lambda opt: (opt.get("premium_cost", 0), -opt.get("max_strength", 0)))
        threshold = BIG_TRUMP_STRENGTH if mode == "stage" else BIG_TRUMP_STRENGTH + 60
        filtered = [opt for opt in candidates if opt.get("max_strength", 0) < threshold]
        return filtered if filtered else candidates

    def _filter_stick_candidates(self, candidates: List[Dict[str, Any]], context: Dict[str, Any], deterministic: bool) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        allowed = list(STICK_PRIORITY) if deterministic else ["5"]
        filtered: List[Dict[str, Any]] = []
        for opt in candidates:
            if opt.get("length") != 1:
                continue
            if opt.get("is_trump"):
                continue
            cards = opt.get("cards") or []
            if not cards:
                continue
            card = cards[0]
            if len(card) != 2:
                continue
            rank = card[1]
            if rank not in allowed:
                continue
            filtered.append(opt)
        filtered.sort(key=lambda opt: allowed.index(opt["cards"][0][1]))
        lead_suit = context.get("lead_suit")
        enemy_void = context.get("enemy_void_suits", set())
        if lead_suit and lead_suit in enemy_void:
            filtered = [opt for opt in filtered if not opt.get("contains_points")]
        return filtered

    def _filter_small_follow(self, candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        stage = context.get("stage", "early")
        threshold = 4 if stage != "late" else 3
        filtered = [opt for opt in candidates if not opt.get("contains_points") and self._max_micro_value(opt) <= threshold]
        if filtered:
            return filtered
        filtered = [opt for opt in candidates if self._max_micro_value(opt) <= threshold + 1]
        return filtered if filtered else candidates

    def _max_micro_value(self, option: Dict[str, Any]) -> int:
        values = []
        for card in option.get("cards", []):
            if card in ("jo", "Jo"):
                values.append(10)
                continue
            if len(card) == 2:
                values.append(MICRO_VALUE_ORDER.get(card[1], 0))
        return max(values) if values else 0

    def _teammate_behind(self, context: Dict[str, Any]) -> bool:
        order = context.get("current_trick_order") or []
        teammate_id = context.get("teammate_id")
        player_id = context.get("player_id")
        if teammate_id is None or player_id is None:
            return False
        if teammate_id not in order or player_id not in order:
            return False
        return order.index(teammate_id) > order.index(player_id)

    def _should_reinforce_teammate_trick(self, context: Dict[str, Any]) -> bool:
        if not context.get("teammate_winning"):
            return False
        points_on_table = context.get("points_on_table", 0)
        if points_on_table < 10:
            return False
        if not self._has_follow_suit_option(context) and self._has_trump_option(context):
            return True
        best_follow = self._best_follow_strength(context)
        return best_follow >= RANK_ORDER.get("K", 11)

    def _has_follow_suit_option(self, context: Dict[str, Any]) -> bool:
        lead = context.get("lead_suit")
        if not lead or lead == "trump":
            return False
        for opt in context.get("options") or []:
            if opt.get("primary_suit") == lead and not opt.get("is_trump"):
                return True
        return False

    def _best_follow_strength(self, context: Dict[str, Any]) -> int:
        lead = context.get("lead_suit")
        if not lead or lead == "trump":
            return -1
        strength = -1
        for opt in context.get("options") or []:
            if opt.get("primary_suit") == lead and not opt.get("is_trump"):
                strength = max(strength, int(opt.get("max_strength", -1)))
        return strength

    def _has_trump_option(self, context: Dict[str, Any]) -> bool:
        return any(opt.get("is_trump") for opt in (context.get("options") or []))

    def _seat_index(self, context: Dict[str, Any]) -> int:
        info = context.get("position_info", {})
        if "seat_index" in info and info["seat_index"] is not None:
            return int(info["seat_index"])
        return len(context.get("history") or [])

    def _lead_policy(self, context: Dict[str, Any]) -> Optional[int]:
        options = context.get("options") or []
        if not options:
            return None
        buckets = self._build_lead_buckets(context)
        if context.get("should_force_trump_open"):
            forced = self._select_forced_trump_open(buckets, context)
            if forced is not None:
                return forced
        sequence = self._choose_lead_bucket_sequence(context, buckets)
        choice = self._pick_lead_from_sequence(sequence, buckets, context)
        if choice is not None:
            return choice
        return options[0]["index"]

    def _build_lead_buckets(self, context: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        options = context.get("options") or []
        trump_opts = [opt for opt in options if opt.get("is_trump")]
        off_opts = [opt for opt in options if not opt.get("is_trump")]
        trump_opts = self._prioritize_pair_release(trump_opts, context)
        off_opts = self._prioritize_pair_release(off_opts, context)
        trump_opts = self._enforce_primary_constraints(trump_opts, context, is_lead=True)
        off_opts = self._enforce_primary_constraints(off_opts, context, is_lead=True)
        trump_opts = self._filter_trump_overkill(trump_opts, context, is_lead=True)
        off_opts = self._filter_small_leads(off_opts, context)
        drag_mode = self._determine_trump_drag_mode(context, context.get("lead_bias") == "trump")
        trump_opts = self._filter_trump_leads_by_mode(trump_opts, context, drag_mode)
        context["trump_drag_mode"] = drag_mode
        trump_opts = self._filter_pair_leads_by_stage(trump_opts, context)
        off_opts = self._filter_pair_leads_by_stage(off_opts, context)
        trump_opts = self._refine_options(trump_opts, context, priority="lead-trump")
        trump_opts, premium_trumps = self._split_premium_trumps(trump_opts, context)
        balanced_off = self._refine_options(off_opts, context, priority="lead-off")
        safe_leads = self._prepare_safe_leads(off_opts, context)
        void_leads = self._prepare_void_shedding_leads(off_opts, context)
        point_leads = self._prepare_point_dump_candidates(off_opts, context)
        allow_point_risk = context.get("strong_peer_mark", False)
        point_cap = 10 if allow_point_risk else 5
        point_leads = [opt for opt in point_leads if opt.get("points_value", 0) <= point_cap]
        minimal_trump = [opt for opt in (trump_opts + premium_trumps) if opt.get("length") == 1]
        context["has_strong_offsuit_lead"] = any(
            opt.get("max_strength", 0) >= STRONG_OFFSUIT_PRESSURE for opt in balanced_off or []
        )
        context["needs_small_trump_pull"] = not context.get("has_strong_offsuit_lead") and context.get("hand_trump_count", 0) >= 2
        context["allow_point_leads"] = allow_point_risk
        return {
            "trump_control": trump_opts,
            "off_balance": balanced_off,
            "safe": safe_leads,
            "void": void_leads,
            "points": point_leads,
            "minimal_trump": minimal_trump or trump_opts or premium_trumps,
        }

    def _split_premium_trumps(
        self,
        trumps: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not trumps:
            return [], []
        premium: List[Dict[str, Any]] = []
        regular: List[Dict[str, Any]] = []
        for opt in trumps:
            if self._is_premium_trump(opt, context):
                premium.append(opt)
            else:
                regular.append(opt)
        if regular:
            return regular, premium
        return trumps, []

    def _is_premium_trump(self, option: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not option.get("is_trump"):
            return False
        level = context.get("level", "2")
        for card in option.get("cards", []):
            if card in ("jo", "Jo"):
                return True
            if len(card) == 2 and card[1] == level:
                return True
        return False

    def _choose_lead_bucket_sequence(
        self,
        context: Dict[str, Any],
        buckets: Dict[str, List[Dict[str, Any]]],
    ) -> List[str]:
        role = context.get("role", "farmer")
        state = context.get("advantage_state", "balanced")
        bias = context.get("lead_bias", "balanced")
        sequence: List[str]
        if context.get("position_info", {}).get("is_first") and not context.get("played"):
            if context.get("should_force_trump_open"):
                sequence = ["trump_control", "minimal_trump", "safe", "off_balance", "void", "points"]
            else:
                sequence = ["off_balance", "safe", "void", "minimal_trump", "trump_control", "points"]
            return self._prioritize_small_trump_pull(sequence, context)
        if role == "banker":
            if state == "advantage":
                if bias == "trump":
                    sequence = ["trump_control", "points", "void", "safe", "off_balance", "minimal_trump"]
                elif bias == "offsuit":
                    sequence = ["void", "safe", "points", "off_balance", "trump_control", "minimal_trump"]
                else:
                    sequence = ["trump_control", "points", "safe", "off_balance", "void", "minimal_trump"]
            elif bias == "offsuit":
                sequence = ["safe", "void", "off_balance", "minimal_trump", "points", "trump_control"]
            else:
                sequence = ["safe", "points", "off_balance", "minimal_trump", "void", "trump_control"]
        else:
            if state == "advantage":
                if bias == "trump":
                    sequence = ["trump_control", "off_balance", "points", "safe", "void", "minimal_trump"]
                else:
                    sequence = ["off_balance", "void", "points", "safe", "trump_control", "minimal_trump"]
            else:
                sequence = ["safe", "void", "off_balance", "minimal_trump", "points", "trump_control"]
        return self._prioritize_small_trump_pull(sequence, context)

    def _prioritize_small_trump_pull(self, sequence: List[str], context: Dict[str, Any]) -> List[str]:
        if not context.get("needs_small_trump_pull"):
            return sequence
        seq = list(sequence)
        if "minimal_trump" in seq:
            idx = seq.index("minimal_trump")
            if idx > 1:
                seq.insert(1, seq.pop(idx))
        else:
            seq.insert(1, "minimal_trump")
        return seq

    def _pick_lead_from_sequence(
        self,
        sequence: List[str],
        buckets: Dict[str, List[Dict[str, Any]]],
        context: Dict[str, Any],
    ) -> Optional[int]:
        blocked_points: Optional[List[Dict[str, Any]]] = None
        for bucket_name in sequence:
            candidates = buckets.get(bucket_name) or []
            candidates = self._apply_micro_lead_filters(bucket_name, candidates, context)
            if not candidates:
                continue
            if bucket_name == "points" and not context.get("allow_point_leads"):
                if blocked_points is None:
                    blocked_points = candidates
                continue
            choice = self._select_lead_from_bucket(bucket_name, candidates, context)
            if choice is not None:
                return choice
        if blocked_points:
            return self._select_lead_from_bucket("points", blocked_points, context)
        return None

    def _select_forced_trump_open(
        self,
        buckets: Dict[str, List[Dict[str, Any]]],
        context: Dict[str, Any],
    ) -> Optional[int]:
        pool: List[Dict[str, Any]] = []
        for key in ("trump_control", "minimal_trump"):
            pool.extend(buckets.get(key) or [])
        candidates = [opt for opt in pool if opt.get("is_trump")]
        if not candidates:
            return None
        level = context.get("level", "2")
        major = context.get("major", "s")
        level_opts = [opt for opt in candidates if self._option_contains_rank(opt, level)]
        if level_opts:
            best = max(level_opts, key=lambda opt: opt.get("max_strength", 0))
            return best["index"]
        joker_opts = [opt for opt in candidates if self._contains_joker(opt)]
        if joker_opts:
            best = max(joker_opts, key=lambda opt: opt.get("max_strength", 0))
            return best["index"]
        strong_major = [
            opt
            for opt in candidates
            if any(len(card) == 2 and card[0] == major and RANK_ORDER.get(card[1], -1) >= RANK_ORDER["Q"] for card in opt.get("cards", []))
        ]
        if strong_major:
            best = max(strong_major, key=lambda opt: opt.get("max_strength", 0))
            return best["index"]
        best = self._select_trump_lead(candidates)
        return best

    def _apply_micro_lead_filters(
        self,
        bucket_name: str,
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        filtered = candidates
        stage = context.get("stage", "early")
        if bucket_name == "trump_control":
            if stage == "late":
                filtered = [opt for opt in candidates if opt.get("max_strength", 0) < BIG_TRUMP_STRENGTH]
                filtered = filtered if filtered else candidates
            if not context.get("strong_peer_mark"):
                no_points = [opt for opt in filtered if not opt.get("contains_points")]
                filtered = no_points if no_points else filtered
        elif bucket_name == "points":
            prioritized = []
            for rank in STICK_PRIORITY:
                prioritized.extend(
                    [
                        opt
                        for opt in candidates
                        if opt.get("length") == 1 and opt.get("cards") and len(opt["cards"][0]) == 2 and opt["cards"][0][1] == rank
                    ]
                )
            filtered = prioritized if prioritized else candidates
        elif bucket_name in {"void", "safe"}:
            filtered = [opt for opt in candidates if not opt.get("contains_points")]
            filtered = filtered if filtered else candidates
        elif bucket_name == "minimal_trump":
            filtered = [opt for opt in candidates if opt.get("length") == 1]
            filtered = filtered if filtered else candidates
        return self._filter_peer_incompatible_leads(bucket_name, filtered, context)

    def _filter_peer_incompatible_leads(
        self,
        bucket_name: str,
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        teammate_void = context.get("teammate_void_suits", set())
        enemy_void = context.get("enemy_void_suits", set())
        if not teammate_void and not enemy_void:
            return candidates
        filtered: List[Dict[str, Any]] = []
        for opt in candidates:
            if self._lead_blocked_by_peers(opt, bucket_name, teammate_void, enemy_void):
                continue
            filtered.append(opt)
        return filtered if filtered else candidates

    def _lead_blocked_by_peers(
        self,
        option: Dict[str, Any],
        bucket_name: str,
        teammate_void: set,
        enemy_void: set,
    ) -> bool:
        suit = option.get("primary_suit")
        if not suit or suit == "trump":
            return False
        if suit in teammate_void:
            return True
        if suit in enemy_void:
            if bucket_name == "points" or option.get("contains_points"):
                return True
            strength = option.get("max_strength", -math.inf)
            if strength < PEER_SAFE_LEAD_MIN_VALUE:
                return True
        return False

    def _select_lead_from_bucket(
        self,
        bucket_name: str,
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[int]:
        if not candidates:
            return None
        if bucket_name == "trump_control":
            return self._select_trump_lead(candidates)
        if bucket_name == "points":
            return self._select_point_dump(candidates)
        if bucket_name in {"safe", "void", "off_balance"}:
            return self._select_offensive_lead(bucket_name, candidates, context)
        if bucket_name == "minimal_trump":
            return self._select_smallest(candidates)
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

    def _select_collab_trump_kill(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> int:
        if not options:
            raise ValueError("_select_collab_trump_kill requires options")
        trump_options = [opt for opt in options if opt.get("is_trump")]
        if trump_options:
            options = trump_options
        points_on_table = context.get("points_on_table", 0)
        level = context.get("level", "2")
        level_opts = [opt for opt in options if self._option_contains_rank(opt, level)]
        if points_on_table <= 10 and level_opts:
            best = min(level_opts, key=lambda opt: (opt.get("premium_cost", 0), opt.get("max_strength", 0)))
            return best["index"]
        best = max(options, key=lambda opt: (opt.get("max_strength", 0), -self._point_penalty(opt)))
        return best["index"]

    def _select_minimal(self, options: List[Dict[str, Any]]) -> int:
        best = min(options, key=lambda opt: (self._point_penalty(opt), opt["max_strength"], opt["length"]))
        return best["index"]

    def _select_smallest(self, options: List[Dict[str, Any]]) -> int:
        return self._select_minimal(options)

    def _select_trump_lead(self, options: List[Dict[str, Any]]) -> int:
        best = max(options, key=lambda opt: (opt["length"], opt["max_strength"] - opt["premium_cost"] - self._point_penalty(opt)))
        return best["index"]

    def _select_offensive_lead(
        self,
        bucket_name: str,
        options: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> int:
        suit_counts = context.get("hand_suit_counts", {})
        if bucket_name == "off_balance":
            best = max(options, key=lambda opt: (opt.get("max_strength", 0), opt.get("length", 1), -self._point_penalty(opt)))
            return best["index"]
        if bucket_name == "void":
            strong = [opt for opt in options if opt.get("max_strength", 0) >= RANK_ORDER.get("9", 7)]
            if strong:
                options = strong
        def key(opt):
            suit = opt.get("primary_suit")
            suit_pressure = suit_counts.get(suit, 0)
            return (
                opt.get("max_strength", 0),
                -suit_pressure,
                -self._point_penalty(opt),
                -opt.get("premium_cost", 0),
            )

        best = max(options, key=key)
        return best["index"]

    def _select_minimum_point_dump(self, options: List[Dict[str, Any]]) -> int:
        best = min(options, key=lambda opt: (self._point_penalty(opt), opt["max_strength"]))
        return best["index"]

    def _select_low_risk_filler(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> int:
        bait = self._bait_point_traps(options, context)
        if bait:
            return self._select_smallest(bait)
        void_ready = self._void_enabling_candidates(options, context)
        if void_ready:
            return self._select_smallest(void_ready)
        non_point = [opt for opt in options if not opt.get("contains_points")]
        if non_point:
            return self._select_smallest(non_point)
        five_only = [opt for opt in options if self._option_points_subset(opt, {"5"})]
        if five_only:
            return self._select_smallest(five_only)
        return self._select_smallest(options)

    def _void_enabling_candidates(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not options:
            return []
        suit_counts = context.get("hand_suit_counts") or {}
        results: List[Dict[str, Any]] = []
        for opt in options:
            if opt.get("contains_points") or opt.get("is_trump"):
                continue
            suit = opt.get("primary_suit")
            if not suit or suit == "trump":
                continue
            remaining = suit_counts.get(suit, 0) - opt.get("length", 1)
            if remaining <= 0:
                results.append(opt)
        return results

    def _bait_point_traps(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        enemy_void = context.get("enemy_void_suits", set())
        if not enemy_void:
            return []
        point_visibility = context.get("point_visibility", {})
        candidates: List[Dict[str, Any]] = []
        for opt in options:
            if opt.get("is_trump") or opt.get("contains_points"):
                continue
            if opt.get("length") != 1:
                continue
            suit = opt.get("primary_suit")
            if not suit or suit == "trump" or suit not in enemy_void:
                continue
            visibility = point_visibility.get(suit, {})
            if visibility.get("0") and visibility.get("5"):
                continue
            if opt.get("max_strength", 0) > RANK_ORDER.get("J", 9):
                continue
            candidates.append(opt)
        return candidates

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

    def _option_contains_rank(self, option: Dict[str, Any], rank: str) -> bool:
        if not rank:
            return False
        return any(len(card) == 2 and card[1] == rank for card in option.get("cards", []))

    def _option_points_subset(self, option: Dict[str, Any], allowed: set) -> bool:
        ranks = self._option_point_ranks(option)
        return bool(ranks) and ranks.issubset(allowed)

    def _classify_game_stage(self, cards_seen: int) -> str:
        if cards_seen <= EARLY_CARD_LIMIT:
            return "early"
        if cards_seen <= MID_CARD_LIMIT:
            return "mid"
        return "late"

    def _apply_stage_constraints(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        options = context.get("options") or []
        if not options:
            return options
        stage = context.get("stage", "early")
        if stage == "early":
            filtered = [opt for opt in options if not self._contains_joker(opt)]
            filtered = [opt for opt in filtered if not (opt.get("is_pair") and opt.get("pair_rank") == context.get("level"))]
            return filtered if filtered else options
        biased = self._bias_off_suit_release(options, context)
        return biased if biased else options

    def _bias_off_suit_release(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not options:
            return options
        suit_counts = context.get("hand_suit_counts") or {}
        stage = context.get("stage", "mid")
        if stage == "early":
            return options

        def priority(opt: Dict[str, Any]) -> Tuple[int, int, int, float]:
            suit = opt.get("primary_suit")
            suit_pressure = suit_counts.get(suit, 4) if suit else 4
            off_suit_bias = 2
            if suit and suit != "trump":
                off_suit_bias = 0
                if suit_counts.get(suit, 0) <= 1:
                    off_suit_bias = -1
            contains_points = 1 if opt.get("contains_points") else 0
            strength = opt.get("max_strength", 0.0)
            return (off_suit_bias, suit_pressure, contains_points, strength)

        return sorted(options, key=priority)

    def _compute_position_valuation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        deck = context.get("deck", [])
        major = context.get("major", "s")
        level = context.get("level", "2")
        stage = context.get("stage", "early")
        role = context.get("role", "farmer")
        trump_count = context.get("hand_trump_count", 0)
        total_points = context.get("total_points_played", 0)
        score_state = context.get("score_state", "balanced")
        score_margin = context.get("score_margin", 0)

        base = 1.0
        stage_adjust = {"early": 0.2, "mid": 0.0, "late": -0.15}.get(stage, 0.0)
        if stage == "late" and total_points < LATE_LOW_POINT_THRESHOLD:
            stage_adjust -= 0.3
        if stage == "early" and total_points > 20:
            stage_adjust += 0.15

        score_adjust = 0.0
        if score_state == "leading":
            score_adjust += 0.15
        elif score_state == "trailing":
            score_adjust -= 0.2
        score_adjust += max(min(score_margin, 40), -40) / 200.0

        hand_strength = self._estimate_hand_strength(deck, major, level)
        trump_density = trump_count / max(len(deck), 1)
        trump_weight = 0.6 if role == "banker" else 0.45
        trump_adjust = trump_density * trump_weight
        hand_adjust = hand_strength / 1000.0
        special_bonus = self._special_weapon_bonus(deck, major, level)

        total = base + stage_adjust + score_adjust + hand_adjust + trump_adjust + special_bonus
        state = "advantage"
        if total <= VALUATION_DISADV_THRESHOLD:
            state = "disadvantage"
        elif total < VALUATION_ADV_THRESHOLD:
            state = "balanced"

        plan_bias = self._decide_lead_bias(context, total)
        return {"score": total, "state": state, "plan_bias": plan_bias}

    def _estimate_hand_strength(self, cards: List[str], major: str, level: str) -> float:
        if not cards:
            return 0.0
        strengths = [self._card_strength(card, major, level) for card in cards]
        strengths.sort(reverse=True)
        top_slice = strengths[:8] if len(strengths) > 8 else strengths
        return sum(top_slice) / len(top_slice)

    def _special_weapon_bonus(self, cards: List[str], major: str, level: str) -> float:
        if not cards:
            return 0.0
        jokers = sum(1 for card in cards if card in ("jo", "Jo"))
        level_pairs = {}
        for card in cards:
            if len(card) != 2:
                continue
            suit, rank = card[0], card[1]
            key = (suit, rank)
            level_pairs[key] = level_pairs.get(key, 0) + 1
        bonus = 0.0
        if jokers >= 2:
            bonus += 0.2
        elif jokers == 1:
            bonus += 0.1
        premium_pairs = [cnt for (suit, rank), cnt in level_pairs.items() if cnt >= 2 and (rank == level or suit == major or rank in {"A", "K"})]
        if premium_pairs:
            bonus += 0.1
        return bonus

    def _decide_lead_bias(self, context: Dict[str, Any], valuation_score: float) -> str:
        cards = context.get("deck", [])
        suit_counts = context.get("hand_suit_counts") or {}
        trump_count = context.get("hand_trump_count", 0)
        trump_ratio = trump_count / max(len(cards), 1)
        shortest_off = min((cnt for suit, cnt in suit_counts.items() if cnt > 0), default=4)
        stage = context.get("stage", "early")
        if trump_ratio >= 0.45:
            return "trump"
        if shortest_off <= 1:
            return "offsuit"
        if stage == "late" and valuation_score <= 1.0:
            return "offsuit"
        return "balanced"

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

    def _detect_teammate_small_trump_signal(self, context: Dict[str, Any]) -> bool:
        order = context.get("current_trick_order") or []
        history = context.get("history") or []
        teammate_id = context.get("teammate_id")
        if not order or not history or teammate_id is None:
            return False
        if order[0] != teammate_id:
            return False
        lead_play = history[0]
        if not lead_play:
            return False
        feature = self._build_option_feature(-1, lead_play, context.get("major", "s"), context.get("level", "2"))
        if not feature.get("is_trump"):
            return False
        if feature.get("premium_cost", 0) >= 15:
            return False
        return feature.get("max_strength", 0) < SMALL_TRUMP_SIGNAL_THRESHOLD

    def _resolve_strong_peer_mark(self, context: Dict[str, Any]) -> bool:
        meta = context.get("meta")
        if isinstance(meta, dict) and meta.get("strong_peer_mark"):
            return True
        detected = self._detect_teammate_strong_trump_mark(context)
        if detected and isinstance(meta, dict):
            meta["strong_peer_mark"] = True
        return detected

    def _detect_teammate_strong_trump_mark(self, context: Dict[str, Any]) -> bool:
        order = context.get("current_trick_order") or []
        history = context.get("history") or []
        teammate_id = context.get("teammate_id")
        if not order or not history or teammate_id is None:
            return False
        if order[0] != teammate_id:
            return False
        if context.get("played"):
            return False
        lead_play = history[0]
        if not lead_play:
            return False
        feature = self._build_option_feature(-1, lead_play, context.get("major", "s"), context.get("level", "2"))
        if not feature.get("is_trump"):
            return False
        if self._contains_joker(feature):
            return True
        level = context.get("level", "2")
        major = context.get("major", "s")
        has_level = any(len(card) == 2 and card[1] == level for card in lead_play)
        if has_level:
            return True
        return any(len(card) == 2 and card[0] == major and RANK_ORDER.get(card[1], -1) > RANK_ORDER["J"] for card in lead_play)

    def _detect_teammate_point_trump_lead(self, context: Dict[str, Any]) -> bool:
        order = context.get("current_trick_order") or []
        history = context.get("history") or []
        teammate_id = context.get("teammate_id")
        if not order or not history or teammate_id is None:
            return False
        if order[0] != teammate_id:
            return False
        lead_play = history[0]
        if not lead_play:
            return False
        feature = self._build_option_feature(-1, lead_play, context.get("major", "s"), context.get("level", "2"))
        return feature.get("is_trump") and feature.get("contains_points")

    def _should_force_trump_open(self, context: Dict[str, Any]) -> bool:
        position = context.get("position_info", {})
        if not position.get("is_first"):
            return False
        if context.get("played"):
            return False
        return context.get("hand_trump_count", 0) >= 12 and self._has_joker(context.get("deck") or [])

    def _has_joker(self, cards: Sequence[str]) -> bool:
        return any(card in ("jo", "Jo") for card in cards)

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
        candidates = [opt for opt in options if opt.get("contains_points") and not opt.get("is_trump")]
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

    def _prepare_void_shedding_leads(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not options:
            return []
        if context.get("meta", {}).get("lead_trump", False):
            return []
        if context.get("hand_trump_count", 0) >= 6:
            return []
        suit_counts = context.get("hand_suit_counts") or {}
        candidates = [opt for opt in options if self._is_void_shedding_candidate(opt, suit_counts, context)]
        bait = self._enemy_void_bait_leads(options, context)
        if bait:
            merged = {opt["index"]: opt for opt in candidates}
            for opt in bait:
                merged[opt["index"]] = opt
            candidates = list(merged.values())
        if not candidates:
            return []
        return self._refine_options(candidates, context, priority="lead-void")

    def _is_void_shedding_candidate(self, option: Dict[str, Any], suit_counts: Dict[str, int], context: Dict[str, Any]) -> bool:
        if option.get("is_trump"):
            return False
        if option.get("contains_points"):
            return False
        if option.get("length") != 1:
            return False
        suit = option.get("primary_suit")
        if not suit or suit == "trump":
            return False
        if suit_counts.get(suit, 0) != 1:
            return False
        rank_value = option.get("max_strength", -math.inf)
        if rank_value >= RANK_ORDER.get("K", 11):
            return False
        if context.get("score_state") == "leading":
            return True
        return context.get("hand_trump_count", 0) <= 4

    def _enemy_void_bait_leads(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        enemy_void = context.get("enemy_void_suits", set())
        if not enemy_void:
            return []
        point_visibility = context.get("point_visibility", {})
        stage = context.get("stage", "mid")
        cap = RANK_ORDER.get("Q", 10) if stage != "late" else RANK_ORDER.get("K", 11)
        candidates: List[Dict[str, Any]] = []
        for opt in options:
            if opt.get("is_trump") or opt.get("contains_points"):
                continue
            if opt.get("length") != 1:
                continue
            suit = opt.get("primary_suit")
            if not suit or suit == "trump" or suit not in enemy_void:
                continue
            visibility = point_visibility.get(suit, {})
            if visibility.get("0") and visibility.get("5"):
                continue
            if opt.get("max_strength", 0) > cap:
                continue
            candidates.append(opt)
        return candidates

    def _filter_small_leads(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not options:
            return []
        filtered: List[Dict[str, Any]] = []
        blocked: List[Dict[str, Any]] = []
        for opt in options:
            if self._should_block_small_lead(opt, context):
                blocked.append(opt)
            else:
                filtered.append(opt)
        return filtered if filtered else options

    def _should_block_small_lead(self, option: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if option.get("is_trump"):
            return False
        if option.get("contains_points"):
            return False
        if option.get("length") != 1:
            return False
        suit = option.get("primary_suit")
        if not suit or suit == "trump":
            return False
        suit_counts = context.get("hand_suit_counts") or {}
        if suit_counts.get(suit, 0) <= 1:
            return False
        rank_value = option.get("max_strength", -math.inf)
        return rank_value < RANK_ORDER[SAFE_LEAD_MIN_RANK]

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
        seat_index = self._seat_index(context)
        teammate_flags = context.get("teammate_flags", {})
        teammate_dumped = teammate_flags.get("dumped_points", False)
        teammate_void = teammate_flags.get("follow_state") == "void"
        filtered: List[Dict[str, Any]] = []
        for opt in options:
            if not self._is_trump_takeover(opt, lead_suit):
                filtered.append(opt)
                continue
            if trump_status == "disadvantage":
                allow = (
                    points_on_table >= 10
                    or (points_on_table >= 5 and score_state == "trailing")
                    or teammate_dumped
                    or (teammate_void and points_on_table > 0)
                    or (seat_index == 3 and points_on_table > 0)
                )
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


