"""
Advanced scoring-based declaration / overcall heuristics for Tractor.

Priorities:
1) Trump structure (longest tractors, number of tractors, quality pairs)
2) Big trump thickness (jokers, level cards, main-suit A/K)
3) Total trump count / ballast
4) Point-card pairs in trump (10/K)
5) Risk penalties for scattered points.

We use lexicographic comparison on the structure tuple and scalar thresholds
for declare / overcall decisions.
"""

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

CARD_SCALE = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
SUIT_SET = ['s', 'h', 'c', 'd']
TRUMP_CANDIDATES = ['s', 'h', 'c', 'd', 'n']

# scoring weights / thresholds
DECLARE_SCORE_THRESHOLD = 15.0  # baseline minimum score to consider declaring
OVERCALL_SCORE_THRESHOLD = 18.0  # baseline minimum score to consider overcalling
STRUCTURE_MARGIN = 1.5  # baseline margin for structure comparison
BANKER_TRUMP_MIN = 6  # baseline minimum trump cards for banker

TRACTOR_LEN_VALUE = 5.0 # value per card in longest tractor
TRACTOR_COUNT_VALUE = 3.0 # value per tractor
LEVEL_PAIR_VALUE = 3.0 # value per pair of level cards
JOKER_PAIR_VALUE_BIG = 4.0 # value per pair of big jokers
JOKER_PAIR_VALUE_SMALL = 3.0 # value per pair of small jokers
AK_PAIR_VALUE = 2.0 # value per pair of A/K
COMMON_PAIR_VALUE = 1.0 # value per pair of other cards

JOKER_SINGLE_VALUE_BIG = 3.5 # value per single big joker
JOKER_SINGLE_VALUE_SMALL = 2.5 # value per single small joker
LEVEL_SINGLE_VALUE = 2.5 # value per single level card
AK_SINGLE_VALUE = 1.5 # value per single A/K
TRUMP_SINGLE_VALUE = 0.5 # value per single trump card

POINT_PAIR_VALUE = 1.5 # value per pair of point cards
POINT_SINGLE_PENALTY = 0.0 # penalty per single point card

def num_to_poker(card: int) -> str:
    num = card % 54
    if num == 52:
        return "jo"
    if num == 53:
        return "Jo"
    rank = CARD_SCALE[num // 4]
    suit = SUIT_SET[num % 4]
    return suit + rank


def normalize_cards(hand: Iterable) -> List[str]:
    normalized = []
    for card in hand:
        if isinstance(card, str):
            normalized.append(card)
        else:
            normalized.append(num_to_poker(int(card)))
    return normalized


def is_trump(card: str, candidate: str, level: str) -> bool:
    if card in ("jo", "Jo"):
        return True
    rank = card[1]
    suit = card[0]
    if candidate == 'n':
        return rank == level
    if rank == level:
        return True
    return suit == candidate


def trump_rank_value(card: str, candidate: str, level: str) -> int:
    if card in ("jo", "Jo"):
        return 200 if card == "Jo" else 190
    rank = card[1]
    order = CARD_SCALE.index(rank)
    if rank == level:
        return 150 + order
    if candidate != 'n' and card[0] == candidate:
        return 100 + order
    return order


def analyze_pairs(trump_cards: List[str], candidate: str, level: str):
    counts = Counter(trump_cards)
    pair_info = {}
    for card, cnt in counts.items():
        if cnt >= 2:
            pair_info[card] = cnt // 2
    ordered_pairs = sorted(
        [card for card in pair_info for _ in range(pair_info[card])],
        key=lambda c: trump_rank_value(c, candidate, level)
    )
    tractor_primary = 0
    tractor_count = 0
    if ordered_pairs:
        streak = 1
        for prev, cur in zip(ordered_pairs, ordered_pairs[1:]):
            pv = trump_rank_value(prev, candidate, level)
            cv = trump_rank_value(cur, candidate, level)
            if cv == pv + 1:
                streak += 1
            else:
                if streak >= 2:
                    tractor_count += 1
                    tractor_primary = max(tractor_primary, streak)
                streak = 1
        if streak >= 2:
            tractor_count += 1
            tractor_primary = max(tractor_primary, streak)
    return pair_info, tractor_primary, tractor_count


def high_pair_score(pair_info: Dict[str, int], level: str) -> float:
    score = 0.0
    for card, pairs in pair_info.items():
        if card == "Jo":
            score += pairs * JOKER_PAIR_VALUE_BIG
        elif card == "jo":
            score += pairs * JOKER_PAIR_VALUE_SMALL
        elif card[1] == level:
            score += pairs * LEVEL_PAIR_VALUE
        elif card[1] in ('A', 'K'):
            score += pairs * AK_PAIR_VALUE
        else:
            score += pairs * COMMON_PAIR_VALUE
    return score


def point_pair_metrics(pair_info: Dict[str, int]) -> Tuple[float, int]:
    value = 0.0
    count = 0
    for card, pairs in pair_info.items():
        if card[1] in ('0', 'K'):
            value += pairs * POINT_PAIR_VALUE
            count += pairs
    return value, count


def big_trump_strength(trump_cards: List[str], level: str, candidate: str) -> float:
    strength = 0.0
    for card in trump_cards:
        if card == "Jo":
            strength += JOKER_SINGLE_VALUE_BIG
        elif card == "jo":
            strength += JOKER_SINGLE_VALUE_SMALL
        elif card[1] == level:
            strength += LEVEL_SINGLE_VALUE
        elif card[1] in ('A', 'K') and (candidate == 'n' or card[0] == candidate):
            strength += AK_SINGLE_VALUE
        else:
            strength += TRUMP_SINGLE_VALUE
    return strength


def point_single_penalty(cards: List[str], candidate: str, level: str) -> float:
    penalty = 0.0
    for card in cards:
        if card[1] in ('0', 'K') and not is_trump(card, candidate, level):
            penalty += POINT_SINGLE_PENALTY
    return penalty


def evaluate_trump_candidate(hand: Iterable, candidate: str, level: str) -> Dict[str, float]:
    cards = normalize_cards(hand)
    trump_cards = [card for card in cards if is_trump(card, candidate, level)]
    pair_info, tractor_primary, tractor_count = analyze_pairs(trump_cards, candidate, level)
    high_pairs = high_pair_score(pair_info, level)
    big_strength = big_trump_strength(trump_cards, level, candidate)
    point_pair_value, point_pair_count = point_pair_metrics(pair_info)
    total_trump = len(trump_cards)
    risk_penalty = point_single_penalty(cards, candidate, level)

    structure_score = (
        tractor_primary * TRACTOR_LEN_VALUE
        + tractor_count * TRACTOR_COUNT_VALUE
        + high_pairs
    )
    ballast_score = total_trump * 0.5
    scalar_score = structure_score + big_strength + ballast_score + point_pair_value - risk_penalty

    return {
        "candidate": candidate,
        "tractor_primary": tractor_primary,
        "tractor_count": tractor_count,
        "high_pairs": high_pairs,
        "big_strength": big_strength,
        "trump_count": total_trump,
        "point_pair_value": point_pair_value,
        "point_pair_count": point_pair_count,
        "risk_penalty": risk_penalty,
        "score_scalar": scalar_score,
        "structure_score": structure_score,
        "profile_tuple": (
            tractor_primary,
            tractor_count,
            high_pairs,
            big_strength,
            total_trump,
            point_pair_value,
            scalar_score,
        ),
    }


def evaluate_all_trumps(hand: Iterable, level: str, candidates: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    candidates = candidates or TRUMP_CANDIDATES
    profiles = {}
    for candidate in candidates:
        profiles[candidate] = evaluate_trump_candidate(hand, candidate, level)
    return profiles


def compare_profiles(a: Dict[str, float], b: Dict[str, float]) -> int:
    if a["profile_tuple"] > b["profile_tuple"]:
        return 1
    if a["profile_tuple"] < b["profile_tuple"]:
        return -1
    return 0


def dynamic_thresholds(hand_size: int) -> Tuple[float, float, float, int]:
    declare_thr = DECLARE_SCORE_THRESHOLD
    overcall_thr = OVERCALL_SCORE_THRESHOLD
    struct_margin = STRUCTURE_MARGIN
    trump_req = BANKER_TRUMP_MIN

    if hand_size <= 10:
        declare_thr += 4
        overcall_thr += 4
        struct_margin += 0.5
        trump_req += 1
    elif hand_size <= 14:
        declare_thr += 2
        overcall_thr += 2
        struct_margin += 0.3
    elif hand_size >= 24:
        declare_thr -= 3
        overcall_thr -= 3
        struct_margin = max(0.7, struct_margin - 0.5)
        trump_req = max(4, trump_req - 2)
    elif hand_size >= 20:
        declare_thr -= 2
        overcall_thr -= 2
        struct_margin = max(0.9, struct_margin - 0.3)
        trump_req = max(4, trump_req - 1)

    declare_thr = max(8.0, declare_thr)
    overcall_thr = max(10.0, overcall_thr)
    return declare_thr, overcall_thr, struct_margin, max(4, trump_req)


def profile_meets_declare(profile: Dict[str, float], hand_size: int, declare_threshold: float, trump_requirement: int) -> bool:
    if profile["tractor_primary"] >= 3:
        return True
    if profile["tractor_primary"] >= 2 and profile["high_pairs"] >= 3.0:
        return True
    if profile["high_pairs"] >= 4.0 and profile["big_strength"] >= 6.0:
        return True
    dynamic_requirement = trump_requirement
    if hand_size <= 12:
        dynamic_requirement += 1
    if hand_size >= 22:
        dynamic_requirement = max(4, dynamic_requirement - 1)
    if profile["score_scalar"] >= declare_threshold and profile["trump_count"] >= dynamic_requirement:
        return True
    return False


def decide_declaration(hand: Iterable, level: str, force_on_level: bool = False) -> Optional[str]:
    hand_list = list(hand)
    profiles = evaluate_all_trumps(hand_list, level)
    if not profiles:
        return None
    if force_on_level:
        normalized_hand = normalize_cards(hand_list)
        for candidate in TRUMP_CANDIDATES:
            if candidate == 'n':
                continue
            target = candidate + level
            if target in normalized_hand:
                return candidate
    hand_size = len(hand_list)
    declare_thr, overcall_thr, struct_margin, trump_req = dynamic_thresholds(hand_size)
    best = max(profiles.values(), key=lambda x: x["profile_tuple"])
    if not profile_meets_declare(best, hand_size, declare_thr, trump_req):
        return None
    best_suit = best["candidate"]
    if best_suit == 'n':
        return None
    return best_suit


def decide_overcall(hand: Iterable, level: str, enemy_trump: str, teammate_called: Optional[bool] = None) -> Optional[str]:
    hand_list = list(hand)
    profiles = evaluate_all_trumps(hand_list, level)
    my_profile = max(profiles.values(), key=lambda x: x["profile_tuple"])
    enemy_profile = profiles.get(enemy_trump)
    if enemy_profile is None:
        enemy_profile = evaluate_trump_candidate(hand_list, enemy_trump, level)

    if my_profile["candidate"] == enemy_trump:
        return None

    hand_size = len(hand_list)
    declare_thr, overcall_thr, struct_margin, trump_req = dynamic_thresholds(hand_size)

    if my_profile["score_scalar"] < overcall_thr:
        return None

    structure_gap = my_profile["structure_score"] - enemy_profile["structure_score"]
    big_gap = my_profile["big_strength"] - enemy_profile["big_strength"]

    if structure_gap > struct_margin:
        return my_profile["candidate"]
    if structure_gap >= struct_margin / 2 and big_gap > 1.5:
        return my_profile["candidate"]
    return None


__all__ = [
    "evaluate_all_trumps",
    "decide_declaration",
    "decide_overcall",
]
