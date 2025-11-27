"""
Rule-based declaration / overcall heuristics for Tractor.

The module exposes helpers that score every candidate trump suit and apply a
a few thresholds to determine whether we should declare / overcall / stay
silent.  The scoring function is intentionally simple and linear so that the
thresholds can be tuned easily if offline analysis suggests better values.

Interface summary
-----------------
- evaluate_all_trumps(hand, level, candidates=None)
    -> {candidate: score, ...}
- decide_declaration(hand, level)
    -> best candidate suit or None (pass)
- decide_overcall(hand, level, enemy_trump, teammate_called=False)
    -> new suit if we should overcall, otherwise None

`hand` should be a list of cards either in int-id form (0..107, like our env)
or in string form such as 'sA', 'h0', 'Jo'.  The helper automatically converts
ints to strings so the caller can pass in raw Botzone ids as well.
"""

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

CARD_SCALE = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
SUIT_SET = ['s', 'h', 'c', 'd']
TRUMP_CANDIDATES = ['s', 'h', 'c', 'd', 'n']  # 'n' denotes no-trump / joker trump

# ---------------------------------------------------------------------------
# Tunable hyper-parameters (easy to tweak)
# ---------------------------------------------------------------------------
T_CALL = 8.0
T_OVERCALL_HIGH = 11.0
DELTA_OVERCALL_MARGIN = 2.0
T_SUPPORT_FRIEND = 6.0
T_OVERCALL_TEAM = 10.0
T_FRIEND_BAD = 3.0
JOKER_BONUS = 3.0
LEVEL_CARD_BONUS = 1.0
BIG_AK_BONUS = 0.5
BASE_TRUMP_POINTS = 1.0
TRACTOR_BONUS = 2.0


def num_to_poker(card: int) -> str:
    """Convert env-style integer id to 'sA' style string."""
    num_in_deck = card % 54
    if num_in_deck == 52:
        return "jo"
    if num_in_deck == 53:
        return "Jo"
    rank = CARD_SCALE[num_in_deck // 4]
    suit = SUIT_SET[num_in_deck % 4]
    return suit + rank


def normalize_cards(hand: Iterable) -> List[str]:
    result = []
    for card in hand:
        if isinstance(card, str):
            result.append(card)
        else:
            result.append(num_to_poker(int(card)))
    return result


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


def count_pairs(trump_cards: List[str]) -> Counter:
    cnt = Counter()
    for card in trump_cards:
        cnt[card] += 1
    return cnt


def tractor_groups(pair_cards: List[str], candidate: str, level: str) -> int:
    """Approximate number of tractor groups (two or more consecutive pairs)."""
    # Use ranks converted to integer order
    def rank_value(card: str) -> int:
        rank = card[1]
        order = CARD_SCALE.index(rank)
        # Level cards should be highest in their suit
        if rank == level:
            return len(CARD_SCALE) + 2
        return order

    # Keep only cards that have at least two copies
    counts = Counter(pair_cards)
    usable = sorted(
        [card for card, c in counts.items() if c >= 2],
        key=lambda c: rank_value(c),
    )
    if not usable:
        return 0

    values = [rank_value(card) for card in usable]
    tractor = 0
    streak = 1
    for prev, cur in zip(values, values[1:]):
        if cur == prev + 1:
            streak += 1
        else:
            if streak >= 2:
                tractor += 1
            streak = 1
    if streak >= 2:
        tractor += 1
    return tractor


def evaluate_trump_strength(hand: Iterable, candidate: str, level: str) -> Tuple[float, Dict[str, float]]:
    cards = normalize_cards(hand)
    trump_cards = [card for card in cards if is_trump(card, candidate, level)]
    trump_count = len(trump_cards)
    base = BASE_TRUMP_POINTS * trump_count

    kings_and_aces = sum(1 for card in trump_cards if card[1] in ('A', 'K'))
    ak_bonus = BIG_AK_BONUS * kings_and_aces

    level_bonus = LEVEL_CARD_BONUS * sum(1 for card in trump_cards if card[1] == level or card in ("jo", "Jo"))
    joker_bonus = JOKER_BONUS * sum(1 for card in trump_cards if card in ("jo", "Jo"))

    pair_counts = count_pairs(trump_cards)
    tractor_bonus = 0
    if pair_counts:
        pair_list = [card for card, cnt in pair_counts.items() if cnt >= 2]
        tractor_bonus = TRACTOR_BONUS * tractor_groups(pair_list, candidate, level)

    score = base + ak_bonus + level_bonus + joker_bonus + tractor_bonus
    stats = {
        "trump_count": trump_count,
        "ak_bonus": ak_bonus,
        "level_bonus": level_bonus,
        "joker_bonus": joker_bonus,
        "tractor_bonus": tractor_bonus,
    }
    return score, stats


def evaluate_all_trumps(hand: Iterable, level: str, candidates: Optional[List[str]] = None) -> Dict[str, float]:
    candidates = candidates or TRUMP_CANDIDATES
    scores = {}
    for candidate in candidates:
        score, _ = evaluate_trump_strength(hand, candidate, level)
        scores[candidate] = score
    return scores


def _best_candidates(scores: Dict[str, float]) -> List[str]:
    if not scores:
        return []
    best_value = max(scores.values())
    return [suit for suit, score in scores.items() if score == best_value]


def decide_declaration(hand: Iterable, level: str) -> Optional[str]:
    scores = evaluate_all_trumps(hand, level)
    best_suits = _best_candidates(scores)
    if not best_suits:
        return None
    best_score = scores[best_suits[0]]
    if best_score < T_CALL:
        return None
    # Preferred ordering when scores tie: no-trump > level suit > rest suits
    if 'n' in best_suits:
        return 'n'
    level_suit = next((suit for suit in best_suits if suit != 'n' and suit == level), None)
    if level_suit:
        return level_suit
    priority = ['s', 'h', 'c', 'd']
    for suit in priority:
        if suit in best_suits:
            return suit
    return best_suits[0]


def decide_overcall(hand: Iterable, level: str, enemy_trump: str, teammate_called: bool = False) -> Optional[str]:
    scores = evaluate_all_trumps(hand, level)
    my_best_suit = max(scores, key=lambda k: scores[k])
    my_best_score = scores[my_best_suit]
    enemy_score = scores.get(enemy_trump, 0.0)

    if teammate_called:
        # Support teammate unless current trump is terrible
        teammate_score = enemy_score
        if teammate_score >= T_SUPPORT_FRIEND:
            return None
        if teammate_score <= T_FRIEND_BAD and my_best_score >= T_OVERCALL_TEAM:
            return my_best_suit if my_best_suit != enemy_trump else None
        return None

    if my_best_score < T_OVERCALL_HIGH:
        return None
    if my_best_score - enemy_score < DELTA_OVERCALL_MARGIN:
        return None
    if my_best_suit == enemy_trump:
        return None
    return my_best_suit


__all__ = [
    "T_CALL",
    "T_OVERCALL_HIGH",
    "DELTA_OVERCALL_MARGIN",
    "T_SUPPORT_FRIEND",
    "T_OVERCALL_TEAM",
    "T_FRIEND_BAD",
    "evaluate_trump_strength",
    "evaluate_all_trumps",
    "decide_declaration",
    "decide_overcall",
]

