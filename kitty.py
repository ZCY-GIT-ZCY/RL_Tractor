"""
Rule-based kitty (bury) strategy.

The selector scores every card in the banker's hand and discards the weakest
ones.  Scores follow a simple additive model so hyper-parameters are easy to
adjust without touching the logic.
"""

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

CARD_SCALE = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'c', 'd']

# ---------------------------------------------------------------------------
# Tunable weights
# ---------------------------------------------------------------------------
BASE_WEAKNESS = {
    '2': 7.0,
    '3': 6.5,
    '4': 6.0,
    '5': 5.5,
    '6': 5.0,
    '7': 4.5,
    '8': 4.0,
    '9': 3.0,
    '0': 0.0,  # ten is a score card
    'J': 2.0,
    'Q': 1.0,
    'K': 0.0,
    'A': -1.0,
}
DEFAULT_WEAKNESS = 2.0  
ISOLATION_BONUS = {1: 4.0, 2: 2.0, 3: 1.0}
NO_CONTROL_PENALTY = 2.5
SCORE_CARD_RISK = 5.0
SCORE_RISK_COUNT_THRESHOLD = 2
TRUMP_PROTECTION = 6.0
LEVEL_PROTECTION = 8.0
JOKER_PROTECTION = 20.0
PAIR_PROTECTION = 3.0
TRACTOR_PROTECTION = 5.0
TRACTOR_HEAD_EXTRA = 1.0  # makes the tractor head effectively -6


def num_to_name(card: int) -> str:
    num_in_deck = card % 54
    if num_in_deck == 52:
        return "jo"
    if num_in_deck == 53:
        return "Jo"
    rank = CARD_SCALE[num_in_deck // 4]
    suit = SUITS[num_in_deck % 4]
    return suit + rank


def normalize_cards(cards: Iterable[int]) -> List[str]:
    return [num_to_name(int(card)) for card in cards]


def is_trump(name: str, major: str, level: str) -> bool:
    if name in ("jo", "Jo"):
        return True
    suit, rank = name[0], name[1]
    if major == 'n':
        return rank == level
    if rank == level:
        return True
    return suit == major


def _rank_value(rank: str, level: str) -> int:
    if rank == level:
        return len(CARD_SCALE)
    return CARD_SCALE.index(rank)


def _build_suit_panel(
    name_counter: Counter,
    major: str,
    level: str,
) -> Dict[str, Dict[str, bool]]:
    info: Dict[str, Dict[str, bool]] = {
        suit: defaultdict(bool, count=0) for suit in SUITS
    }
    for suit in SUITS:
        info[suit]['count'] = 0

    for name, cnt in name_counter.items():
        if name in ("jo", "Jo") or len(name) < 2:
            continue
        suit, rank = name[0], name[1]
        if is_trump(name, major, level):
            continue
        info[suit]['count'] += cnt
        if rank == 'A':
            info[suit]['has_ace'] = True
        if cnt >= 2:
            info[suit]['has_pair'] = True

    for suit in SUITS:
        ranks = [
            rank
            for rank in CARD_SCALE
            if name_counter[f"{suit}{rank}"] >= 2
            and not is_trump(f"{suit}{rank}", major, level)
        ]
        streak = []
        prev = None
        has_tractor = False
        for rank in sorted(ranks, key=lambda rk: _rank_value(rk, level)):
            value = _rank_value(rank, level)
            if prev is None or value == prev + 1:
                streak.append(rank)
            else:
                if len(streak) >= 2:
                    has_tractor = True
                    break
                streak = [rank]
            prev = value
        if not has_tractor and len(streak) >= 2:
            has_tractor = True
        info[suit]['has_tractor'] = has_tractor
        info[suit]['no_control'] = not (
            info[suit]['has_ace'] or info[suit]['has_pair'] or has_tractor
        )
    return info


def _tractor_memberships(name_counter: Counter, level: str) -> Tuple[set, set]:
    members = set()
    heads = set()
    for suit in SUITS:
        ranked_pairs = [
            rank for rank in CARD_SCALE if name_counter[f"{suit}{rank}"] >= 2
        ]
        if len(ranked_pairs) < 2:
            continue
        streak: List[str] = []
        prev = None
        for rank in sorted(ranked_pairs, key=lambda rk: _rank_value(rk, level)):
            value = _rank_value(rank, level)
            if prev is None or value == prev + 1:
                streak.append(rank)
            else:
                if len(streak) >= 2:
                    members.update(f"{suit}{rk}" for rk in streak)
                    heads.add(f"{suit}{streak[-1]}")
                streak = [rank]
            prev = value
        if len(streak) >= 2:
            members.update(f"{suit}{rk}" for rk in streak)
            heads.add(f"{suit}{streak[-1]}")
    return members, heads


def _base_weakness(rank: str) -> float:
    return BASE_WEAKNESS.get(rank, DEFAULT_WEAKNESS)


def _score_card(
    name: str,
    suit_panel: Dict[str, Dict[str, bool]],
    pair_names: set,
    tractor_members: set,
    tractor_heads: set,
    major: str,
    level: str,
) -> float:
    if name in ("jo", "Jo"):
        return -JOKER_PROTECTION
    suit, rank = name[0], name[1]
    score = _base_weakness(rank)
    if is_trump(name, major, level):
        score -= TRUMP_PROTECTION
        if rank == level:
            score -= LEVEL_PROTECTION
    else:
        count = suit_panel[suit]['count']
        score += ISOLATION_BONUS.get(count, 0.0)
        if suit_panel[suit].get('no_control'):
            score += NO_CONTROL_PENALTY
        if rank in ('K', '0'):
            if count <= SCORE_RISK_COUNT_THRESHOLD or suit_panel[suit].get('no_control'):
                score += SCORE_CARD_RISK

    if name in pair_names:
        score -= PAIR_PROTECTION
    if name in tractor_members:
        score -= TRACTOR_PROTECTION
        if name in tractor_heads:
            score -= TRACTOR_HEAD_EXTRA
    return score


def _tie_break_key(name: str, level: str) -> float:
    if name in ("jo", "Jo"):
        return -100.0
    return -float(_rank_value(name[1], level))


def select_kitty_cards(
    hand: Sequence[int],
    level: str,
    major: str,
    bury_count: int,
) -> List[int]:
    """
    Select `bury_count` cards to discard from `hand`.
    """
    if bury_count <= 0:
        return []
    major = major or 'n'
    cards = list(hand)
    names = normalize_cards(cards)
    name_counter = Counter(names)
    suit_panel = _build_suit_panel(name_counter, major, level)
    pair_names = {name for name, cnt in name_counter.items() if cnt >= 2}
    tractor_members, tractor_heads = _tractor_memberships(name_counter, level)

    scored = []
    for card_id, name in zip(cards, names):
        score = _score_card(
            name,
            suit_panel,
            pair_names,
            tractor_members,
            tractor_heads,
            major,
            level,
        )
        scored.append((score, _tie_break_key(name, level), card_id))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [card_id for _, _, card_id in scored[:bury_count]]
    return selected


__all__ = [
    "select_kitty_cards",
]
