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
VOID_TARGET_COUNT = 2  # aim to void short suits when possible
VOID_PRIORITY_BONUS = 4.0  # extra weakness when voiding a suit
POINT_RISK_MULT_HIGH = 1.5  # multiplier when protection is weak
POINT_RISK_MULT_LOW = 0.8   # multiplier when protection is strong
TRUMP_LACK_PENALTY = 1.5  # soft penalty when kitty lacks big trumps


def _dynamic_scales(hand_size: int, final_size: int, initial_bury: int) -> Tuple[float, float, float]:
    """
    Return (isolaton_scale, risk_scale, protection_scale) based on how many
    cards still need to be buried. When many cards remain to bury (aggressive
    phase) we amplify risk penalties to shed points quickly; as we approach the
    target hand size we increase protection to avoid breaking strong structures.
    """
    remaining_to_bury = max(0, hand_size - final_size)
    aggressiveness = remaining_to_bury / max(1, initial_bury)
    aggressiveness = min(1.0, max(0.0, aggressiveness))
    isolation_scale = 0.6 + 0.9 * aggressiveness
    risk_scale = 0.6 + 0.9 * aggressiveness
    protection_scale = 1.0 + (1.0 - aggressiveness)
    return isolation_scale, risk_scale, protection_scale


def num_to_name(card: int) -> str:
    num_in_deck = card % 54
    if num_in_deck == 52:
        return "jo"
    if num_in_deck == 53:
        return "Jo"
    rank = CARD_SCALE[num_in_deck // 4]
    suit = SUITS[num_in_deck % 4]
    return suit + rank


def normalize_cards(cards: Iterable) -> List[str]:
    normalized: List[str] = []
    for card in cards:
        if isinstance(card, str):
            normalized.append(card)
        else:
            normalized.append(num_to_name(int(card)))
    return normalized


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
    isolation_scale: float,
    risk_scale: float,
    protection_scale: float,
    void_targets: Dict[str, bool],
    score_risk_multiplier: float,
) -> float:
    if name in ("jo", "Jo"):
        return -JOKER_PROTECTION * protection_scale
    suit, rank = name[0], name[1]
    score = _base_weakness(rank)
    if is_trump(name, major, level):
        score -= TRUMP_PROTECTION * protection_scale
        if rank == level:
            score -= LEVEL_PROTECTION * protection_scale
        if rank in ('0', 'K', '5'):
            score += SCORE_CARD_RISK * 0.4 * score_risk_multiplier
    else:
        count = suit_panel[suit]['count']
        score += ISOLATION_BONUS.get(count, 0.0) * isolation_scale
        if suit_panel[suit].get('no_control'):
            score += NO_CONTROL_PENALTY * risk_scale
        if rank in ('K', '0', '5'):
            if count <= SCORE_RISK_COUNT_THRESHOLD or suit_panel[suit].get('no_control'):
                score += SCORE_CARD_RISK * risk_scale * score_risk_multiplier
        if void_targets.get(suit):
            score += VOID_PRIORITY_BONUS * isolation_scale

    if name in pair_names:
        score -= PAIR_PROTECTION * protection_scale
    if name in tractor_members:
        score -= TRACTOR_PROTECTION * protection_scale
        if name in tractor_heads:
            score -= TRACTOR_HEAD_EXTRA * protection_scale
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
    cards = list(hand)
    major = major or 'n'
    final_size = max(0, len(cards) - bury_count)
    initial_bury = len(cards) - final_size
    selected: List[int] = []

    for _ in range(min(bury_count, len(cards))):
        if len(cards) <= final_size:
            break
        names = normalize_cards(cards)
        name_counter = Counter(names)
        suit_panel = _build_suit_panel(name_counter, major, level)
        pair_names = {name for name, cnt in name_counter.items() if cnt >= 2}
        tractor_members, tractor_heads = _tractor_memberships(name_counter, level)
        iso_scale, risk_scale, protection_scale = _dynamic_scales(len(cards), final_size, initial_bury)

        current_trumps = [name for name in names if is_trump(name, major, level)]
        trump_counter = Counter(current_trumps)
        has_big = trump_counter.get("Jo", 0) > 0
        small_pairs = trump_counter.get("jo", 0) // 2
        trump_count = len(current_trumps)

        if has_big and small_pairs >= 1 and trump_count >= 10:
            score_risk_multiplier = POINT_RISK_MULT_LOW
        else:
            score_risk_multiplier = 1.0
            if not has_big:
                score_risk_multiplier += 0.5
            if small_pairs == 0:
                score_risk_multiplier += 0.3
            if trump_count < 8:
                score_risk_multiplier += 0.2
            score_risk_multiplier = min(score_risk_multiplier, POINT_RISK_MULT_HIGH)

        trump_deficit = 0.0
        if not has_big:
            trump_deficit += TRUMP_LACK_PENALTY
        if small_pairs == 0:
            trump_deficit += TRUMP_LACK_PENALTY * 0.5

        void_targets = {}
        sorted_suits = sorted(
            SUITS,
            key=lambda suit: suit_panel[suit]['count'],
        )
        for suit in sorted_suits[:VOID_TARGET_COUNT]:
            if suit_panel[suit]['count'] <= 2 and not any(
                is_trump(f"{suit}{rank}", major, level)
                for rank in CARD_SCALE
            ):
                void_targets[suit] = True

        prefer_offsuit = any(not is_trump(name, major, level) for name in names)
        prefer_safe_non_ace = any(
            (not is_trump(name, major, level)) and name[1] != 'A'
            for name in names
        )

        def _append_scores(allow_trump: bool, allow_aces: bool):
            local_scored = []
            for idx, (card_id, name) in enumerate(zip(cards, names)):
                if prefer_offsuit and not allow_trump and is_trump(name, major, level):
                    continue
                if prefer_safe_non_ace and not allow_aces and name[1] == 'A' and not is_trump(name, major, level):
                    continue
                score = _score_card(
                    name,
                    suit_panel,
                    pair_names,
                    tractor_members,
                    tractor_heads,
                    major,
                    level,
                    iso_scale,
                    risk_scale,
                    protection_scale,
                    void_targets,
                    score_risk_multiplier,
                )
                local_scored.append((score, _tie_break_key(name, level), idx))
            return local_scored

        scored = _append_scores(allow_trump=False, allow_aces=False)
        if not scored:
            scored = _append_scores(allow_trump=False, allow_aces=True)
        if not scored:
            scored = _append_scores(allow_trump=True, allow_aces=True)

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _, _, remove_idx = scored[0]
        selected.append(cards.pop(remove_idx))

    return selected


__all__ = [
    "select_kitty_cards",
]
