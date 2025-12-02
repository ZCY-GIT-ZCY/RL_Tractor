import json
import os
from data.model import CNNModel
import torch
from data.wrapper import cardWrapper
from data.mvGen import move_generator
import numpy as np
from collections import Counter
from data.declaration import decide_declaration, decide_overcall
from data.kitty import select_kitty_cards

cardscale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
suitset = ['s','h','c','d']
Major = ['jo', 'Jo']
pointorder = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']
PLAYER_ID_KEYS = ("player", "player_id", "playerID", "self", "self_id", "seat_id")
SELF_PLAYER_ID = None


def maybe_update_self_player(container):
    global SELF_PLAYER_ID
    if not isinstance(container, dict):
        return SELF_PLAYER_ID
    for key in PLAYER_ID_KEYS:
        value = container.get(key)
        if isinstance(value, int):
            SELF_PLAYER_ID = value % 4
            return SELF_PLAYER_ID
    return SELF_PLAYER_ID


def _group_cards_by_name(cards):
    grouped = {}
    for card in cards:
        name = Num2Poker(card)
        grouped.setdefault(name, []).append(card)
    return grouped


def _pick_declaration_cards(grouped_cards, suit, level):
    target = suit + level
    candidates = grouped_cards.get(target, [])
    return candidates[:1]


def _pick_snatch_cards(grouped_cards, candidate, level):
    if candidate == 'n':
        for joker in ("Jo", "jo"):
            possible = grouped_cards.get(joker, [])
            if len(possible) >= 2:
                return possible[:2]
        return []
    target = candidate + level
    possible = grouped_cards.get(target, [])
    if len(possible) >= 2:
        return possible[:2]
    return []

def setMajor(major, level):
    global Major
    if major != 'n': # 非无主
        Major = [major+point for point in pointorder if point != level] + [suit + level for suit in suitset if suit != major] + [major + level] + Major
    else: # 无主
        Major = [suit + level for suit in suitset] + Major
    pointorder.remove(level)
    
def Num2Poker(num): # num: int-[0,107]
    # Already a poker
    if type(num) is str and (num in Major or (num[0] in suitset and num[1] in cardscale)):
        return num
    # Locate in 1 single deck
    NumInDeck = num % 54
    # joker and Joker:
    if NumInDeck == 52:
        return "jo"
    if NumInDeck == 53:
        return "Jo"
    # Normal cards:
    pokernumber = cardscale[NumInDeck // 4]
    pokersuit = suitset[NumInDeck % 4]
    return pokersuit + pokernumber

def Poker2Num(poker, deck): # poker: str
    NumInDeck = -1
    if poker[0] == "j":
        NumInDeck = 52
    elif poker[0] == "J":
        NumInDeck = 53
    else:
        NumInDeck = cardscale.index(poker[1])*4 + suitset.index(poker[0])
    if NumInDeck in deck:
        return NumInDeck
    else:
        return NumInDeck + 54

def Poker2Num_seq(pokers, deck):
    id_seq = []
    deck_copy = deck + []
    for card_name in pokers:
        card_id = Poker2Num(card_name, deck_copy)
        id_seq.append(card_id)
        deck_copy.remove(card_id)
    return id_seq
    
def checkPokerType(poker, level): #poker: list[int]
    poker = [Num2Poker(p) for p in poker]
    if len(poker) == 1:
        return "single" #一张牌必定为单牌
    if len(poker) == 2:
        if poker[0] == poker[1]:
            return "pair" #同点数同花色才是对子
        else:
            return "suspect" #怀疑是甩牌
    if len(poker) % 2 == 0: #其他情况下只有偶数张牌可能是整牌型（连对）
    # 连对：每组两张；各组花色相同；各组点数在大小上连续(需排除大小王和级牌)
        count = Counter(poker)
        if "jo" in count.keys() and "Jo" in count.keys() and count['jo'] == 2 and count['Jo'] == 2:
            return "tractor"
        elif "jo" in count.keys() or "Jo" in count.keys(): # 排除大小王
            return "suspect"
        for v in count.values(): # 每组两张
            if v != 2:
                return "suspect"
        pointpos = []
        suit = list(count.keys())[0][0] # 花色相同
        for k in count.keys():
            if k[0] != suit or k[1] == level: # 排除级牌
                return "suspect"
            pointpos.append(pointorder.index(k[1])) # 点数在大小上连续
        pointpos.sort()
        for i in range(len(pointpos)-1):
            if pointpos[i+1] - pointpos[i] != 1:
                return "suspect"
        return "tractor" # 说明是拖拉机
    
    return "suspect"
    
def call_Snatch(get_card, deck, called, snatched, level, current_trump, self_player=None):
# get_card: new card in this turn (int)
# deck: your deck (list[int]) before getting the new card
# called & snatched: player_id, -1 if not called/snatched
# level: level
# return -> list[int]
    hand_ids = list(deck)
    if get_card is not None:
        hand_ids.append(get_card)
    grouped_cards = _group_cards_by_name(hand_ids)

    if snatched not in (-1, None):
        return []

    if called in (-1, None):
        candidate = decide_declaration(hand_ids, level)
        if not candidate:
            return []
        cards = _pick_declaration_cards(grouped_cards, candidate, level)
        return cards if cards else []

    teammate_called = None
    if self_player is not None:
        if called == self_player:
            return []
        teammate_id = (self_player + 2) % 4
        teammate_called = called == teammate_id

    active_trump = current_trump or 'n'
    candidate = decide_overcall(
        hand_ids,
        level,
        active_trump,
        teammate_called=teammate_called,
    )
    if not candidate:
        return []
    cards = _pick_snatch_cards(grouped_cards, candidate, level)
    return cards if cards else []

def cover_Pub(old_public, deck, level, major):
# old_public: raw publiccard (list[int])
    full_hand = list(deck) + list(old_public)
    return select_kitty_cards(full_hand, level, major, len(old_public))

def playCard(history, hold, played, level, wrapper, mv_gen, model):
    """
    history: 当前这一轮（局部）历史，来自 Botzone 的 curr_request["history"][1]
    hold:    自己当前手牌（Botzone 牌 id 列表）
    played:  四家已经打出的牌（id 列表的列表）
    level:   级牌点数（如 '2', '3', ...）
    """

    # 1. 构造 obs（注意这里用的是牌名）
    obs = {
        "id": selfid,
        "deck": [Num2Poker(p) for p in hold],
        "history": [[Num2Poker(p) for p in move] for move in history],
        "major": [Num2Poker(p) for p in Major],
        "played": [[Num2Poker(p) for p in cardset] for cardset in played],
    }

    # 2. 生成合法动作集合（牌名形式）
    action_options = get_action_options(hold, history, level, mv_gen)
    if not action_options:
        # 理论上 mv_gen 一定能生成至少一个动作；这里加个兜底，直接 pass（不出牌）
        return []

    # 3. 用 wrapper 编码 obs + action_options
    obs_mat, action_mask = wrapper.obsWrap(obs, action_options)

    # 4. 组装模型输入
    state = {
        "observation": torch.tensor(obs_mat, dtype=torch.float32).unsqueeze(0),   # (1, 128, 4, 14)
        "action_mask": torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0),  # (1, 54)
        # Botzone 这里只在“出牌阶段”调用，所以直接用 PLAY = 2 对应的 head
        "stage": torch.tensor([2], dtype=torch.long),  # (1,)
    }

    # 5. 调模型选一个 action index
    action_idx = obs2action(model, state)

    # 6. 把 action（牌名列表）转成 Botzone 要的牌 id 列表
    response = action_intpt(action_options[action_idx], hold)
    return response


def get_action_options(deck, history, level, mv_gen):
    deck = [Num2Poker(p) for p in deck]
    if len(history) == 4 or len(history) == 0: # first to play
        return mv_gen.gen_all(deck)
    else:
        tgt = [Num2Poker(p) for p in history[0]]
        poktype = checkPokerType(history[0], level)
        if poktype == "single":
            return mv_gen.gen_single(deck, tgt)
        elif poktype == "pair":
            return mv_gen.gen_pair(deck, tgt)
        elif poktype == "tractor":
            return mv_gen.gen_tractor(deck, tgt)
        elif poktype == "suspect":
            return mv_gen.gen_throw(deck, tgt)    

def obs2action(model, state):
    """
    state: {
        "observation": (1, 128, 4, 14) tensor,
        "action_mask": (1, 54) tensor
    }
    """
    model.eval()
    with torch.no_grad():
        logits, value = model(state)  # logits 已经是 mask 之后的
        # 这里可以采样，也可以贪心；我这里给你用贪心，稳定一点
        action = torch.argmax(logits, dim=-1).item()
    return action


def action_intpt(action, deck):
    '''
    interpreting action(cardname) to response(dick{'player': int, 'action': list[int]})
    action: list[str(cardnames)]
    '''
    action = Poker2Num_seq(action, deck)
    return action

_online = os.environ.get("USER", "") == "root"
if _online:
    full_input = json.loads(input())
else:
    with open("log_forAI.json") as fo:
        full_input = json.load(fo)
maybe_update_self_player(full_input)

# loading model
model = CNNModel()
data_dir = '/data/tractor_model.pt' # to be modified
model.load_state_dict(torch.load(data_dir, map_location = torch.device('cpu')))

hold = []
played = [[], [], [], []]
for i in range(len(full_input["requests"])-1):
    req = full_input["requests"][i]
    maybe_update_self_player(req)
    if req["stage"] == "deal":
        hold.extend(req["deliver"])
    elif req["stage"] == "cover":
        hold.extend(req["deliver"])
        action_cover = full_input["responses"][i]
        for id in action_cover:
            hold.remove(id)
    elif req["stage"] == "play":
        history = req["history"]
        selfid = (history[3] + len(history[1])) % 4
        maybe_update_self_player({"player": selfid})
        if len(history[0]) != 0:
            self_move = history[0][(selfid-history[2]) % 4]
            #print(hold)
            #print(self_move)
            for id in self_move:
                hold.remove(id)
            for player_rec in range(len(history[0])): # Recovering played cards
                played[(history[2]+player_rec) % 4].extend(history[0][player_rec])
curr_request = full_input["requests"][-1]
maybe_update_self_player(curr_request)
if curr_request["stage"] == "deal":
    get_card = curr_request["deliver"][0]
    called = curr_request["global"]["banking"]["called"]
    snatched = curr_request["global"]["banking"]["snatched"]
    level = curr_request["global"]["level"]
    current_trump = curr_request["global"]["banking"].get("major")
    response = call_Snatch(get_card, hold, called, snatched, level, current_trump, SELF_PLAYER_ID)
elif curr_request["stage"] == "cover":
    publiccard = curr_request["deliver"]
    level = curr_request["global"]["level"]
    major = curr_request["global"]["banking"]["major"]
    response = cover_Pub(publiccard, hold, level, major)
elif curr_request["stage"] == "play":
    level = curr_request["global"]["level"]
    major = curr_request["global"]["banking"]["major"]
    setMajor(major, level)
    # instantiate move_generator and cardwrapper 
    card_wrapper = cardWrapper()
    mv_gen = move_generator(level, major)
    
    history = curr_request["history"]
    selfid = (history[3] + len(history[1])) % 4
    maybe_update_self_player({"player": selfid})
    if len(history[0]) != 0:
        self_move = history[0][(selfid-history[2]) % 4]
        #print(hold)
        #print(self_move)
        for id in self_move:
            hold.remove(id)
        for player_rec in range(len(history[0])): # Recovering played cards
            played[(history[2]+player_rec) % 4].extend(history[0][player_rec])
        for player_rec in range(len(history[1])):
            played[(history[3]+player_rec) % 4].extend(history[1][player_rec])
    history_curr = history[1]
    
    response = playCard(history_curr, hold, played, level, card_wrapper, mv_gen, model)

print(json.dumps({
    "response": response
}))



