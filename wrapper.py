import numpy as np

class cardWrapper:
    def __init__(self,
                 suit_sequence = ['s', 'h', 'c', 'd'],
                 point_sequence = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']):
        """
        suit_sequence:   花色顺序，用于在 (4,14) 的维度上定位
        point_sequence:  点数顺序，用于在 (4,14) 的维度上定位
        """
        # 这里的 card_scale 主要是提供一个从点字符到序号的查表（部分函数会用到）
        self.card_scale = ['A','2','3','4','5','6','7','8','9','0','J','Q','K']
        self.suit_sequence = suit_sequence
        self.point_sequence = point_sequence

        # 大/小王用“花色维度”存位置（其实是伪花色）
        self.J_pos = self.suit_sequence.index('h')  # 你原来的设定：大王放在 'h' 那个 slot
        self.j_pos = self.suit_sequence.index('s')  # 小王放在 's' 那个 slot

    # ---------- 名称 / 坐标互转 ----------

    def name2pos(self, cardname):
        """
        将牌名映射到 (suit_idx, point_idx)
        约定：
        - 'jo' 小王，对应 cardname[0] == 'j'
        - 'Jo' 大王，对应 cardname[0] == 'J'
        - 其余： cardname[0] 是花色，cardname[1] 是点数
        """
        if cardname[0] == "J":   # 大王 "Jo"
            return (self.J_pos, 13)
        if cardname[0] == "j":   # 小王 "jo"
            return (self.j_pos, 13)

        suit = cardname[0]
        point = cardname[1]
        suit_idx = self.suit_sequence.index(suit)
        point_idx = self.point_sequence.index(point)
        return (suit_idx, point_idx)

    def pos2name(self, cardpos):
        """
        从 (suit_idx, point_idx) 还原牌名
        """
        suit_idx, point_idx = cardpos
        if point_idx == 13:
            if suit_idx == self.j_pos:
                return "jo"
            if suit_idx == self.J_pos:
                return "Jo"
            raise RuntimeError("Card not exists.")

        return self.suit_sequence[suit_idx] + self.point_sequence[point_idx]

    # ---------- 对 (2,4,14)/(8,4,14)/(108,4,14) 的增删 ----------

    def add_card(self, cardset: np.ndarray, cards):
        """
        cardset: (..., 4, 14)，最前面的维度表示“副数/槽位”（2 / 8 / 108）
        cards:   list[str] 或单个 str
        """
        if isinstance(cards, str):
            cards = [cards]

        for card in cards:
            suit_idx, point_idx = self.name2pos(card)
            # 在最前面的那一维上找第一个 0 的槽位
            placed = False
            for k in range(cardset.shape[0]):
                if cardset[k, suit_idx, point_idx] == 0:
                    cardset[k, suit_idx, point_idx] = 1
                    placed = True
                    break
            # 超过可容纳副数时，直接忽略（一般不会发生）
            if not placed:
                pass

        return cardset

    def remove_card(self, cardset: np.ndarray, cards):
        if isinstance(cards, str):
            cards = [cards]

        for card in cards:
            suit_idx, point_idx = self.name2pos(card)
            removed = False
            for k in range(cardset.shape[0]-1, -1, -1):
                if cardset[k, suit_idx, point_idx] != 0:
                    cardset[k, suit_idx, point_idx] = 0
                    removed = True
                    break
            if not removed:
                raise RuntimeError("Card not in cardset! Please recheck.")

        return cardset

    def Unwrap(self, cardset: np.ndarray):
        """
        从 cardset (2,4,14) 或 (8,4,14) 中恢复牌名列表
        """
        cards = []
        card_poses = np.nonzero(cardset)
        # card_poses: (axis0_indices, suit_indices, point_indices)
        for i in range(card_poses[0].size):
            suit_idx = card_poses[1][i]
            point_idx = card_poses[2][i]
            card_name = self.pos2name((suit_idx, point_idx))
            cards.append(card_name)
        return cards

    # ---------- 包装观测 ----------

    def obsWrap(self, obs, options):
        """
        Wrapping the observation and crafting the action_mask.

        obs: 来自 env 的 dict（见 TractorEnv._get_obs）
        options: 当前阶段所有合法动作（牌名列表的列表）
        """
        pid = obs['id']
        # major / deck: 每个是 (2,4,14)
        major_mat  = np.zeros((2, 4, 14), dtype=np.float32)
        deck_mat   = np.zeros((2, 4, 14), dtype=np.float32)

        # history / played: 最多 4 墩×2 层 = (8,4,14)
        hist_mat   = np.zeros((8, 4, 14), dtype=np.float32)
        played_mat = np.zeros((8, 4, 14), dtype=np.float32)

        # options: 最多 54 个动作，每个 2 层 = (108,4,14)
        option_mat = np.zeros((108, 4, 14), dtype=np.float32)

        # --- 填充 major ---
        self.add_card(major_mat, obs['major'])

        # --- 填充自己的手牌 ---
        self.add_card(deck_mat, obs['deck'])

        # --- 填充当前这一墩的 history（从首家开始按顺序最多 4 家）---
        for i in range(len(obs['history'])):
            if i * 2 >= hist_mat.shape[0]:
                break
            self.add_card(hist_mat[i*2:(i+1)*2], obs['history'][i])

        # --- 填充整局已出牌（按自己视角旋转）---
        played_cards = obs['played'][pid:] + obs['played'][:pid]
        for i in range(len(played_cards)):
            if i * 2 >= played_mat.shape[0]:
                break
            self.add_card(played_mat[i*2:(i+1)*2], played_cards[i])

        # --- 填充 action options ---
        for i in range(len(options)):
            if i * 2 >= option_mat.shape[0]:
                break
            self.add_card(option_mat[i*2:(i+1)*2], options[i])

        # --- 构造 action_mask ---
        action_mask = np.zeros(54, dtype=np.float32)
        action_mask[:len(options)] = 1.0

        # 最终观测：128 = 2 + 2 + 8 + 8 + 108
        obs_mat = np.concatenate((major_mat, deck_mat, hist_mat, played_mat, option_mat), axis=0)
        return obs_mat, action_mask
