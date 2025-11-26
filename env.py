import random
from collections import Counter
from mvGen import move_generator

class Error(Exception):
    def __init__(self, ErrorInfo):
        self.ErrorInfo = ErrorInfo
        
    def __str__(self):
        return self.ErrorInfo


class TractorEnv():
    STAGE_SNATCH = 0
    STAGE_BURY = 1
    STAGE_PLAY = 2

    def __init__(self, config={}):
        if 'seed' in config:
            self.seed = config['seed']
        else:
            self.seed = None

        self.suit_set = ['s','h','c','d']
        self.card_scale = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K']
        self.major = None
        self.level = None
        self.agent_names = ['player_%d' % i for i in range(4)]
        self.mode = None
        self.stage = self.STAGE_SNATCH
    def reset(self, level='2', banker_pos=0, major='s'):
        """
        Reset the environment.

        - 不再一次性发完 25 张牌。
        - 进入 STAGE_SNATCH：每发一张牌，就让拿到牌的玩家立刻有一次报主 / 反主 / 不报 的决策机会。
        """
        # basic game parameters
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self.Major = ['jo', 'Jo']
        self.level = level
        self.first_round = True
        self.banker_pos = banker_pos

        # 初始主花色（之后可以被报主 / 反主覆盖）
        if major == 'r':
            self.major = random.sample(self.suit_set, 1)[0]
        else:
            self.major = major

        # report / snatch 状态
        self.reporter = None
        self.snatcher = None

        # 初始化整副牌，并拆分为：前 100 张用于发牌，后 8 张为底牌
        self.total_deck = list(range(108))
        random.shuffle(self.total_deck)
        self.card_todeal = self.total_deck[:100]   # 顺序发出的 100 张牌
        self.card_public = self.total_deck[100:]   # 原始底牌 8 张

        # 四家起始手牌为空，之后通过 _deal_one 逐步发牌
        self.player_decks = [[] for _ in range(4)]

        # 扣底后的底牌；开始为空，STAGE_BURY 时从庄家手中选出
        self.covered_card = []

        # game state
        self.score = 0
        self.history = []
        self.played_cards = [[] for _ in range(4)]
        self.reward = None
        self.done = False

        # 发牌 / 阶段状态
        self.stage = self.STAGE_SNATCH
        self.deal_index = 0          # 已经发出的牌数（0..100）
        self.bury_left = len(self.card_public)  # 庄家需要扣的牌数（等于原始底牌数）
        self.round = 0               # 正常出牌阶段的轮次计数（只在 PLAY 用）

        # 首张牌：发给 player 0（如果你想对齐线上规则，也可以改成 banker_pos）
        self.curr_player = self.banker_pos
        self._deal_one(self.curr_player)

        obs = self._get_obs(self.curr_player)
        action_options = self._get_action_options(self.curr_player)
        return obs, action_options

    def _deal_one(self, player: int):
        """给指定玩家发一张牌。"""
        if self.deal_index >= len(self.card_todeal):
            return None
        card = self.card_todeal[self.deal_index]
        self.deal_index += 1
        self.player_decks[player].append(card)
        return card

    def step(self, response):
        """
        response: dict{'player': int, 'action': int}
        这里的 `action` 是当前 action_options 的索引。
        """
        self.reward = None
        curr_player = response['player']
        action_idx = response['action']

        # ---------- STAGE 0: 发牌 + 报主 / 反主 ----------
        if self.stage == self.STAGE_SNATCH:
            assert curr_player == self.curr_player

            options = self._get_action_options(curr_player)
            if not options:
                options = [[]]  # 至少要有个“pass”
            if action_idx < 0 or action_idx >= len(options):
                raise Error(f"Invalid action index {action_idx} for player {curr_player} in SNATCH stage")

            # options 里每个元素是一个牌名列表；空列表 [] 表示“什么都不报”
            chosen_names = options[action_idx]

            # 非空则进行报主 / 反主
            if len(chosen_names) > 0:
                chosen_ids = self._name2id_seq(chosen_names, self.player_decks[curr_player])
                if len(chosen_ids) == 1:
                    # 报主
                    self._report(chosen_ids, curr_player)
                elif len(chosen_ids) == 2:
                    # 反主
                    self._snatch(chosen_ids, curr_player)
                else:
                    self._raise_error(curr_player, "INVALID_SNATCH_ACTION")

            # 之后要么继续发牌，要么发牌阶段结束
            if self.deal_index < len(self.card_todeal):
                # 还有牌可以发 -> 发给下家，然后轮到下家报主/反主/不报
                next_player = (curr_player + 1) % 4
                self._deal_one(next_player)
                self.curr_player = next_player

                obs = self._get_obs(next_player)
                action_options = self._get_action_options(next_player)
                return obs, action_options, None, False
            else:
                # 100 张牌已经发完：确定庄家与主花色，发底牌给庄家，进入扣底阶段
                if self.banker_pos is None or self.banker_pos not in range(4):
                    if self.snatcher is not None:
                        self.banker_pos = self.snatcher
                    elif self.reporter is not None:
                        self.banker_pos = self.reporter
                    else:
                        # 没有人报主：随机选庄家和主
                        self.random_pick_major()

                # 将原始底牌发给庄家
                self._deliver_public()

                # 庄家开始扣底
                self.stage = self.STAGE_BURY
                self.curr_player = self.banker_pos
                # 重置 covered_card；真正的底牌将从庄家手中选出
                self.covered_card = []
                self.bury_left = len(self.card_public)

                obs = self._get_obs(self.curr_player)
                action_options = self._get_action_options(self.curr_player)
                return obs, action_options, None, False

        # ---------- STAGE 1: 庄家扣底 ----------
        if self.stage == self.STAGE_BURY:
            assert curr_player == self.curr_player == self.banker_pos

            options = self._get_action_options(curr_player)
            if not options:
                raise Error("No bury options for banker")

            if action_idx < 0 or action_idx >= len(options):
                raise Error(f"Invalid action index {action_idx} for player {curr_player} in BURY stage")

            chosen_names = options[action_idx]
            chosen_ids = self._name2id_seq(chosen_names, self.player_decks[curr_player])

            self._bury(chosen_ids)   # 更新 covered_card 并从庄家手中移除
            self.bury_left -= len(chosen_ids)

            if self.bury_left <= 0:
                # 扣底完成，进入正式出牌阶段
                # 此时主花色已确定，构建 Major 集和 move_generator
                self._setMajor()
                self.mv_gen = move_generator(self.level, self.major)

                self.stage = self.STAGE_PLAY
                self.curr_player = self.banker_pos
                # 重置一墩的历史
                self.history = []
            else:
                # 继续让庄家扣底
                self.curr_player = self.banker_pos

            obs = self._get_obs(self.curr_player)
            action_options = self._get_action_options(self.curr_player)
            return obs, action_options, None, False

        # ---------- STAGE 2: 正常出牌 ----------
        if self.stage == self.STAGE_PLAY:
            assert curr_player == self.curr_player

            options = self._get_action_options(curr_player)
            if not options:
                raise Error("No play options available")

            if action_idx < 0 or action_idx >= len(options):
                raise Error(f"Invalid action index {action_idx} for player {curr_player} in PLAY stage")

            chosen_names = options[action_idx]
            chosen_ids = self._name2id_seq(chosen_names, self.player_decks[curr_player])

            real_action = self._checkLegalMove(chosen_ids, curr_player)
            real_action_ids = self._name2id_seq(real_action, self.player_decks[curr_player])
            self._play(curr_player, real_action_ids)

            next_player = (curr_player + 1) % 4

            # 一墩打完
            if len(self.history) == 4:
                winner = self._checkWinner(curr_player)
                next_player = winner
                if len(self.player_decks[0]) == 0:
                    # 全部打完，游戏结束
                    self._reveal(curr_player, winner)
                    self.done = True

            self.curr_player = next_player
            self.round += 1

            obs = self._get_obs(next_player)
            action_options = self._get_action_options(next_player)
            if self.reward:
                return obs, action_options, self.reward, self.done
            return obs, action_options, None, self.done

        # 不应该走到这里
        raise Error(f"Invalid game stage {self.stage}")

    def _report(self, repo_card_ids, reporter):
        """
        repo_card_ids: list[int]，长度为 1
        不能用大小王报主。
        """
        if self.reporter is not None:
            self._raise_error(reporter, "ALREADY_REPORTED")

        repo_name = self._id2name(repo_card_ids[0])
        # 必须是级牌且不是大小王
        if repo_name in ("jo", "Jo") or repo_name[1] != self.level:
            self._raise_error(reporter, "INVALID_MOVE")

        self.major = repo_name[0]
        self.reporter = reporter

    def _snatch(self, snatch_card_ids, snatcher):
        """
        snatch_card_ids: list[int]，长度为 2，必须是合法对子（级牌对子或大小王对子）。
        """
        if self.reporter is None:
            self._raise_error(snatcher, "CANNOT_SNATCH")
        if self.snatcher is not None:
            self._raise_error(snatcher, "ALREADY_SNATCHED")

        if (snatch_card_ids[1] - snatch_card_ids[0]) % 54 != 0:
            self._raise_error(snatcher, "INVALID_MOVE")

        snatch_name = self._id2name(snatch_card_ids[0])
        # 大小王对：无主；其他情况用级牌花色
        if snatch_name[1] == 'o':
            self.major = 'n'
        else:
            if snatch_name[1] != self.level:
                self._raise_error(snatcher, "INVALID_MOVE")
            self.major = snatch_name[0]

        self.snatcher = snatcher

    def random_pick_major(self):
        """发牌阶段没人报主时，随机选庄和主。"""
        if self.first_round:
            self.banker_pos = random.choice(range(4))
        self.major = random.choice(self.suit_set)

    def _deliver_public(self):
        """将原始底牌发给庄家（真正的底牌之后由庄家从手牌中选出）。"""
        for card in self.card_public:
            self.player_decks[self.banker_pos].append(card)

    def _bury(self, cover_cards):
        """庄家扣底：从庄家手牌中移除 cover_cards 放入 covered_card。"""
        for card in cover_cards:
            self.covered_card.append(card)
            self.player_decks[self.banker_pos].remove(card)

    def _get_obs(self, player):
        obs = {
            "id": player,
            "deck": [self._id2name(p) for p in self.player_decks[player]],
            "history": [[self._id2name(p) for p in move] for move in self.history],
            "major": self.Major,
            "played": [[self._id2name(p) for p in move] for move in self.played_cards],
            "stage": self.stage,
        }
        return obs

    def _get_action_options(self, player):
        """
        按阶段返回 action options（全部是*牌名列表*）：

        - STAGE_SNATCH: 报主 / 反主 / 不报
        - STAGE_BURY:   庄家扣底
        - STAGE_PLAY:   正常出牌（用 move_generator）
        """
        deck_names = [self._id2name(p) for p in self.player_decks[player]]

        if self.stage == self.STAGE_SNATCH:
            return self._get_declaration_options(player, deck_names)

        if self.stage == self.STAGE_BURY:
            return self._get_bury_options(player, deck_names)

        # default: STAGE_PLAY
        return self._get_play_options(player, deck_names)

    def _get_declaration_options(self, player, deck_names):
        """
        SNATCH 阶段的报主 / 反主动作：
        options[0] 永远是 [] 代表“pass”。
        """
        options = [[]]  # pass

        # 没有人报主 -> 可以用单张级牌（非大小王）报主
        if self.reporter is None:
            for name in deck_names:
                if len(name) == 2 and name[1] == self.level and name not in ("jo", "Jo"):
                    options.append([name])

        # 已经有人报主，但还没人反主 -> 可以用级牌对子或大小王对子反主
        if (self.reporter is not None) and (self.snatcher is None):
            cnt = Counter(deck_names)
            # 级牌对子
            for name, c in cnt.items():
                if len(name) == 2 and name[1] == self.level and c >= 2:
                    options.append([name, name])
            # 大小王对子
            for joker in ("jo", "Jo"):
                if cnt[joker] >= 2:
                    options.append([joker, joker])

        return options

    def _get_bury_options(self, player, deck_names):
        """
        庄家扣底选牌：这里给一个最简单版本——一次选一张。
        你之后可以自己扩展为“一次选多张”。
        """
        if player != self.banker_pos:
            return []

        return [[name] for name in deck_names]

    def _get_play_options(self, player, deck_names):
        """
        正常出牌阶段，沿用原来的 mvGen 逻辑。
        """
        if len(self.history) == 4 or len(self.history) == 0:  # 首家出牌
            return self.mv_gen.gen_all(deck_names)
        else:
            tgt = [self._id2name(p) for p in self.history[0]]
            poktype = self._checkPokerType(self.history[0], (player - len(self.history)) % 4)
            if poktype == "single":
                return self.mv_gen.gen_single(deck_names, tgt)
            elif poktype == "pair":
                return self.mv_gen.gen_pair(deck_names, tgt)
            elif poktype == "tractor":
                return self.mv_gen.gen_tractor(deck_names, tgt)
            elif poktype == "suspect":
                return self.mv_gen.gen_throw(deck_names, tgt)
            else:
                return []

        
    

    
        
    
    
    # ... Helper functions from original env.py (Play, Snatch, etc) ...
    # Keeping original implementations where possible
    
    def _id2name(self, card_id): 
        NumInDeck = card_id % 54
        if NumInDeck == 52: return "jo"
        if NumInDeck == 53: return "Jo"
        pokernumber = self.card_scale[NumInDeck // 4]
        pokersuit = self.suit_set[NumInDeck % 4]
        return pokersuit + pokernumber
    
    def _name2id(self, card_name, deck):
        # ... original implementation ...
        NumInDeck = -1
        if card_name[0] == "j": NumInDeck = 52
        elif card_name[0] == "J": NumInDeck = 53
        else: NumInDeck = self.card_scale.index(card_name[1])*4 + self.suit_set.index(card_name[0])
        if NumInDeck in deck: return NumInDeck
        else: return NumInDeck + 54
    
    def _name2id_seq(self, card_names, deck):
        id_seq = []
        deck_copy = deck + []
        for card_name in card_names:
            card_id = self._name2id(card_name, deck_copy)
            id_seq.append(card_id)
            deck_copy.remove(card_id)
        return id_seq
        
    def _play(self, player, cards):
        for card in cards:
            self.player_decks[player].remove(card)
            self.played_cards[player].append(card)
        if len(self.history) == 4: 
            self.history = []
        self.history.append(cards)
        
    
            
    def _reveal(self, currplayer, winner): 
        # Logic from original
        if self._checkPokerType(self.history[0], (currplayer-3)%4) != "suspect":
            mult = len(self.history[0])
        else:
            divided, _ = self._checkThrow(self.history[0], (currplayer-3)%4, check=False)
            divided.sort(key=lambda x: len(x), reverse=True)
            if len(divided[0]) >= 4: mult = len(divided[0]) * 2
            elif len(divided[0]) == 2: mult = 4
            else: mult = 2

        publicscore = 0
        for pok in self.covered_card: # Fixed var name
            p = self._id2name(pok)
            if p[1] == "5": publicscore += 5
            elif p[1] == "0" or p[1] == "K": publicscore += 10
        
        self._reward(publicscore*mult, winner)        
    
    def _setMajor(self):
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self.Major = ['jo', 'Jo']
        if self.level in self.point_order:
             self.point_order.remove(self.level)
             
        if self.major != 'n': 
            self.Major = [self.major+point for point in self.point_order if point != self.level] + [suit + self.level for suit in self.suit_set if suit != self.major] + [self.major + self.level] + self.Major
        else: 
            self.Major = [suit + self.level for suit in self.suit_set] + self.Major
        
    def _checkPokerType(self, poker, currplayer):
        # Copied from original
        level = self.level
        poker = [self._id2name(p) for p in poker]
        if len(poker) == 1: return "single"
        if len(poker) == 2:
            if poker[0] == poker[1]: return "pair"
            else: return "suspect"
        if len(poker) % 2 == 0: 
            count = Counter(poker)
            if "jo" in count.keys() and "Jo" in count.keys() and count['jo'] == 2 and count['Jo'] == 2 and len(poker) == 4:
                return "tractor"
            elif "jo" in count.keys() or "Jo" in count.keys(): return "suspect"
            for v in count.values(): 
                if v != 2: return "suspect"
            pointpos = []
            suit = list(count.keys())[0][0] 
            for k in count.keys():
                if k[0] != suit or k[1] == level: return "suspect"
                pointpos.append(self.point_order.index(k[1])) 
            pointpos.sort()
            for i in range(len(pointpos)-1):
                if pointpos[i+1] - pointpos[i] != 1: return "suspect"
            return "tractor" 
        return "suspect"

    def _checkBigger(self, poker, currplayer):
        # Copied
        own = self.player_decks
        level = self.level
        major = self.major
        # tyPoker = self._checkPokerType(poker, currplayer) # Already str
        # poker is list[str] here usually called from checkThrow
        # BUT original checkBigger assumes input is list[str] from checkThrow, OR list[int] wrapper?
        # Re-reading original: checkBigger called with 'poktype' which is list[str] from outpok.
        # But wait, original _checkPokerType took list[int].
        # In checkThrow: pok = [self._id2name(p) for p in poker].
        # In checkBigger: poker input is list[str] from checkThrow.
        
        # FIX: The original code logic for types was slightly mixed.
        # Let's ensure types are consistent.
        # If input is list[str], don't convert.
        if type(poker[0]) == int:
             poker = [self._id2name(p) for p in poker]
             
        own_pok = [[self._id2name(num) for num in hold] for hold in own]
        if poker[0] in self.Major: 
            for i in range(len(own_pok)):
                if i == currplayer: continue
                hold = own_pok[i]
                major_pok = [pok for pok in hold if pok in self.Major]
                count = Counter(major_pok)
                if len(poker) <= 2:
                    if poker[0][1] == level and poker[0][0] != major: 
                        if major == 'n': 
                            for k,v in count.items(): 
                                if (k == 'jo' or k == 'Jo') and v >= len(poker): return True
                        else:
                            for k,v in count.items():
                                if (k == 'jo' or k == 'Jo' or k == major + level) and v >= len(poker): return True
                    else: 
                        for k,v in count.items():
                            if self.Major.index(k) > self.Major.index(poker[0]) and v >= len(poker): return True
                else: 
                    if "jo" in poker: return False 
                    if len(poker) == 4 and "jo" in count.keys() and "Jo" in count.keys():
                        if count["jo"] == 2 and count["Jo"] == 2: return True
                    pos = []
                    for k, v in count.items():
                        if v == 2:
                            if k != 'jo' and k != 'Jo' and k[1] != level and self.point_order.index(k[1]) > self.point_order.index(poker[-1][1]): 
                                pos.append(self.point_order.index(k[1]))
                    if len(pos) >= 2:
                        pos.sort()
                        tmp = 0
                        suc_flag = False
                        for i in range(len(pos)-1):
                            if pos[i+1]-pos[i] == 1:
                                if not suc_flag:
                                    tmp = 2
                                    suc_flag = True
                                else:
                                    tmp += 1
                                if tmp >= len(poker)/2: return True
                            elif suc_flag:
                                tmp = 0
                                suc_flag = False
        else: 
            suit = poker[0][0]
            for i in range(len(own_pok)):
                if i == currplayer: continue
                hold = own_pok[i]
                suit_pok = [pok for pok in hold if pok[0] == suit and pok[1] != level]
                count = Counter(suit_pok)
                if len(poker) <= 2:
                    for k, v in count.items():
                        if self.point_order.index(k[1]) > self.point_order.index(poker[0][1]) and v >= len(poker): return True
                else:
                    pos = []
                    for k, v in count.items():
                        if v == 2:
                            if self.point_order.index(k[1]) > self.point_order.index(poker[-1][1]):
                                pos.append(self.point_order.index(k[1]))
                    if len(pos) >= 2:
                        pos.sort()
                        tmp = 0
                        suc_flag = False
                        for i in range(len(pos)-1):
                            if pos[i+1]-pos[i] == 1:
                                if not suc_flag:
                                    tmp = 2
                                    suc_flag = True
                                else:
                                    tmp += 1
                                if tmp >= len(poker)/2: return True
                            elif suc_flag:
                                tmp = 0
                                suc_flag = False
        return False

    def _checkThrow(self, poker, currplayer, check=False):
        # Copied
        own = self.player_decks
        level = self.level
        major = self.major
        ilcnt = 0
        
        # Fix input type
        if type(poker[0]) == int:
            pok = [self._id2name(p) for p in poker]
        else:
            pok = poker
            
        outpok = []
        failpok = []
        count = Counter(pok)
        if check:
            if list(count.keys())[0] in self.Major: 
                for p in count.keys():
                    if p not in self.Major: self._raise_error(currplayer, "INVALID_POKERTYPE")
            else: 
                suit = list(count.keys())[0][0] 
                for k in count.keys():
                    if k[0] != suit: self._raise_error(currplayer, "INVALID_POKERTYPE")
        pos = []
        tractor = []
        suit = ''
        for k, v in count.items():
            if v == 2:
                if k != 'jo' and k != 'Jo' and k[1] != level: 
                    pos.append(self.point_order.index(k[1]))
                    suit = k[0]
        if len(pos) >= 2:
            pos.sort()
            tmp = []
            suc_flag = False
            for i in range(len(pos)-1):
                if pos[i+1]-pos[i] == 1:
                    if not suc_flag:
                        tmp = [suit + self.point_order[pos[i]], suit + self.point_order[pos[i]], suit + self.point_order[pos[i+1]], suit + self.point_order[pos[i+1]]]
                        del count[suit + self.point_order[pos[i]]]
                        del count[suit + self.point_order[pos[i+1]]] 
                        suc_flag = True
                    else:
                        tmp.extend([suit + self.point_order[pos[i+1]], suit + self.point_order[pos[i+1]]])
                        del count[suit + self.point_order[pos[i+1]]]
                elif suc_flag:
                    tractor.append(tmp)
                    suc_flag = False
            if suc_flag:
                tractor.append(tmp)
        for k,v in count.items(): 
            outpok.append([k for i in range(v)])
        outpok.extend(tractor)

        if check:
            for poktype in outpok:
                if self._checkBigger(poktype, currplayer): 
                    ilcnt += len(poktype)
                    failpok.append(poktype)  
        
        if ilcnt > 0:
            finalpok = []
            kmin = ""
            for poktype in failpok:
                getmark = poktype[-1] 
                if kmin == "":
                    finalpok = poktype
                    kmin = getmark
                elif kmin in self.Major: 
                    if self.Major.index(getmark) < self.Major.index(kmin):
                        finalpok = poktype
                        kmin = getmark
                else: 
                    if self.point_order.index(getmark[1]) < self.point_order.index(kmin[1]):
                        finalpok = poktype
                        kmin = getmark
            finalpok = [[finalpok[0]]]
        else: 
            finalpok = outpok

        return finalpok, ilcnt 
        
    def _checkRes(self, poker, own):
        # Copied
        level = self.level
        if type(poker[0]) == int:
            pok = [self._id2name(p) for p in poker]
        else: pok = poker
        own_pok = [self._id2name(p) for p in own]
        if pok[0] in self.Major:
            major_pok = [pok for pok in own_pok if pok in self.Major]
            count = Counter(major_pok)
            if len(poker) <= 2:
                for v in count.values():
                    if v >= len(poker): return True
            else: 
                pos = []
                for k, v in count.items():
                    if v == 2:
                        if k != 'jo' and k != 'Jo' and k[1] != level: 
                            pos.append(self.point_order.index(k[1]))
                if len(pos) >= 2:
                    pos.sort()
                    tmp = 0
                    suc_flag = False
                    for i in range(len(pos)-1):
                        if pos[i+1]-pos[i] == 1:
                            if not suc_flag:
                                tmp = 2
                                suc_flag = True
                            else:
                                tmp += 1
                            if tmp >= len(poker)/2: return True
                        elif suc_flag:
                            tmp = 0
                            suc_flag = False
        else:
            suit = pok[0][0]
            suit_pok = [pok for pok in own_pok if pok[0] == suit and pok[1] != level]
            count = Counter(suit_pok)
            if len(poker) <= 2:
                for v in count.values():
                    if v >= len(poker): return True
            else:
                pos = []
                for k, v in count.items():
                    if v == 2:
                        pos.append(self.point_order.index(k[1]))
                if len(pos) >= 2:
                    pos.sort()
                    tmp = 0
                    suc_flag = False
                    for i in range(len(pos)-1):
                        if pos[i+1]-pos[i] == 1:
                            if not suc_flag:
                                tmp = 2
                                suc_flag = True
                            else:
                                tmp += 1
                            if tmp >= len(poker)/2: return True
                        elif suc_flag:
                            tmp = 0
                            suc_flag = False
        return False
        
    def _checkWinner(self, currplayer):
        # Copied from original, simplified for readability but logic kept
        # ... (Large logic block, same as original) ...
        # Since I'm pasting the whole file content, I must include it.
        # ...
        level = self.level
        major = self.major
        history = self.history
        histo = history + []
        hist = [[self._id2name(p) for p in x] for x in histo]
        score = 0 
        for move in hist:
            for pok in move:
                if pok[1] == "5": score += 5
                elif pok[1] == "0" or pok[1] == "K": score += 10
        win_seq = 0 
        win_move = hist[0] 
        tyfirst = self._checkPokerType(history[0], currplayer)
        if tyfirst == "suspect": 
            first_parse, _ = self._checkThrow(history[0], currplayer, check=False)
            first_parse.sort(key=lambda x: len(x), reverse=True)
            for i in range(1,4):
                move_parse, r = self._checkThrow(history[i], currplayer, check=False)
                move_parse.sort(key=lambda x: len(x), reverse=True)
                move_cnt = [len(x) for x in move_parse]
                matching = True
                for poktype in first_parse: 
                    if move_cnt[0] >= len(poktype):
                        move_cnt[0] -= len(poktype)
                        move_cnt.sort(reverse=True)
                    else:
                        matching = False
                        break
                if not matching: continue
                if hist[i][0] not in self.Major: continue
                if win_move[0] not in self.Major and hist[i][0] in self.Major: 
                    win_move = hist[i]
                    win_seq = i
                elif len(first_parse[0]) >= 4: 
                    if major == 'n': continue
                    win_parse, s = self._checkThrow(history[win_seq], currplayer, check=False)
                    win_parse.sort(key=lambda x: len(x), reverse=True)
                    if self.Major.index(win_parse[0][-1]) < self.Major.index(move_parse[0][-1]):
                        win_move = hist[i]
                        win_seq = i
                else: 
                    step = len(first_parse[0])
                    win_count = Counter(win_move)
                    win_max = 0
                    for k,v in win_count.items():
                        if v >= step and self.Major.index(k) >= win_max: 
                            win_max = self.Major.index(k)
                    move_count = Counter(hist[i])
                    move_max = 0
                    for k,v in move_count.items():
                        if v >= step and self.Major.index(k) >= move_max:
                            move_max = self.Major.index(k)
                    if major == 'n': 
                        if self.Major[win_max][1] == level:
                            if self.Major[move_max] == 'jo' or self.Major[move_max] == 'Jo':
                                win_move = hist[i]
                                win_seq = i
                        elif self.Major.index(move_max) > self.Major.index(win_max):
                            win_move = hist[i]
                            win_seq = i
                    elif self.Major[win_max][1] == level and self.Major[win_max][0] != major:
                        if (self.Major[move_max][0] == major and self.Major[move_max][1] == level) or self.Major[move_max] == "jo" or self.Major[move_max] == "Jo":
                            win_move = hist[i]
                            win_seq = i
                    elif self.Major.index(win_max) < self.Major.index(move_max):
                        win_move = hist[i]
                        win_seq = i
        else: 
            for i in range(1, 4):
                if self._checkPokerType(history[i], currplayer) != tyfirst: continue
                if (hist[0][0] in self.Major and hist[i][0] not in self.Major) or (hist[0][0] not in self.Major and (hist[i][0] not in self.Major and hist[i][0][0] != hist[0][0][0])):
                    continue
                elif win_move[0] in self.Major: 
                    if hist[i][0] not in self.Major: continue
                    if major == 'n':
                        if win_move[-1][1] == level:
                            if hist[i][-1] == 'jo' or hist[i][-1] == 'Jo': 
                                win_move = hist[i]
                                win_seq = i
                        elif self.Major.index(hist[i][-1]) > self.Major.index(win_move[-1]):
                            win_move = hist[i]
                            win_seq = i
                    else:
                        if win_move[-1][0] != major and win_move[-1][1] == level:
                            if (hist[i][-1][0] == major and hist[i][-1][1] == level) or hist[i][-1] == 'jo' or hist[i][-1] == 'Jo':
                                win_move = hist[i]
                                win_seq = i
                        elif self.Major.index(hist[i][-1]) > self.Major.index(win_move[-1]):
                            win_move = hist[i]
                            win_seq = i
                else: 
                    if hist[i][0] in self.Major: 
                        win_move = hist[i]
                        win_seq = i
                    elif self.point_order.index(win_move[0][-1]) < self.point_order.index(hist[i][0][-1]):
                        win_move = hist[i]
                        win_seq = i
        win_id = (currplayer - 3 + win_seq) % 4
        self._reward(win_id, score)
        return win_id

    def _reward(self, player, points):
        if (player-self.banker_pos) % 2 != 0: # farmer getting points
            self.score += points
        self.reward = {}
        for i in range(4):
            if (i-player) % 2 == 0:
                self.reward[self.agent_names[i]] = points
            else:
                self.reward[self.agent_names[i]] = -points

    def _punish(self, player, points):
        if (player-self.banker_pos) % 2 != 0:
            self.score -= points
        else:
            self.score += points
    def _raise_error(self, player, msg):
        raise Error(f"Player {player}: {msg}")
    
    def _checkLegalMove(self, chosen_ids, currplayer):
        """
        根据当前阶段和历史，修正/确认一个动作对应的真实出牌列表（牌名列表）。

        约定：
        - 训练时，Actor 传进来的 `chosen_ids` 一定是当前 `action_options` 中某个元素
          对应的那一组牌的 ID（即已经是 move_generator 挑出来的“形式合法动作”）。
        - 因此这里不再重复做“跟牌是否合法”的细节校验（那个已经在
          `_get_action_options` + `move_generator` 中完成）。
        - 我们只在「首家出牌」时处理“甩牌”的判定，让它和论文 / 原始环境一致。
        """

        # 把 id 转回牌名（便于和 _checkThrow 等逻辑复用）
        deck_names = [self._id2name(p) for p in chosen_ids]

        # 如果这一墩是首家出牌（history 为空或已经打完一整墩重新开始）
        if len(self.history) == 0 or len(self.history) == 4:
            # 使用已有的甩牌判定逻辑：
            #   - check=True 表示按照原始环境里的规则，判断这一手甩牌是否合法
            #   - final_groups: List[List[str]]，表示最终被允许“甩出去”的若干牌型
            #   - ilcnt: 非 0 表示有不合法部分，被自动缩减为较小的合法牌型
            final_groups, ilcnt = self._checkThrow(deck_names, currplayer, check=True)

            # _checkThrow 的返回有两种典型形态：
            #   1) 多个牌型组成的甩牌：例如 [[tractor1], [pair1], [pair2], ...]
            #   2) 被裁剪成单一牌型：例如 [[pair1]]
            # 对于 env 来说，我们只需要把所有这些牌 flatten 成一个出牌集合即可。
            real_names = []
            for group in final_groups:
                # group 本身就是一个牌名列表
                real_names.extend(group)

            # 注意：real_names 是牌名列表（例如 ["sA", "sA", "sK", "sK"]）
            return real_names

        # 非首家出牌：
        # 这里的合法性已经由 move_generator.gen_single / gen_pair / gen_tractor / gen_throw
        # 严格保证了（包括跟牌、同花色、主牌优先等复杂规则）。
        # 我们只需要把 id 转回牌名即可。
        return deck_names



    
