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
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self.Major = ['jo', 'Jo']
        self.level = level
        self.first_round = True 
        self.banker_pos = banker_pos
        self.curr_player = banker_pos # Start dealing/snatching from banker or random?
        
        # Determine Major if fixed, else random or wait for snatch
        if major == 'r': 
            self.major = 'n' # Placeholder until snatched
        else:
            self.major = major

        if self.banker_pos:
            self.first_round = False

        self.reporter = None
        self.snatcher = None
        self.snatch_passes = 0
        
        # Initialize decks
        self.total_deck = [i for i in range(108)] 
        random.shuffle(self.total_deck)
        self.public_card = self.total_deck[100:] 
        self.covered_cards = [] 
        
        # Deal all cards immediately for the staged approach
        self.player_decks = [[] for _ in range(4)]
        for i in range(100):
            card = self.total_deck[i]
            target = i % 4
            self.player_decks[target].append(card)
            
        self._setMajor() # Preliminary major set (might change during snatch)
        self.mv_gen = move_generator(self.level, self.major)
        
        self.score = 0
        self.history = []
        self.played_cards = [[] for _ in range(4)]
        self.reward = None
        self.done = False
        self.round = 0 
        
        # Start at Snatch Stage
        self.stage = self.STAGE_SNATCH
        self.curr_player = random.randint(0, 3) # Random start for snatching
        
        return self._get_obs(self.curr_player), self._get_action_options(self.curr_player)

    def step(self, response): 
        # response: dict{'player': player_id, 'action': action_index} (Note: Input is index now, converted to cards internally if needed)
        self.reward = None
        curr_player = response['player']
        action_idx = response['action'] # This is an INT index now
        
        # Get options to interpret action
        options = self._get_action_options_internal(curr_player) # Need internal method to avoid recursion/state change
        action_cards = options[action_idx]
        
        if self.stage == self.STAGE_SNATCH:
            if len(action_cards) == 0: # Pass
                self.snatch_passes += 1
                if self.snatch_passes >= 3 and self.reporter is not None:
                     # End snatching
                     self._end_snatch()
                elif self.snatch_passes >= 4 and self.reporter is None:
                     # Force end - Random major
                     self.random_pick_major()
                     self._end_snatch()
                else:
                    self.curr_player = (curr_player + 1) % 4
            else: # Snatch/Report
                self.snatch_passes = 0
                if len(action_cards) == 1:
                    self._report(action_cards, curr_player)
                elif len(action_cards) == 2:
                    self._snatch(action_cards, curr_player)
                # If someone snatches, continue bidding until stabilized
                self.curr_player = (curr_player + 1) % 4
                
        elif self.stage == self.STAGE_BURY:
             if curr_player != self.banker_pos:
                 self._raise_error(curr_player, "NOT_BANKER_IN_BURY")
             
             # Action is a single card to bury
             # Convert name to ID
             bury_cards_id = self._name2id_seq(action_cards, self.player_decks[curr_player])
             card_to_bury_id = bury_cards_id[0]
             
             self.player_decks[curr_player].remove(card_to_bury_id)
             self.covered_cards.append(card_to_bury_id)
             
             if len(self.covered_cards) == 8:
                 self.stage = self.STAGE_PLAY
                 self.curr_player = self.banker_pos
                 self.round = 1
                 self.history = []
             else:
                 # Continue burying
                 self.curr_player = self.banker_pos
                 
        elif self.stage == self.STAGE_PLAY:
            self.mode = "play"
            # Logic from original step
            real_action = self._name2id_seq(action_cards, self.player_decks[curr_player])
            # Check legal move (re-verify in case)
            # real_action = self._checkLegalMove(real_action, curr_player) # Assumed legal from options
            self._play(curr_player, real_action)
            self.curr_player = (curr_player + 1) % 4
            
            if len(self.history) == 4: # finishing a round
                winner = self._checkWinner(curr_player)
                self.curr_player = winner
                if len(self.player_decks[0]) == 0: # Ending the game
                    self._reveal(curr_player, winner)
                    self.done = True
            
            # Note: self.round is incremented in original per card play or per trick?
            # Original: self.round += 1 at end of step. round seems to track turns.
            self.round += 1

        if self.reward:
             return self._get_obs(self.curr_player), self._get_action_options(self.curr_player), self.reward, self.done
        return self._get_obs(self.curr_player), self._get_action_options(self.curr_player), None, self.done

    def _end_snatch(self):
        self._setMajor()
        self.mv_gen = move_generator(self.level, self.major)
        self._deliver_public()
        self.stage = self.STAGE_BURY
        self.curr_player = self.banker_pos
        self.covered_cards = []

    def _get_action_options(self, player):
        return self._get_action_options_internal(player)
        
    def _get_action_options_internal(self, player):
        deck = [self._id2name(p) for p in self.player_decks[player]]
        
        if self.stage == self.STAGE_SNATCH:
            return self.mv_gen.gen_snatch(deck, self.reporter)
            
        elif self.stage == self.STAGE_BURY:
            if player == self.banker_pos:
                return self.mv_gen.gen_bury(deck)
            else:
                return [[]] # Dummy for others
                
        elif self.stage == self.STAGE_PLAY:
            if len(self.history) == 4 or len(self.history) == 0: 
                return self.mv_gen.gen_all(deck)
            else:
                tgt = [self._id2name(p) for p in self.history[0]]
                poktype = self._checkPokerType(self.history[0], (player-len(self.history))%4)
                if poktype == "single":
                    return self.mv_gen.gen_single(deck, tgt)
                elif poktype == "pair":
                    return self.mv_gen.gen_pair(deck, tgt)
                elif poktype == "tractor":
                    return self.mv_gen.gen_tractor(deck, tgt)
                elif poktype == "suspect":
                    return self.mv_gen.gen_throw(deck, tgt)
        return []

    def _raise_error(self, player, info):
        raise Error("Player_"+str(player)+": "+info)
        
    def _get_obs(self, player):
        obs = {
            "id": player,
            "stage": self.stage, # Added stage
            "deck": [self._id2name(p) for p in self.player_decks[player]],
            "history": [[self._id2name(p) for p in move] for move in self.history],
            "major": self.Major,
            "played": [[self._id2name(p) for p in move] for move in self.played_cards]
        }
        return obs
    
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
        
    def _report(self, repo_card: list, reporter): 
        # repo_card is list of str names here because comes from options
        # Need name2id to get suit? Or just parse name
        # The original code took list[int] from action
        # Here input is list[str] from options
        repo_name = repo_card[0]
        major_suit = repo_name[0]
        self.major = major_suit
        self.reporter = reporter
        self._setMajor() # Update major list immediately
    
    def _snatch(self, snatch_card: list, snatcher):
        snatch_name = snatch_card[0]
        if snatch_name[1] == 'o': 
            self.major = 'n'
        else:
            self.major = snatch_name[0]
        self.snatcher = snatcher
        self.reporter = snatcher # Snatcher becomes reporter
        self._setMajor()

    def random_pick_major(self): 
        if self.first_round: 
            self.banker_pos = random.choice(range(4))
        self.major = random.choice(self.suit_set)
        self._setMajor()

    def _deliver_public(self): 
        for card in self.public_card:
            self.player_decks[self.banker_pos].append(card)
            
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
        for pok in self.covered_cards: # Fixed var name
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

    def action_intpt(self, action, player):
        # Already handled partially in step, but wrapper calls this
        # action here is LIST of card names
        player_deck = self.player_decks[player]
        # In new version, step takes action INDEX from options.
        # But this function might be used by Actor to log 'action'?
        # Wait, existing Actor calls action_intpt with 'action_cards'.
        # Actor: response = env.action_intpt(action_cards, player)
        # So this function just wraps it into dict.
        # But wait, original action_intpt converts name2id_seq.
        # My step function takes index? 
        # Actually my step function in this new code assumes `response['action']` is `action_index`.
        # BUT the original step takes `response['action']` as list of IDs!
        # Let's align.
        # Actor calls: `response = env.action_intpt(action_cards, player)`
        # Then `env.step(response)`.
        # So action_intpt should package the data needed by step.
        # If I want `step` to use index (for cleaner logic), I should pass index.
        # But `action_cards` is what Actor gets from `action_options[action_index]`.
        # If I want `step` to use `action_index`, Actor needs to pass `action` (int).
        # But `env.step` signature in `actor.py` is fixed unless I change it.
        # Let's change `actor.py` to pass the index directly or let `action_intpt` return index.
        # However, `action_intpt` receives `action_cards` (list of strings). It doesn't know the index!
        # The actor has the index `action`.
        
        # FIX: Modify `actor.py` to pass the index.
        # OR: Modify `action_intpt` to just return card IDs, and `step` uses card IDs to figure out logic.
        # But for Burying, dealing with IDs is annoying.
        # Easier: `step` takes Index.
        # So `action_intpt` is deprecated or needs to change signature?
        # User said "更改...接口".
        # Let's stick to: Step takes Index.
        # So I will modify `actor.py` later.
        # Here `action_intpt` can just return the index if passed?
        # No, let's make `action_intpt` return the dict with `action` being the index.
        # But `action_intpt` in `actor.py` is called with `action_options[action]`.
        # I should change `actor.py` to NOT look up `action_options` before calling `action_intpt`?
        # Or just pass the integer `action` to `action_intpt`.
        
        # New signature for action_intpt to be compatible with my new step:
        # It just passes the index through.
        pass

    # I'll update action_intpt to accept int and pass it.
    # But wait, the previous code used action_intpt to convert str->int IDs.
    # My new step logic uses `_get_action_options_internal` to look up cards from index.
    # So step expects index.
    pass

