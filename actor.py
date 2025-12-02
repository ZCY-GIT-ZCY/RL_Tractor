from multiprocessing import Process
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import TractorEnv
from model import CNNModel

from wrapper import cardWrapper
from declaration import decide_declaration, decide_overcall
from kitty import select_kitty_cards

class Actor(Process):
    
    def __init__(self, config, replay_buffer):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        
    def run(self):
        torch.set_num_threads(1)
    
        # connect to model pool
        model_pool = ModelPoolClient(self.config['model_pool_name'])
        
        # create network model
        model = CNNModel()
        
        # load initial model
        version = model_pool.get_latest_model()
        state_dict = model_pool.load_model(version)
        model.load_state_dict(state_dict)
        
        # collect data
        env = TractorEnv()
        self.wrapper = cardWrapper()
        policies = {player : model for player in env.agent_names} # all four players use the latest model
        
        for episode in range(self.config['episodes_per_actor']):
            # update model
            latest = model_pool.get_latest_model()
            if latest['id'] > version['id']:
                state_dict = model_pool.load_model(latest)
                model.load_state_dict(state_dict)
                version = latest
            
            # run one episode and collect data
            obs, action_options = env.reset(major='r')
            episode_data = {agent_name: {
                'state' : {
                    'observation': [],
                    'action_mask': []
                },
                'action' : [],
                'reward' : [],
                'value' : []
            } for agent_name in env.agent_names}
            done = False
            while not done:
                stage = obs.get('stage', TractorEnv.STAGE_PLAY)
                player = obs['id']
                agent_name = env.agent_names[player]
                agent_data = episode_data[agent_name]

                # handle declaration / kitty stages via rule-based heuristics
                if stage == TractorEnv.STAGE_SNATCH:
                    action = self._select_declaration_action(env, obs, action_options)
                    response = {'player': player, 'action': action}
                    next_obs, action_options, rewards, done = env.step(response)
                    obs = next_obs
                    continue
                if stage == TractorEnv.STAGE_BURY:
                    action = self._select_bury_action(env, action_options)
                    response = {'player': player, 'action': action}
                    next_obs, action_options, rewards, done = env.step(response)
                    obs = next_obs
                    continue

                state = {}
                obs_mat, action_mask = self.wrapper.obsWrap(obs, action_options)

                agent_data['state']['observation'].append(obs_mat)
                agent_data['state']['action_mask'].append(action_mask)

                state['observation'] = torch.tensor(obs_mat, dtype = torch.float).unsqueeze(0)
                state['action_mask'] = torch.tensor(action_mask, dtype = torch.float).unsqueeze(0)

                model.train(False) # Batch Norm inference mode
                with torch.no_grad():
                    logits, value = model(state)
                    action_dist = torch.distributions.Categorical(logits = logits)
                    action = action_dist.sample().item()
                    value = value.item()

                agent_data['action'].append(action)
                agent_data['value'].append(value)
                agent_data['reward'].append(0) # Initialize reward for this step

                response = {'player': player, 'action': action}

                # interact with env
                next_obs, action_options, rewards, done = env.step(response)
                if rewards:
                    # rewards are added per four moves (1 move for each player) on all four players
                    for agent_name in rewards: 
                        # Add to the last reward entry (credit assignment)
                        if len(episode_data[agent_name]['reward']) > 0:
                            episode_data[agent_name]['reward'][-1] += rewards[agent_name]
                obs = next_obs
            print(self.name, 'Episode', episode, 'Model', latest['id'], 'Reward', rewards)
            
            # postprocessing episode data for each agent
            for agent_name, agent_data in episode_data.items():
                # Ensure alignment (though logic above should keep them aligned)
                length = min(len(agent_data['action']), len(agent_data['reward']))
                
                obs = np.stack(agent_data['state']['observation'][:length])
                mask = np.stack(agent_data['state']['action_mask'][:length])
                actions = np.array(agent_data['action'][:length], dtype = np.int64)
                rewards = np.array(agent_data['reward'][:length], dtype = np.float32)
                values = np.array(agent_data['value'][:length], dtype = np.float32)
                # Next value estimation
                # If done, next value is 0. If truncated, use bootstrap? Assuming done at episode end.
                next_values = np.array(agent_data['value'][1:length+1] + [0], dtype = np.float32)
                # Note: agent_data['value'] might be longer by 1 if we appended? No, 1-to-1.
                # Actually next_values should be shifted. 
                # value[t+1] corresponds to reward[t] + gamma * value[t+1].
                # If length is N, values has N. values[1:] has N-1. + [0] makes N. Correct.
                
                td_target = rewards + next_values * self.config['gamma']
                td_delta = td_target - values
                advs = []
                adv = 0
                for delta in td_delta[::-1]:
                    adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                    advs.append(adv) # GAE
                advs.reverse()
                advantages = np.array(advs, dtype = np.float32)
                
                # send samples to replay_buffer (per agent)
                self.replay_buffer.push({
                    'state': {
                        'observation': obs,
                        'action_mask': mask
                    },
                    'action': actions,
                    'adv': advantages,
                    'target': td_target
                })

    def _select_declaration_action(self, env: TractorEnv, obs, action_options):
        """
        Use rule-based heuristics for declaring or overcalling trump during the snatch stage.
        """
        level = env.level
        deck = obs.get('deck', [])
        if env.reporter is None:
            candidate = decide_declaration(deck, level)
            if not candidate:
                return 0  # pass
            target = [candidate + level]
            idx = self._find_option_index(action_options, target)
            return idx if idx is not None else 0

        candidate = decide_overcall(deck, level, env.major or 'n')
        if not candidate:
            return 0
        if candidate == 'n':
            for joker in ("Jo", "jo"):
                idx = self._find_option_index(action_options, [joker, joker])
                if idx is not None:
                    return idx
            return 0
        target = [candidate + level, candidate + level]
        idx = self._find_option_index(action_options, target)
        return idx if idx is not None else 0

    def _select_bury_action(self, env: TractorEnv, action_options):
        """
        Use the kitty heuristics to pick a single card to bury at this step.
        """
        banker = env.banker_pos
        if banker is None:
            return 0
        bury_count = env.bury_left
        deck_ids = list(env.player_decks[banker])
        selected = select_kitty_cards(deck_ids, env.level, env.major or 'n', bury_count)
        if not selected:
            return 0
        target_name = env._id2name(selected[0])
        idx = self._find_option_index(action_options, [target_name])
        if idx is not None:
            return idx
        return 0

    @staticmethod
    def _find_option_index(action_options, target_cards):
        if not action_options:
            return None
        for idx, option in enumerate(action_options):
            if len(option) != len(target_cards):
                continue
            if option == target_cards:
                return idx
        return None
