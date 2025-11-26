from multiprocessing import Process
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import TractorEnv
from model import CNNModel

from wrapper import cardWrapper

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
                    'action_mask': [],
                    'stage': [] # Add stage storage
                },
                'action' : [],
                'reward' : [],
                'value' : []
            } for agent_name in env.agent_names}
            done = False
            while not done:
                state = {}
                player = obs['id']
                agent_name = env.agent_names[player]
                agent_data = episode_data[agent_name]
                
                # Wrapper now returns stage
                obs_mat, action_mask, stage_val = self.wrapper.obsWrap(obs, action_options)
                
                agent_data['state']['observation'].append(obs_mat)
                agent_data['state']['action_mask'].append(action_mask)
                agent_data['state']['stage'].append(stage_val)
                
                state['observation'] = torch.tensor(obs_mat, dtype = torch.float).unsqueeze(0)
                state['action_mask'] = torch.tensor(action_mask, dtype = torch.float).unsqueeze(0)
                state['stage'] = torch.tensor(stage_val, dtype = torch.long).unsqueeze(0) # Add stage to input
                
                model.train(False) # Batch Norm inference mode
                with torch.no_grad():
                    logits, value = model(state)
                    action_dist = torch.distributions.Categorical(logits = logits)
                    action = action_dist.sample().item()
                    value = value.item()
                    
                agent_data['action'].append(action)
                agent_data['value'].append(value)
                agent_data['reward'].append(0) # Initialize reward for this step
                
                # interpreting actions
                # Pass index directly as per new Env interface
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
                stage = np.array(agent_data['state']['stage'][:length], dtype=np.int64) 
                
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
                        'action_mask': mask,
                        'stage': stage # Push stage
                    },
                    'action': actions,
                    'adv': advantages,
                    'target': td_target
                })
