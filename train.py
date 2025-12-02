from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner

if __name__ == '__main__':
    config = {
        'replay_buffer_size': 60000,  # max number of samples in replay buffer
        'replay_buffer_episode': 1000,  # max number of episodes in replay buffer
        'model_pool_size': 20,  # max number of models in model pool
        'model_pool_name': 'model-pool',   # name of the model pool
        'num_actors': 4,  # number of parallel actors
        'episodes_per_actor': 1000, # episodes per actor before restarting
        'gamma': 0.98,  # discount factor
        'lambda': 0.95,  # GAE lambda
        'min_sample': 8000,   # min samples before learner starts
        'batch_size': 256,   # batch size for learner
        'epochs': 3,  # number of epochs per update
        'clip': 0.2,   # PPO clip parameter
        'lr': 1e-4, # learning rate
        'value_coeff': 1,   # value loss coefficient
        'entropy_coeff': 0.01,  # entropy loss coefficient
        'device': 'cuda',  # device to run the model on
        'ckpt_save_interval': 1800,  # checkpoint save interval in seconds
        'ckpt_save_path': 'checkpoint/',  # checkpoint save path
        'init_model_path': 'Code/Pre_trained_Data/Overfit/spv_model_epoch22_1.5072.pt',  # optional initial weights
    }
    
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    learner = Learner(config, replay_buffer)
    
    for actor in actors: actor.start()
    learner.start()
    
    for actor in actors: actor.join()
    learner.terminate()
