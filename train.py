from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
from multiprocessing import Value, Event
from tqdm import tqdm
import time

if __name__ == '__main__':
    config = {
        'replay_buffer_size': 70000,  # max number of samples in replay buffer
        'replay_buffer_episode': 1000,  # max number of episodes in replay buffer
        'model_pool_size': 20,  # max number of models in model pool
        'model_pool_name': 'model-pool',   # name of the model pool
        'num_actors': 6,  # number of parallel actors
        'episodes_per_actor': 80000, # episodes per actor before restarting
        'gamma': 0.99,  # discount factor
        'lambda': 0.95,  # GAE lambda
        'min_sample': 10000,   # min samples before learner starts
        'batch_size': 128,   # batch size for learner
        'epochs': 3,  # number of epochs per update
        'clip': 0.2,   # PPO clip parameter
        'lr': 1e-4, # learning rate
        'value_coeff': 1,   # value loss coefficient
        'entropy_coeff': 0.01,  # entropy loss coefficient
        'device': 'cuda',  # device to run the model on
        'ckpt_save_interval': 1800,  # checkpoint save interval in seconds
        'ckpt_save_path': 'checkpoint/',  # checkpoint save path
        'init_model_path': 'Pre_trained_Data/init_model.pt',  # optional initial weights
        # Global progress totals
        'learner_iterations': 100000,  # total learner iterations before stopping
        'normalize_adv': True,
        'max_grad_norm': 1.0,
    }
    
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    # Shared counters and completion events
    actor_episode_counter = Value('i', 0)
    learner_iter_counter = Value('i', 0)
    actor_done = Event()
    learner_done = Event()
    stop_event = Event()

    # Spawn processes
    actors = []
    for i in range(config['num_actors']):
        cfg = dict(config)
        cfg['name'] = 'Actor-%d' % i
        cfg['actor_episode_counter'] = actor_episode_counter
        cfg['episodes_total'] = config['episodes_per_actor'] * config['num_actors']
        cfg['stop_event'] = stop_event
        actor = Actor(cfg, replay_buffer)
        actors.append(actor)
    learner_cfg = dict(config)
    learner_cfg['learner_iter_counter'] = learner_iter_counter
    learner_cfg['learner_done_event'] = learner_done
    learner_cfg['stop_event'] = stop_event
    learner = Learner(learner_cfg, replay_buffer)

    for actor in actors:
        actor.start()
    learner.start()

    # Global progress bars
    try:
        actor_pbar = tqdm(total=config['episodes_per_actor'] * config['num_actors'], desc='Actor Sampling', mininterval=0.5, smoothing=0.1)
        learner_pbar = tqdm(total=config['learner_iterations'], desc='Learner Training', mininterval=0.5, smoothing=0.1)
        last_actor = 0
        last_learner = 0
        while True:
            if stop_event.is_set():
                actor_done.set()
                learner_done.set()
                break

            curr_actor = actor_episode_counter.value
            curr_learner = learner_iter_counter.value
            if curr_actor > last_actor:
                actor_pbar.update(curr_actor - last_actor)
                last_actor = curr_actor
            if curr_learner > last_learner:
                learner_pbar.update(curr_learner - last_learner)
                last_learner = curr_learner

            # Completion checks
            if not actor_done.is_set() and curr_actor >= config['episodes_per_actor'] * config['num_actors']:
                actor_done.set()
            if not learner_done.is_set() and curr_learner >= config['learner_iterations']:
                learner_done.set()

            if actor_done.is_set() and learner_done.is_set():
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()
        pass
    finally:
        actor_pbar.close()
        learner_pbar.close()

    # Graceful shutdown
    for actor in actors:
        actor.join(timeout=1.0)
        if actor.is_alive():
            actor.terminate()
    learner.join(timeout=1.0)
    if learner.is_alive():
        learner.terminate()
