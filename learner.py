from multiprocessing import Process
import time
import numpy as np
import torch
import os
from torch.nn import functional as F
from tqdm import tqdm

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel

class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
    
    def run(self):
        # create model pool
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])
        
        # initialize model params
        device = torch.device(self.config['device'])
        model = CNNModel()
        init_model_path = self.config.get('init_model_path')
        if init_model_path:
            if os.path.isfile(init_model_path):
                state_dict = torch.load(init_model_path, map_location='cpu')
                try:
                    model.load_state_dict(state_dict)
                    print(f"[Learner] Loaded initial weights from {init_model_path}")
                except RuntimeError as err:
                    print(f"[Learner] Failed to load initial weights: {err}")
            else:
                print(f"[Learner] init_model_path {init_model_path} not found, training from scratch.")
        
        # send to model pool
        model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
        model = model.to(device)
        
        # training
        optimizer = torch.optim.Adam(model.parameters(), lr = self.config['lr'])
        
        # wait for initial samples
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)
        
        cur_time = time.time()
        iterations = 0
        # tqdm 监控：未知总长度的持续训练，用动态后缀展示指标
        pbar = tqdm(total=None, desc="Learner", mininterval=0.5, smoothing=0.1)
        while True:
            # sample batch
            batch = self.replay_buffer.sample(self.config['batch_size'])
            obs = torch.tensor(batch['state']['observation']).to(device)
            mask = torch.tensor(batch['state']['action_mask']).to(device)
            states = {
                'observation': obs,
                'action_mask': mask
            }
            actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
            advs = torch.tensor(batch['adv']).to(device)
            targets = torch.tensor(batch['target']).to(device)
            
            # 监控信息（轻量）：迭代与缓冲区
            # print 改为 tqdm.postfix 展示，更整洁
            
            # calculate PPO loss
            model.train(True) # Batch Norm training mode
            old_logits, _ = model(states)
            old_probs = F.softmax(old_logits, dim = 1).gather(1, actions)
            old_log_probs = torch.log(old_probs).detach()
            last_policy_loss = None
            last_value_loss = None
            last_entropy_loss = None
            last_total_loss = None
            for _ in range(self.config['epochs']):
                logits, values = model(states)
                action_dist = torch.distributions.Categorical(logits = logits)
                probs = F.softmax(logits, dim = 1).gather(1, actions)
                log_probs = torch.log(probs)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                entropy_loss = -torch.mean(action_dist.entropy())
                loss = policy_loss + self.config['value_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 保存最新一轮的可视化指标
                last_policy_loss = policy_loss.detach().item()
                last_value_loss = value_loss.detach().item()
                last_entropy_loss = (-entropy_loss).detach().item()  # 显示正的熵值
                last_total_loss = loss.detach().item()

            # push new model
            model = model.to('cpu')
            model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
            model = model.to(device)
            
            # save checkpoints
            t = time.time()
            if t - cur_time > self.config['ckpt_save_interval']:
                path = self.config['ckpt_save_path'] + 'model_%d.pt' % iterations
                if not os.path.exists(self.config['ckpt_save_path']):
                    os.makedirs(self.config['ckpt_save_path'])
                torch.save(model.state_dict(), path)
                cur_time = t
                pbar.write(f"[CKPT] Saved {path}")
            iterations += 1
            # 更新 tqdm 显示
            current_lr = optimizer.param_groups[0]['lr']
            pbar.update(1)
            pbar.set_postfix({
                'iter': iterations,
                'buf': self.replay_buffer.size(),
                'in': self.replay_buffer.stats.get('sample_in', 0),
                'out': self.replay_buffer.stats.get('sample_out', 0),
                'lr': f"{current_lr:.2e}",
                'pol': f"{(last_policy_loss if last_policy_loss is not None else float('nan')):.3f}",
                'val': f"{(last_value_loss if last_value_loss is not None else float('nan')):.3f}",
                'ent': f"{(last_entropy_loss if last_entropy_loss is not None else float('nan')):.3f}",
                'loss': f"{(last_total_loss if last_total_loss is not None else float('nan')):.3f}",
            })
