from multiprocessing import Process
import time
import numpy as np
import torch
import os
from torch.nn import functional as F

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel

class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        # Shared progress / coordination
        self._iter_counter = config.get('learner_iter_counter')
        self._done_event = config.get('learner_done_event')
        self._stop_event = config.get('stop_event')
    
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
        target_iters = int(self.config.get('learner_iterations', 0))
        stop_reason = None
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

            if self.config.get('normalize_adv', False):
                adv_mean = advs.mean()
                adv_std = advs.std(unbiased=False)
                advs = (advs - adv_mean) / (adv_std + 1e-8)
            
            # 监控信息（轻量）：迭代与缓冲区
            # print 改为 tqdm.postfix 展示，更整洁
            
            # calculate PPO loss
            model.train(True) # Batch Norm training mode
            old_logits, _ = model(states)
            if not torch.isfinite(old_logits).all():
                print("[Learner] Non-finite logits detected before update, aborting training cycle.")
                stop_reason = "non_finite_old_logits"
                break
            old_log_probs = F.log_softmax(old_logits, dim=1).gather(1, actions).detach()
            last_policy_loss = None
            last_value_loss = None
            last_entropy_loss = None
            last_total_loss = None
            invalid_update = False
            for _ in range(self.config['epochs']):
                logits, values = model(states)
                if (not torch.isfinite(logits).all()) or (not torch.isfinite(values).all()):
                    print("[Learner] Non-finite tensor detected during update, stopping learner loop.")
                    stop_reason = "non_finite_update"
                    invalid_update = True
                    break
                action_dist = torch.distributions.Categorical(logits=logits)
                log_probs = F.log_softmax(logits, dim=1).gather(1, actions)
                log_ratio = (log_probs - old_log_probs).clamp(-20, 20)
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                entropy_loss = -torch.mean(action_dist.entropy())
                loss = policy_loss + self.config['value_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                max_grad_norm = self.config.get('max_grad_norm', 0)
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                # 保存最新一轮的可视化指标
                last_policy_loss = float(policy_loss.detach().item())
                last_value_loss = float(value_loss.detach().item())
                last_entropy_loss = float((-entropy_loss).detach().item())  # 显示正的熵值
                last_total_loss = float(loss.detach().item())
            if invalid_update:
                break

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
                # checkpoint saved
            iterations += 1
            # Update shared counter
            if self._iter_counter is not None:
                try:
                    with self._iter_counter.get_lock():
                        self._iter_counter.value += 1
                except Exception:
                    pass
            # Check for completion
            if target_iters and iterations >= target_iters:
                stop_reason = "max_iterations"
                break

        self._finalize(model, iterations, stop_reason)

    def _finalize(self, model, iterations, stop_reason):
        model_cpu = model.to('cpu')
        ckpt_dir = self.config['ckpt_save_path']
        os.makedirs(ckpt_dir, exist_ok=True)
        final_path = os.path.join(ckpt_dir, f"model_final_iter{iterations}.pt")
        torch.save(model_cpu.state_dict(), final_path)
        print(f"[Learner] Saved final model to {final_path} (reason={stop_reason})")

        if self._stop_event is not None:
            try:
                self._stop_event.set()
            except Exception:
                pass
        if self._done_event is not None:
            try:
                self._done_event.set()
            except Exception:
                pass
