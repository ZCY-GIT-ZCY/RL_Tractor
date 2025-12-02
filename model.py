import torch
from torch import nn

class CNNModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Conv2d(256, 32, 3, 1, 1, bias = False),
            nn.ReLU(True),
            nn.Flatten()
        )
        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 14, 256),
            nn.ReLU(True),
            nn.Linear(256, 54)
        )
        
        self._value_branch = nn.Sequential(
            nn.Linear(32 * 4 * 14, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def  forward(self, input_dict):
        # observation: (B, 128, 4, 14); action_mask: (B, 54)
        # 原实现对 mask 取 log 在 0 处产生 -inf，再经 autocast(fp16) 与 label smoothing 可能形成 NaN/Inf。
        # 改为使用一个 FP16 可表示的稳定大负数进行屏蔽，不调用 log。
        obs = input_dict["observation"].float()
        mask = input_dict["action_mask"].float()
        hidden = self._tower(obs)
        logits = self._logits(hidden).float()
        # 选择 -1e4：在 FP16 范围内 (>-65504)，softmax 后概率≈0，不会溢出为 -inf。
        invalid = mask <= 0
        masked_logits = logits.masked_fill(invalid, -100)
        # 可选的数值断言（仅首批使用时可在调用端添加）：
        # assert torch.isfinite(masked_logits).all(), masked_logits.min()
        value = self._value_branch(hidden)
        return masked_logits, value
