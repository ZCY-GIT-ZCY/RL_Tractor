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
        
        # Three heads for three stages
        self._logits_snatch = nn.Sequential(
            nn.Linear(32 * 4 * 14, 256),
            nn.ReLU(True),
            nn.Linear(256, 54)
        )
        self._logits_bury = nn.Sequential(
            nn.Linear(32 * 4 * 14, 256),
            nn.ReLU(True),
            nn.Linear(256, 54)
        )
        self._logits_play = nn.Sequential(
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

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        stage = input_dict["stage"].long() # Shape (B,) or (B, 1)
        
        hidden = self._tower(obs)
        
        # Dispatch based on stage
        # Since we might have a batch with different stages (though unlikely in sync training, possible in buffers)
        # We compute all heads? Or gather?
        # Computing all is 3x compute on head. Masking is easier.
        
        l_snatch = self._logits_snatch(hidden)
        l_bury = self._logits_bury(hidden)
        l_play = self._logits_play(hidden)
        
        # Select based on stage
        # stage shape: (B, 1) or (B,)
        if len(stage.shape) == 1:
            stage = stage.unsqueeze(1) # (B, 1)
            
        # Broadcast stage to (B, 54)
        # This approach: construct logits tensor by stacking?
        # Or just use torch.where
        
        # Optimized: logits = torch.where(stage==0, l_snatch, torch.where(stage==1, l_bury, l_play))
        # But we have 3 stages.
        # Let's stack them: (B, 3, 54)
        stacked = torch.stack([l_snatch, l_bury, l_play], dim=1)
        # Gather: index along dim 1
        # index shape needs to be (B, 1, 54)
        idx = stage.unsqueeze(-1).expand(-1, -1, 54)
        logits = stacked.gather(1, idx).squeeze(1)
        
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        
        value = self._value_branch(hidden)
        return masked_logits, value
