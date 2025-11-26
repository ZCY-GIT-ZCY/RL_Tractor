from model import CNNModel
import torch
from torch import nn, optim
import numpy as np
# EPOCH = int(1e5)
THRESHOLD = 1e-4
SEED = 123
VERBOSE = 500
SAVE_FREQ = 2000
SAVE_PATH = "check_points/"
DATA_PATH = "Pre_trained_Data/data.npz"
data = np.load(DATA_PATH)

obs = data["observation"]
mask = data["action_mask"]
stage = data["stage"]
labels = data["action"]

num_samples = obs.shape[0]
rng = np.random.default_rng(SEED)
indices = rng.permutation(num_samples)
split = max(1, int(num_samples * 0.8))
train_idx = indices[:split]
test_idx = indices[split:] if split < num_samples else indices[-1:]

obs_train = torch.from_numpy(obs[train_idx]).type(torch.float32)
obs_test = torch.from_numpy(obs[test_idx]).type(torch.float32)
mask_train = torch.from_numpy(mask[train_idx]).type(torch.float32)
mask_test = torch.from_numpy(mask[test_idx]).type(torch.float32)
stage_train = torch.from_numpy(stage[train_idx]).type(torch.long)
stage_test = torch.from_numpy(stage[test_idx]).type(torch.long)
Y_train = torch.from_numpy(labels[train_idx]).type(torch.long)
Y_test = torch.from_numpy(labels[test_idx]).type(torch.long)

CNN_Action_model = CNNModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN_Action_model.parameters(), 5e-5)

idx = 0
current_loss = float("inf")

def build_state(observation, action_mask, stage_tensor):
    return {
        "observation": observation,
        "action_mask": action_mask,
        "stage": stage_tensor,
    }

while float(current_loss) >= THRESHOLD:
    idx += 1
    CNN_Action_model.train()
    optimizer.zero_grad()

    train_state = build_state(obs_train, mask_train, stage_train)
    y_logits, _ = CNN_Action_model(train_state)
    current_loss = loss_fn(y_logits, Y_train)
    current_loss.backward()
    optimizer.step()

    CNN_Action_model.eval()
    with torch.inference_mode():
        test_state = build_state(obs_test, mask_test, stage_test)
        y_test_logits, _ = CNN_Action_model(test_state)
        test_loss = loss_fn(y_test_logits, Y_test)

    if idx % VERBOSE == 0:
        print(
            f"Epoch {idx} Training Progress as follows:",
            f"Training Loss: {current_loss}.",
            f"Test Loss: {test_loss}.",
            sep="\n",
        )
    if idx % SAVE_FREQ == 0:
        torch.save(CNN_Action_model.state_dict(), SAVE_PATH + f"spv_model_{idx}.pt")
        print(
            f"Training completed after {idx} epochs.",
            f"Model saved to {SAVE_PATH}spv_model_{idx}.pt",
            sep="\n",
        )
            
    
    
