from model import CNNModel
import torch
from torch import nn, optim
import numpy as np
from sklearn.model_selection import train_test_split as splt

# EPOCH = int(1e5)
THRESHOLD = 1e-4
SEED = 123
VERBOSE = 500
SAVE_FREQ = 2000
SAVE_PATH = "check_points/"
DATA_PATH = "Pre_trained_Data/data.npy"
data = np.load(DATA_PATH, allow_pickle=True)

X, Y = data[:,0], data[:,1]
CNN_Action_model = CNNModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN_Action_model.parameters(), 5e-5)

X = torch.from_numpy(X).type(torch.float)
Y = torch.from_numpy(Y).type(torch.LongTensor)
X_train, X_test, Y_train, Y_test = splt(X, Y, random_state = SEED, test_size = 0.2)

idx = 0
current_loss = float('inf')
# for idx in range(EPOCH):
while float(current_loss) >= THRESHOLD:
    idx += 1
    CNN_Action_model.train()
    optimizer.zero_grad()
    
    y_logits, _ = CNN_Action_model(X_train)
    current_loss = loss_fn(y_logits, Y_train)
    current_loss.backward()
    optimizer.step()
    
    CNN_Action_model.eval()
    with torch.inference_mode():
        y_test, _ = CNN_Action_model(X_test)
        test_loss = loss_fn(y_test, Y_test)
        
    if idx % VERBOSE == 0:
        print(f"Epoch {idx} Training Progress as follows:",
        f"Training Loss: {current_loss}.",f"Test Loss: {test_loss}.", sep="\n")
    if idx % SAVE_FREQ == 0:
        torch.save(CNN_Action_model.state_dict(), SAVE_PATH + f"spv_model_{idx}.pt")
        print(f"Training completed after {idx} epochs.",
              f"Model saved to {SAVE_PATH}spv_model_{idx}.pt", sep="\n")
            
    
    
