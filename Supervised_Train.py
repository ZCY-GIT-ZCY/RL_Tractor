from model import CNNModel
import torch
from torch import nn, optim
import numpy as np
from sklearn.model_selection import train_test_split as splt

EPOCH = int(1e5)
SEED = 123
VERBOSE = 200
DATA_PATH = "Pre_trained_Data/data."

X,Y = None, None
CNN_Action_model = CNNModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN_Action_model.parameters(), 5e-5)

X = torch.from_numpy(X).type(torch.float)
Y = torch.from_numpy(Y).type(torch.LongTensor)
X_train, X_test, Y_train, Y_test = splt(X, Y, random_state = SEED, test_size = 0.2)

for idx in range(EPOCH):
    CNN_Action_model.train()
    optimizer.zero_grad()
    
    y_logits, _ = CNN_Action_model(X_train)
    current_loss = loss_fn(y_logits, Y_train)
    current_loss.backwrads()
    optimizer.step()
    
    CNN_Action_model.eval()
    with torch.inference_mode():
        y_test, _ = CNN_Action_model(X_test)
        test_loss = loss_fn(y_test, Y_test)
        
    if idx % VERBOSE == 0:
        print(f"Epoch {idx} Training Progress as follows:",
        f"Training Loss: {current_loss}.",f"Test Loss: {test_loss}.", sep="\n")
    
    
