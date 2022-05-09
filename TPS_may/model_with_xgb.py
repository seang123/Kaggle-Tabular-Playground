print("model.py top")
import torch
import torch.nn as nn
import torch.optim as optim
import analyse
import numpy as np
import time
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import pipeline
import tqdm

# Check for CUDA availability
print("cuda is available:", torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

# Parameter
def def_config():
    config = Config()
    config.batch_size = 4096 #2048 #1024 #4096
    config.epochs = 300
    config.learning_rate = 0.0001
    return config

config = def_config()


## --== Data ==--
#X_train, X_test, y_train, y_test, _, train, test = analyse.load_data(load_cache=True)
X_train, X_test, y_train, y_test, _, _, test  = pipeline.load_data(load_cache=True, save=False)
print("X_train", X_train.shape)
print("X_test ", X_test.shape)
# Train data
dataset_train = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1))
loader_train  = torch.utils.data.DataLoader(dataset_train, batch_size = config.batch_size, shuffle=True, pin_memory=True)
# Validation data
dataset_val = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1))
loader_val  = torch.utils.data.DataLoader(dataset_val, batch_size = config.batch_size, shuffle=False, pin_memory=True)
# Test Data
dataset_test = torch.utils.data.TensorDataset(torch.tensor(test, dtype=torch.float32), torch.tensor(np.zeros((test.shape[0], 1)), dtype=torch.float32).unsqueeze(-1))
loader_test  = torch.utils.data.DataLoader(dataset_test, batch_size = config.batch_size, shuffle=False, pin_memory=True)
print("Data loaded")


## --== Model ==--
# N -> 64 -> 64 -> 16 -> 1
model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.LeakyReLU(0.1),
        nn.LayerNorm(64),

        nn.Linear(64, 64),
        nn.LeakyReLU(0.1),
        nn.LayerNorm(64)
        ).to(device)




"""
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)
        self.act = nn.LeakyReLU(0.1)
        self.sig = nn.Sigmoid()
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(16)
        self.emb = nn.Embedding(20, 16)
        #self.lstm = nn.GRU(64, 64, 10, batch_first=True) # (input_dim, hidden_state_dim, seq_len)

    def forward2(self, features, fstring):
        x = self.ln1(self.act(self.fc1(features)))

        emb = self.emb(fstring)
        out, h_n = self.lstm(emb)
        x += h_n[-1] # (bs, h_dim)

        x = self.ln1(self.act(self.fc2(x)))
        x = self.ln1(self.act(self.fc3(x)))
        x = self.ln2(self.act(self.fc4(x)))
        x = self.sig(self.fc5(x))
        return x

    def forward(self, features, fstring):
        x = self.ln1(self.act(self.fc1(features)))
        emb = torch.sum(self.emb(fstring), axis=1)
        x = self.ln1(self.act(self.fc2(x)))
        x = self.ln1(self.act(self.fc3(x)))
        x = self.ln2(self.act(self.fc4(x) + emb))
        x = self.sig(self.fc5(x))
        return x
model = Model().to(device)
"""
print(f"model built (on cuda: {next(model.parameters()).is_cuda})")


criterion = nn.BCELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1.0e-10)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1, verbose=False) # Reduce lr 10x after 200 epochs

def accuracy_score(ytrue, ypred):
    score = 0
    for i in range(len(ytrue)):
        if ytrue[i] == 0 and ypred[i] <= 0.5:
            score += 1
        elif ytrue[i] == 1 and ypred[i] > 0.5:
            score += 1
        else:
            continue
    return score / len(ytrue)

from xgboost import XGBClassifier
params = {'n_estimators': 200, #900, # 4096, # nr. boosting rounds
        'max_depth': 6, #12, # 8,
        'max_leaves': 0,
        'learning_rate': 0.15,
        'subsample': 0.95,
        'colsample_bytree': 0.95,
        'reg_alpha': 1.5, # L1
        'reg_lambda': 1.5, # L2
        'gamma': 1.5,
        'booster': 'gbtree', #'dart', #'gbtree', # gbtree - default
        'random_state': 46,
        #'scale_pos_weight':0, # 1 -defualt for when high class imbalance
        'objective': 'binary:logistic',
        'base_score': 0.49, # initial prediction score of all instances (global bias)
        'tree_method': 'hist', # 'approx', # 'gpu_hist' for gpu training
        #'early_stopping_rounds':300,
        'eval_metric':['auc'],
        }
xgb_model = XGBClassifier(**params)


def train(model, loader_train, loader_val, optimizer, criterion):
    train_time = time.time()
    print("Starting training")
    for epoch in range(config.epochs):
        epoch_loss = []
        losses = defaultdict(list)
        start_time = time.time()
        for i, data in enumerate(loader_train, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass model
            outputs = model(inputs)

            # XGBModel
            with torch.no_grad():
                f = outputs.cpu().detach().numpy()
                xgb_model.fit(f, labels.cpu().numpy())
                pred = torch.Tensor(xgb_model.predict_proba(f)[:,1][:,None])

            # Loss
            loss = criterion(pred, labels)
            print(loss)
            # Backprop
            #outputs.backward(gradient=loss)
            loss.backward()
            optimizer.step()
            # Store loss
            losses['epoch_loss'].append( loss.detach().item() )

        for i, data in enumerate(loader_val, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            losses['epoch_loss_val'].append( loss.detach().item() )
            losses['epoch_val_roc_auc'].append(
                roc_auc_score(labels.cpu().numpy(), outputs.cpu().detach().numpy())
            )
            losses['ep_val_acc'].append(
                accuracy_score(labels.cpu().numpy(), outputs.cpu().detach().numpy())
            )


        scheduler.step() # Step the scheduler forward
        end_time = time.time()
        mean_ = np.mean(losses['epoch_val_roc_auc'])
        mean_acc = np.mean(losses['ep_val_acc'])
        print(f"epoch: {epoch} ({(end_time-start_time):.2f}s) | train loss: {np.mean(losses['epoch_loss']):.4f} | val loss: {np.mean(losses['epoch_loss_val']):.4f} | val roc_auc: {mean_:.4f} | val acc: {mean_acc:.4f}")
    print(f"Finished training ({(time.time() - train_time):.3f})")



def generate_predictions(model, loader, name='test'):
    print("Performing inference")

    predictions = []
    for i, data in enumerate(loader, 0):
        inputs = data[0].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        predictions.append(outputs.cpu().detach().numpy())

    predictions = np.concatenate(predictions, axis=0)
    np.save(open(f"mlp_predictions_{name}.npy", "wb"), predictions)


if __name__ == '__main__':

    # Training
    try:
        train(model, loader_train, loader_val, optimizer, criterion)
    except KeyboardInterrupt:
        print("-- Interrupt training --")
    except Exception as e:
        raise e

    # Validation predictions
    generate_predictions(model, loader_val, 'val')

    # Inference
    generate_predictions(model, loader_test)
    print("Done.")





