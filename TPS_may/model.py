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
    config.batch_size = 4096
    config.epochs = 200
    config.learning_rate = 0.0001
    return config

config = def_config()


## --== Data ==--
#X_train, X_test, y_train, y_test, _, test = analyse.load_data()
X_train, X_test, y_train, y_test, _, test  = pipeline.load_data()
print("X_train", X_train.shape)
print("X_test ", X_test.shape)
# Train data
dataset_train = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1))
loader_train  = torch.utils.data.DataLoader(dataset_train, batch_size = config.batch_size, shuffle=True, pin_memory=True)
# Validation data
dataset_val = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1))
loader_val  = torch.utils.data.DataLoader(dataset_val, batch_size = config.batch_size, shuffle=False, pin_memory=True)
# Test Data
dataset_test = torch.utils.data.TensorDataset(torch.tensor(test, dtype=torch.float32), torch.tensor(np.zeros((test.shape[0], 1)), dtype=torch.float32).unsqueeze(-1))
loader_test  = torch.utils.data.DataLoader(dataset_test, batch_size = config.batch_size, shuffle=False, pin_memory=True)
print("Data loaded")


## --== Model ==--
# N -> N -> 64 -> 16 -> 1
model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        #nn.ReLU(),
        nn.Hardswish(),
        nn.LayerNorm(64),
        nn.Linear(64, 64),
        #nn.ReLU(),
        nn.Hardswish(),
        nn.LayerNorm(64),
        nn.Linear(64, 16),
        #nn.ReLU(),
        nn.Hardswish(),
        nn.LayerNorm(16),
        nn.Linear(16, 1),
        nn.Sigmoid()
        ).to(device)
print(f"model built (on cuda: {next(model.parameters()).is_cuda})")


#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)#, weight_decay=5.0e-4)


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
            # Loss
            loss = criterion(outputs, labels)
            # Backprop
            loss.backward()
            optimizer.step()

            losses['epoch_loss'].append( loss.detach().item() )
            #losses['epoch_loss'].append( loss.detach() )

        for i, data in enumerate(loader_val, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            losses['epoch_loss_val'].append( loss.detach().item() )
            #losses['epoch_loss_val'].append( loss.detach() )

            roc_auc = roc_auc_score(labels.cpu().numpy(), outputs.cpu().detach().numpy())
            losses['epoch_val_roc_auc'].append(roc_auc)


        end_time = time.time()
        mean_ = np.mean(losses['epoch_val_roc_auc'])
        #min_  = np.min(losses['epoch_val_roc_auc'])
        #max_  = np.max(losses['epoch_val_roc_auc'])
        print(f"epoch: {epoch} ({(end_time-start_time):.2f}s) | train loss: {np.mean(losses['epoch_loss']):.3f} | val loss: {np.mean(losses['epoch_loss_val']):.3f} | val roc_auc: {mean_:.3f}")
    print(f"Finished training ({(time.time() - train_time):.3f})")


def generate_predictions(model, loader):
    print("Performing inference")

    predictions = []
    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        predictions.append(outputs.cpu().detach().numpy())

    predictions = np.concatenate(predictions, axis=0)
    np.save(open("mlp_predictions.npy", "wb"), predictions)


if __name__ == '__main__':

    # Training
    train(model, loader_train, loader_val, optimizer, criterion)

    # Inference
    generate_predictions(model, loader_test)
    print("Done.")





