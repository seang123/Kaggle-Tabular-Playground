print("model.py top")
import torch
import torch.nn as nn
import torch.optim as optim
import analyse
import numpy as np
import time
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torchinfo import summary
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
    config.batch_size = 8192 # 4096 #1024 #4096
    config.epochs = 100
    config.learning_rate = 0.0009 # 0.0001
    return config

config = def_config()


## --== Data ==--
#X_train, X_test, y_train, y_test, _, train, test = analyse.load_data(load_cache=True)
X_train, X_test, y_train, y_test, targets, feature_names, train_raw, test_raw  = pipeline.load_data(load_cache=False, save=False)
print("train", train_raw.shape)
print("targets:", targets.shape)
print("X_train", X_train.shape)
print("X_test ", X_test.shape)

def create_loader(X, Y, **kwargs):
    """ Return a torch Dataset and DataLoader for some data """
    dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32).unsqueeze(-1))
    loader = torch.utils.data.DataLoader(dataset, batch_size = config.batch_size, shuffle=kwargs.get('shuffle',True), pin_memory=True)
    return dataset, loader

# Train data
dataset_train, loader_train = create_loader(X_train, y_train)
# Validation data
dataset_val, loader_val = create_loader(X_test, y_test)
# Test Data
dataset_test, loader_test = create_loader(test_raw, np.zeros((test_raw.shape[0], 1)), shuffle=False)
print("Data loaded")


## --== Model ==--
def def_model(in_shape):
    dropout_rate = 0.1
    h_dim = 128 #256
    model = nn.Sequential(
            nn.Linear(in_shape, h_dim),
            nn.ReLU(),
            #nn.LayerNorm(64),
            nn.Dropout(dropout_rate),

            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(h_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(16, 1),
            nn.Sigmoid()
            ).to(device)
    return model
model = def_model(X_train.shape[1])
print(f"model built (on cuda: {next(model.parameters()).is_cuda})")
print(summary(model, input_size=(config.batch_size, X_train.shape[1]),verbose=0))


criterion = nn.BCELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1.0e-10)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(np.ceil(config.epochs-.1*config.epochs))], gamma=0.1, verbose=False) # Reduce lr 10x after 200 epochs

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def accuracy_score(ytrue, ypred):
    return sum([a==b for a, b in zip(ytrue, np.around(ypred))]) / ytrue.shape[0]

def feature_addition(X, Y, criterion):
    """
    Parameters:
    -----------
        X - data
        Y - labels
    """

    n_features = X.shape[1]

    scores = []
    for i in range(1, n_features):
        model = def_model(in_shape=i)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1.0e-10)
        print(f"--== Training on first {i} features ==--")
        train_sample = X[:,0:i]
        X_train, X_test, y_train, y_test = train_test_split(train_sample, Y, test_size=0.1, random_state=4242) # 0.005

        _, train_loader = create_loader(X_train, y_train)
        _, val_loader   = create_loader(X_test, y_test)
        losses = train(model, train_loader, val_loader, optimizer, criterion)
        loss = np.mean(losses['epoch_val_roc_auc'])
        print(f">> final val roc-auc: {loss:.4f} for {i} features\n")
        scores.append(loss)

    print(feature_names)
    print(scores)

    for i in range(len(feature_names)):
        print(f"{feature_names[i]} - {scores[i]:.4f}")


def train(model, loader_train, loader_val, optimizer, criterion):
    train_time = time.time()
    print("Starting training")
    for epoch in range(config.epochs):
        epoch_loss = []
        losses = defaultdict(list)
        start_time = time.time()
        model.train()
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
            # Store loss
            losses['epoch_loss'].append( loss.detach().item() )
            #losses['epoch_acc'].append( accuracy_score(labels.cpu().numpy(), outputs.cpu().detach().numpy()) )

        model.eval()
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
                accuracy_score(labels.cpu().numpy(), outputs.detach().cpu().numpy())
            )


        scheduler.step() # Step the scheduler forward
        end_time = time.time()
        mean_ = np.mean(losses['epoch_val_roc_auc'])
        mean_acc = np.mean(losses['ep_val_acc'])
        #mean_acc_train = np.mean(losses['epoch_acc'])
        cur_lr = f"lr: {get_lr(optimizer)}" if epoch % 10 == 0 else ""
        print(f"epoch: {epoch:02} ({(end_time-start_time):.2f}s) | train loss: {np.mean(losses['epoch_loss']):.4f} | val loss: {np.mean(losses['epoch_loss_val']):.4f} | val roc_auc: {mean_:.4f} | val acc: {mean_acc:.4f} {cur_lr}", flush=True)
    print(f"Finished training ({(time.time() - train_time):.3f})")

    return losses



def generate_predictions(model, loader, name='test'):
    print("Performing inference")

    model.eval()
    predictions = []
    for i, data in enumerate(loader, 0):
        inputs = data[0].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        predictions.append(outputs.cpu().detach().numpy())

    predictions = np.concatenate(predictions, axis=0)
    np.save(open(f"mlp_predictions_{name}.npy", "wb"), predictions)


if __name__ == '__main__':

    #feature_addition(train_raw, targets, criterion)

    # Training
    try:
        train(model, loader_train, loader_val, optimizer, criterion)
    except KeyboardInterrupt:
        print("-- Interrupt training --")
    except Exception as e:
        raise e

    # Train predictions
    #generate_predictions(model, loader_train, 'train')

    # Validation predictions
    #generate_predictions(model, loader_val, 'val')

    # Inference
    generate_predictions(model, loader_test)
    print("Done.")





