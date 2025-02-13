import numpy as np
import torch
from .C2ST_utils import BinaryClassifier, LabeledDataset
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def wrapped_gaussian_log_prob(theta, mu, sigma, K=10):
    theta = theta % (2 * np.pi)
    batch_size = theta.shape[0]
    ks = torch.arange(-K, K + 1, device=theta.device).unsqueeze(0)
    delta = theta + 2 * np.pi * ks 
    exponents = -0.5 * ((delta - mu) ** 2) / (sigma ** 2)
    log_sum_exp = torch.logsumexp(exponents, dim=1)
    log_const = -0.5 * np.log(2 * np.pi) - torch.log(sigma.squeeze())
    log_probs = log_const + log_sum_exp 
    return log_probs

def iterate_over_manifolds(func, kwargs, in_axes=-2, out_axes=-2, mul=2):
    #efficiently iterate over product manifold
    method = func
    in_axes = []
    args_list = []
    for key, value in kwargs.items():
        if hasattr(value, "reshape"):
            value = value.reshape((*value.shape[:-1], mul, -1))
            in_axes.append(-2)
        else:
            in_axes.append(None)
        args_list.append(value)
    out = torch.vmap(method, in_dims=tuple(in_axes), out_dims=out_axes)(*args_list)
    out = out.reshape((*out.shape[:out_axes], -1))
    return out

def loglikelihood(sample, mu=None, sigma=None, mul=2):
    #computes the loglikelihood for wrapped Gaussian distribution on the Torus
    if mu is None:
        mu = torch.zeros(mul)
    if sigma is None:
        sigma = (0.2 * torch.ones(mul)).sqrt()
    llh = iterate_over_manifolds(wrapped_gaussian_log_prob, \
                {"theta": sample, "mu":  mu.broadcast_to(sample.shape), "sigma":sigma.broadcast_to(sample.shape)}, mul=mul)
    return llh.reshape(-1, llh.shape[0]).mean()


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in tqdm(test_loader):
            outputs = model(data).squeeze()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    accuracy = accuracy * 100
    return accuracy

def C2ST(target, sample, manifold):
    data_positive = target
    data_negative = sample
    batch_size = data_positive.shape[0]
    sampled_indices = torch.randperm(batch_size, device=data_positive.device)[:data_negative.shape[0]]
    data_positive = data_positive[sampled_indices]
    labels_positive = torch.ones(len(data_positive), dtype=sample.dtype, device=sample.device) 
    labels_negative = torch.zeros(len(data_negative), dtype=sample.dtype, device=sample.device)
    data_tensor = torch.cat([data_positive, data_negative], dim=0)
    labels_tensor = torch.cat([labels_positive, labels_negative], dim=0)
    full_dataset = LabeledDataset(data_tensor, labels_tensor)
    train_size = int(0.8 * len(full_dataset)) 
    test_size = len(full_dataset) - train_size 
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    input_size = sample.shape[-1]
    model = BinaryClassifier(input_size, manifold)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 150
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    score = evaluate_model(model, test_loader)
    score = 0.5 + abs(0.5 - score)
    return score

