import torch
import torch.nn as nn
import torch.optim as optim

def train(net, num_epochs, optimizer, criterion, trainloader, device='cuda'):
    net.train()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
