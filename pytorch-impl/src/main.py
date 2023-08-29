"""
This implementation reflects how neural network example, where we explicitly use a softmax followed by a cross entropy.
This implementation follows the same code to make better comparison.
For general better train-time performance use CrossEntropyLoss without Softmax layer.
"""

from __future__ import print_function

import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import pandas as pd


class CustomSoftmaxCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(CustomSoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):
        # inputs are assumed to be already softmax-ed
        # targets contain class indices

        batch_size = inputs.size(0)
        # Gather the probabilities corresponding to the true labels
        true_probs = inputs[range(batch_size), targets]

        # Compute the negative log likelihood
        loss = -torch.sum(torch.log(true_probs))

        # Normalize by batch size
        loss = loss / batch_size
        return loss

# Follow scala neural network architecture
class Net(nn.Module):
    def __init__(self, n_hidden_units=36):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, n_hidden_units)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden_units, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = CustomSoftmaxCrossEntropyLoss().forward(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_accuracy))

    return test_accuracy

def main():
    # Training settings
    BATCH_SIZE = 64
    DEVICE = torch.device("cpu")

    epochs = 2
    lr = 0.01

    train_kwargs = {'batch_size': BATCH_SIZE}
    test_kwargs = {'batch_size': BATCH_SIZE}

    # Follow scala preprocessing
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.reshape(28*28)),
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    assert len(train_dataset.data) == 60000

    model = Net().to(DEVICE)
    # Use same weight initialization we do in scala
    torch.nn.init.uniform_(model.fc1.bias, -0.5, 0.5)
    torch.nn.init.uniform_(model.fc1.weight, -0.5, 0.5)
    torch.nn.init.uniform_(model.fc2.bias, -0.5, 0.5)
    torch.nn.init.uniform_(model.fc2.weight, -0.5, 0.5)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    results = defaultdict(list)

    for experiment_id in range(1, 10):
        for epoch in range(1, epochs + 1):
            print(epoch)
            # Measure train time for epoch
            start_time = time.perf_counter()
            train(model, DEVICE, train_loader, optimizer)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            test_acc = test(model, DEVICE, test_loader)
            results['experiment_id'].append(experiment_id)
            results['epoch'].append(epoch)
            results['elapsed_time'].append(elapsed_time)
            results['test_acc'].append(test_acc)

    out_path = "out/results"
    out_file = os.path.join(out_path, "results.csv")
    os.mkdir(out_path)
    pd.DataFrame.from_dict(results).to_csv(out_file)


if __name__ == '__main__':
    main()