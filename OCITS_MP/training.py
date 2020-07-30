import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import numpy as np

from data import get_datasets
import utils


def simple_train(model, batch_size):
    num_epochs = 1

    train_loader, _ = get_datasets(batch_size)
    loss_fn = nn.CrossEntropyLoss().cuda()

    model.train()
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for _ in range(num_epochs):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # run forward pass
            optimizer.zero_grad()
            outputs = model(inputs.to("cuda:1"))

            # run backward pass
            labels = labels.to(outputs.device)
            loss_fn(outputs, labels).backward()
            optimizer.step()

            if batch_idx == 0:
                memory = utils.get_memory_usage()

    return memory
