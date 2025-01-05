import numpy as np
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
import matplotlib.pyplot as plt


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding='valid',
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding='valid',
        )
        self.linear1 = nn.Linear(256, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, inputs):
        inputs = self.relu(self.conv1(inputs))
        inputs = self.pool(inputs)
        inputs = self.relu(self.conv2(inputs))
        inputs = self.pool(inputs)
        # inputs = self.relu(self.conv3(inputs))  # num_examples x 120 x 1 x 1 --> num_examples x 120
        inputs = inputs.reshape(inputs.shape[0], -1)
        inputs = self.relu(self.linear1(inputs))
        inputs = self.linear2(inputs)
        inputs = self.linear3(inputs)
        return inputs

batch_size = 32
learning_rate = 0.001
num_epochs = 5


train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

#Initial model
le_net = LeNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(le_net.parameters(), lr=learning_rate)

# Train Network
loss_value = []
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Forward
        scores = le_net(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()
        loss_value.append(loss.item())

plt.plot(np.arange(len(loss_value)), loss_value)
plt.title('Loss function value')
plt.grid(True)
plt.show()