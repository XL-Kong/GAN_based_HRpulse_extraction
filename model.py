import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(968, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 1)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.activation(x)
        # return nn.Sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self,):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 968)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return nn.Tanh()(x)