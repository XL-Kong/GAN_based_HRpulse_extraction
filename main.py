from model import Discriminator, Generator
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np


class MyDataset(Dataset):
    def __init__(self, transform=None):
        cam_path = 'C:/Users/12452/Desktop/msc project/project1/cam_data/'
        oxi_path = 'C:/Users/12452/Desktop/msc project/project1/oxi_data/'
        self.transform = transform
        self.cam_data = []
        self.oxi_data = []
        for f in os.listdir(cam_path):
            self.cam_data.append(np.load(cam_path+f))
        for j in os.listdir(oxi_path):
            self.oxi_data.append(np.load(oxi_path+j))
        # print(len(self.cam_data[0]))

    def __getitem__(self, index):
        return self.cam_data[index], self.oxi_data[index]

    def __len__(self):
        return len(self.cam_data)


epoch_num = 100
batch_size = 1

input_size = 968
z_dimension = 128

d_learning_rate = 0.005
g_learning_rate = 0.005


G = Generator()
D = Discriminator()
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)
criterion = nn.KLDivLoss(reduction='batchmean')

dataset = MyDataset()
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epoch_num):
    for i, (data, label) in enumerate(dataloader):
        num_data = len(data)
        real_data = torch.as_tensor(data).float().cuda()
        real_label = torch.as_tensor(label).float().cuda()
        real_out = D(real_data)
        d_loss = criterion(real_out, real_label)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        z = torch.randn(num_data, z_dimension).cuda()
        fake_data = G(z).detach()
        fake_out = D(fake_data)
        g_loss = criterion(fake_out, real_label)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 5 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '.format(
                epoch, epoch_num, d_loss.data.item(), g_loss.data.item(),
            ))


p = "runs/"
train_num = sum([os.path.isdir(os.path.join(p, listx)) for listx in os.listdir(p)])
os.mkdir('runs/'+str(train_num+1))
torch.save(G.state_dict(), 'runs/'+str(train_num+1)+'/generator.pth')
torch.save(D.state_dict(), 'runs/'+str(train_num+1)+'/discriminator.pth')




