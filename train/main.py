import numpy as np
import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
from torch.nn import init
import copy

train_data = pd.read_csv('../data/train_data.csv')
train_label = pd.read_csv('../data/train_label.csv')
test_data = pd.read_csv('../data/test_data.csv')
test_label = pd.read_csv('../data/test_label.csv')

train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)

train_label = train_label.reshape((len(train_label), 1))
test_label = test_label.reshape((len(test_label), 1))

# X = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
# Y = torch.utils.data.DataLoader(train_label, batch_size=1, shuffle=False)
# test_x = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
# test_y = torch.utils.data.DataLoader(test_label, batch_size=1, shuffle=False)

X = torch.from_numpy(train_data).float()
Y = torch.from_numpy(train_label).float()
test_x = torch.from_numpy(test_data).float()
test_y = torch.from_numpy(test_label).float()

num_inputs, num_hiddens,num_hiddens2, num_outputs = 13, 16, 10, 1


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.hidden = nn.Linear(num_inputs, num_hiddens)
        self.hidden2 = nn.Linear(num_hiddens, num_hiddens2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(num_hiddens2, num_outputs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


net = Classification()
init.normal_(net.hidden.weight, mean=0, std=0.01)
init.normal_(net.hidden2.weight, mean=0, std=0.01)
init.normal_(net.output.weight, mean=0, std=0.01)
init.constant_(net.hidden.bias, val=0)
init.constant_(net.hidden2.bias, val=0)
init.constant_(net.output.bias, val=0)

#
loss = nn.L1Loss()


def evaluate_accuracy(x, y, net):
    out = net(x)
    # print(out)
    # cnt = 0
    # for i in range(len(out)):
    #     if round(i,0) == y[i]:
    #         cnt = cnt + 1
    # n = y.shape[0]
    return 1


def train(net, train_x, train_y, loss, num_epochs, optimizer=None):
    optimizer = torch.optim.SGD(net.parameters(), lr=1.2, momentum=0.01)
    for epoch in range(num_epochs):
        out = net(train_x)
        l = loss(out, train_y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_loss = l.item()

        if (epoch + 1) % 10 == 0:
            train_acc = evaluate_accuracy(test_x, test_y, net)
            print('epoch %d ,loss %.4f' % (epoch + 1, train_loss))


train(net, X, Y, loss, len(X), None)
