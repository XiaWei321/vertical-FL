import torch
from torch import nn
import numpy as np
import pandas as pd

test_data = pd.read_csv('../data/test_data.csv')
test_label = pd.read_csv('../data/test_label.csv')
test_data = np.array(test_data)
test_label = np.array(test_label)
test_label = test_label.reshape((len(test_label), 1))

test_x = torch.from_numpy(test_data).float()
test_y = torch.from_numpy(test_label).float()

num_inputs, num_hiddens, num_outputs = 13, 16, 1


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.hidden = nn.Linear(num_inputs, num_hiddens)
        self.relu = nn.ReLU()
        self.output = nn.Linear(num_hiddens, num_outputs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


model = torch.load('./model.pkl')


def evaluate_accuracy(x, y, net):
    out = net(x)
    res = 0
    for i in range(len(y)):
        if y[i] == 0.95:
            if abs(out[i] - y[i]) < 0.05:
                res = res + 1
        else:
            if abs(out[i] - y[i]) < 0.1125 / 2:
                res = res + 1
    print('out:', out)
    print('y:', y)
    print('res:', res / len(y))


evaluate_accuracy(test_x, test_y, model)
