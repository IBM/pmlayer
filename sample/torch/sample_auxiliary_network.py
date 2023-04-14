import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pmlayer.torch.layers import HLattice


def square(x,y):
    return (x*x + y*y) / 2.0

class MLP(nn.Module):
    def __init__(self, input_len, output_len, num_neuron):
        super().__init__()

        self.fc1 = nn.Linear(input_len, num_neuron)
        self.fc2 = nn.Linear(num_neuron, output_len)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

if __name__ == '__main__':
    # prepare data
    a = np.linspace(0.0, 1.0, 10)
    x1, x2 = np.meshgrid(a, a)
    y = square(x1,x2)
    x = np.concatenate([x1.reshape(-1,1),x2.reshape(-1,1)], 1)
    data_x = torch.from_numpy(x.astype(np.float32)).clone()
    data_y = torch.from_numpy(y.reshape(-1,1).astype(np.float32)).clone()

    # set monotonicity
    num_input_dims = 2
    lattice_sizes = torch.tensor([4], dtype=torch.long)
    indices_increasing = [0]

    # auxiliary neural network
    input_len = num_input_dims - len(indices_increasing)
    output_len = torch.prod(lattice_sizes).item()
    ann = MLP(input_len, output_len, 32)

    # train model
    model = HLattice(num_input_dims,lattice_sizes,indices_increasing,ann)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for epoch in range(5000):
        pred_y = model(data_x)
        loss = loss_function(pred_y, data_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('epoch=%d, loss=%f' % (epoch,loss))
