import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pmlayer.torch.layers import HLattice


def square(x,y):
    return (x*x + y*y) / 2.0

if __name__ == '__main__':
    # prepare data
    a = np.linspace(0.0, 1.0, 10)
    x1, x2 = np.meshgrid(a, a)
    y = square(x1,x2)
    x = np.concatenate([x1.reshape(-1,1),x2.reshape(-1,1)], 1)
    data_x = torch.from_numpy(x.astype(np.float32)).clone()
    data_y = torch.from_numpy(y.reshape(-1,1).astype(np.float32)).clone()

    # train model
    lattice_sizes = torch.tensor([4,4], dtype=torch.long)
    model = HLattice(2,lattice_sizes,[0,1])
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for epoch in range(5000):
        pred_y = model(data_x)
        loss = loss_function(pred_y, data_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('epoch=%d, loss=%f' % (epoch,loss))

    # plot
    pred_y_np = pred_y.to('cpu').detach().numpy().copy().reshape(x1.shape)
    plt.figure(figsize=(4,3))
    ax = plt.subplot(1, 1, 1)
    im = ax.contourf(x1, x2, pred_y_np, levels=[0.0,0.2,0.4,0.6,0.8,1.0])
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7, top=0.9)
    cax = plt.axes([0.8, 0.1, 0.05, 0.8])
    plt.colorbar(im,cax=cax)
    plt.show()
