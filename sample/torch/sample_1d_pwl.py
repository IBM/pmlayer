import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('../../')
from pmlayer.torch.layers import PiecewiseLinear


if __name__ == '__main__':
    # prepare data
    x = np.linspace(0.0, 1.0, 10)
    y = 2.0*(x-0.3)*(x-0.3)
    x = np.tile(x.reshape(-1,1), 2)
    y = np.tile(y.reshape(-1,1), 2)
    data_x = torch.from_numpy(x.astype(np.float32)).clone()
    data_y = torch.from_numpy(y.astype(np.float32)).clone()

    # train model
    boundaries = torch.linspace(0.0,1.0,5)
    model = PiecewiseLinear(boundaries,num_dims=2,indices_increasing=[0])
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(10000):
        pred_y = model(data_x)
        loss = loss_function(pred_y, data_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('epoch=%d, loss=%f' % (epoch,loss))

    # plot
    pred_y_np = pred_y.to('cpu').detach().numpy().copy()
    fig = plt.figure(figsize=(7,3))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.set_title('Increasing')
    ax1.plot(x[:,0], y[:,0], color='gray', linestyle = 'dotted')
    ax1.plot(x[:,0], pred_y_np[:,0], marker='o')
    ax2.set_title('Unconstrained')
    ax2.plot(x[:,1], y[:,1], color='gray', linestyle = 'dotted')
    ax2.plot(x[:,1], pred_y_np[:,1], marker='o')
    plt.show()
