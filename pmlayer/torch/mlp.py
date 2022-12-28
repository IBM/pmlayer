import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''
    Multi layer perceptron (MLP)
    Sigmoid function is used for the last layer.
    '''

    def __init__(self, input_len, output_len, num_neuron=128):
        '''
        Parameters
        ----------
        input_len : int
            Length of input
        output_len : Tensor
            Length of output
        num_neuron : Tensor
            The number of neurons in intemediate layers
        '''

        super().__init__()

        self.fc1 = nn.Linear(input_len, num_neuron)
        self.fc2 = nn.Linear(num_neuron, num_neuron)
        self.fc3 = nn.Linear(num_neuron, num_neuron)
        self.fc4 = nn.Linear(num_neuron, output_len)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            x.shape = [batch_size, input_len]
        Returns
        -------
        ret : Tensor
            ret.shape = [batch_size, output_len]
        '''

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x
