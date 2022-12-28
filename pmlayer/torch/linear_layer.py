import math
import torch
import torch.nn as nn

from pmlayer.common import util


class PMLinear(nn.Module):
    '''
    Linear layer with monotonicity constraints.
    Monotonicity can be specified to ensure that the weight of
    corresponding feature is positive or negative.
    '''

    def __init__(self,
                 num_input_dims,
                 num_output_dims=1,
                 indices_increasing=[],
                 indices_decreasing=[],
                 use_bias=True):
        super().__init__()
        '''
        Initialize this layer.
        Parameters
        ----------
        num_input_dims : int
            The number of input features
        num_output_dims : int
            The number of output features
        indices_increasing : list of indices
            The list of indices of monotonically increasing features.
        indices_decreasing : list of indices
            The list of indices of monotonically decreasing features.
        use_bias : bool
            Indicate if bias is used
        '''

        if num_input_dims <= 0:
            message = 'num_input_dims must be a positive integer'
            raise ValueError(message, num_input_dims)
        if num_output_dims <= 0:
            message = 'num_output_dims must be a positive integer'
            raise ValueError(message, num_output_dims)

        k_sqrt = math.sqrt(1.0 / num_input_dims)
        size = (num_input_dims, num_output_dims)
        w = 2.0 * k_sqrt * torch.rand(size) - k_sqrt
        self.weight = nn.Parameter(w)
        if use_bias:
            b = 2.0 * k_sqrt * torch.rand(1, num_output_dims) - k_sqrt
            self.bias = nn.Parameter(b)
        else:
            self.bias = torch.zeros(1, num_output_dims)
        self.idx_inc = indices_increasing
        self.idx_dec = indices_decreasing
        self.idx_none = []
        for i in range(num_input_dims):
            if i in indices_increasing:
                continue
            if i in indices_decreasing:
                continue
            self.idx_none.append(i)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            x.shape = [*, num_input_dims]
        Returns
        -------
        output : Tensor
            output.shape = [*, num_output_dims]
        '''

        wx = torch.matmul(x[:,self.idx_none], self.weight[self.idx_none,:])
        w_inc = torch.exp(self.weight[self.idx_inc,:])
        wx += torch.matmul(x[:,self.idx_inc], w_inc)
        w_dec = torch.exp(self.weight[self.idx_dec,:])
        wx -= torch.matmul(x[:,self.idx_dec], w_dec)
        return wx + self.bias
