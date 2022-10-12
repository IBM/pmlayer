import math
import torch
import torch.nn as nn

from pmlayer.common import util


class PMLinear(nn.Module):
    '''
    Partially monotone linear layer.

    Monotonicity can be specified to ensure that the weight of
    corresponding feature is positive or negative.
    '''

    def __init__(self,
                 num_input_dims,
                 num_output_dims=1,
                 monotonicities=None,
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

        monotonicies : str or int or list of length num_input_dims
            Specifies monotonicities of input features

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
        indices = util.parse_monotonicities(monotonicities, num_input_dims)
        self.idx_inc = indices[0]
        self.idx_dec = indices[1]
        self.idx_none = indices[2]

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            x.shape = [batch_size, num_input_dims]

        Returns
        -------
        output : Tensor
            output.shape = [batch_size, num_output_dims]
        '''

        wx = torch.matmul(x[:,self.idx_none], self.weight[self.idx_none,:])
        w_inc = torch.exp(self.weight[self.idx_inc,:])
        wx += torch.matmul(x[:,self.idx_inc], w_inc)
        w_dec = torch.exp(self.weight[self.idx_dec,:])
        wx -= torch.matmul(x[:,self.idx_dec], w_dec)
        return wx + self.bias
