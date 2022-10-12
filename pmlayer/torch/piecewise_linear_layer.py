import math
import torch
import torch.nn as nn

from pmlayer.common import util


class PiecewiseLinear(nn.Module):
    '''
    Piecewise linear layer.

    If monotonicity is specified, then the corresponding feature
    is transformed by using a piecewise linear function.
    Otherwise, the corresponding feature is not changed.

    @Note current implementation does not support a decreasing function.
    '''

    def __init__(self,
                 input_keypoints,
                 num_dims=1,
                 output_min=0.0,
                 output_max=1.0,
                 monotonicities=None):
        '''
        Initialize this layer.

        Parameters
        ----------
        input_keypoints : Tensor
            Increasing sequence of boundaries

        num_dims : int
            The number of input features

        output_min : float
            The minimum of output

        output_max : float
            The maximum of output

        monotonicies : str or int or list of length num_dims
            Specifies monotonicities of input features
        '''

        super().__init__()

        self.keypoints_x = input_keypoints
        self.num_dims = num_dims
        indices = util.parse_monotonicities(monotonicities, num_dims)
        self.idx_inc = indices[0]
        self.idx_dec = indices[1]
        self.idx_none = indices[2]
        l = len(input_keypoints)-1
        size = (num_dims, l)
        var = torch.full(size, math.sqrt(1.0 / l))
        self.weight = nn.Parameter(torch.normal(torch.zeros(size), var))
        self.softmax = nn.Softmax(dim=1)
        self.a = output_max - output_min
        self.b = output_min

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            x.shape = [batch_size, num_dims]

        Returns
        -------
        ret : Tensor
            ret.shape = [batch_size, num_dims]
        '''

        ret = x

        weights_cumsum = torch.cumsum(self.softmax(self.weight), dim=1)
        zero = torch.zeros(self.num_dims, 1)
        weights_cumsum = torch.cat((zero, weights_cumsum), 1).T
        keypoints_y = self.a * weights_cumsum + self.b

        index = torch.searchsorted(self.keypoints_x, x[self.idx_inc])
        index = torch.clamp(index, min=1, max=len(self.keypoints_x))
        len_fragment = self.keypoints_x[index] - self.keypoints_x[index-1]
        x_frac = (x[self.idx_inc] - self.keypoints_x[index-1]) / len_fragment
        x_frac = torch.clamp(x_frac, min=0.0, max=1.0)
        y_left = torch.gather(keypoints_y, 0, index-1)
        y_right = torch.gather(keypoints_y, 0, index)
        ret[self.idx_inc] = torch.lerp(y_left, y_right, x_frac)

        return ret
