import math
import torch
import torch.nn as nn

from pmlayer.common import util


class PiecewiseLinear(nn.Module):
    '''
    This layer transforms input x = [x_1, x_2, ..., x_d ] into
    f(x) = [ f_1(x_1), f_2(x_2), ..., f_d(x_d) ], where d is equal
    to num_dims and each f_i is a piece-wise linear function.
    The parameter boundaries specifies the x-axis of the endpoints of f_i.
    The y-axis values of the endpoints of f_i are trainable parameters
    of this layer.
    
    The features that are not specified in indices_increasing,
    indices_decreasing, and indices_transform are unchanged.
    '''

    def __init__(self,
                 input_keypoints,
                 num_dims=1,
                 output_min=0.0,
                 output_max=1.0,
                 indices_increasing=[],
                 indices_decreasing=[]):
        '''
        Initialize this layer.

        Parameters
        ----------
        input_keypoints : Tensor
            The boundaries of input features.
            input_keypoints.shape = [ number of endpoints ]

        num_dims : int
            The number of input features.

        output_min : float
            The minimum of output.

        output_max : float
            The maximum of output.

        indices_increasing : list of indices
            The list of indices of monotonically increasing features.

        indices_decreasing : list of indices
            The list of indices of monotonically decreasing features.
        '''

        super().__init__()

        self.output_max = output_max
        self.output_min = output_min
        self.keypoints_x = input_keypoints
        self.num_dims = num_dims
        self.idx_inc = indices_increasing
        self.idx_dec = indices_decreasing
        self.idx_tra = []
        for i in range(num_dims):
            if i in self.idx_inc:
                continue
            if i in self.idx_dec:
                continue
            self.idx_tra.append(i)
        l = len(input_keypoints)
        size_inc = (len(self.idx_inc), l-1)
        size_dec = (len(self.idx_dec), l-1)
        size_tra = (len(self.idx_tra), l)
        var_inc = torch.full(size_inc, math.sqrt(1.0 / l))
        var_dec = torch.full(size_dec, math.sqrt(1.0 / l))
        var_tra = torch.full(size_tra, math.sqrt(1.0 / l))
        initrand_inc = torch.normal(torch.zeros(size_inc), var_inc)
        initrand_dec = torch.normal(torch.zeros(size_dec), var_dec)
        initrand_tra = torch.normal(torch.zeros(size_tra), var_tra)
        self.weight_inc = nn.Parameter(initrand_inc)
        self.weight_dec = nn.Parameter(initrand_dec)
        self.weight_tra = nn.Parameter(initrand_tra)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def _interpolate(self, keypoints_y, x):
        index = torch.searchsorted(self.keypoints_x, x)
        index = torch.clamp(index, min=1, max=len(self.keypoints_x))
        len_fragment = self.keypoints_x[index] - self.keypoints_x[index-1]
        x_frac = (x - self.keypoints_x[index-1]) / len_fragment
        x_frac = torch.clamp(x_frac, min=0.0, max=1.0)
        y_left = torch.gather(keypoints_y, 0, index-1)
        y_right = torch.gather(keypoints_y, 0, index)
        return torch.lerp(y_left, y_right, x_frac)

    def forward(self, x):
        '''
        Transform x into f(x).

        Parameters
        ----------
        x : Tensor
            x.shape = [*, num_dims]

        Returns
        -------
        ret : Tensor
            ret.shape = [*, num_dims]
        '''

        ret = torch.ones_like(x)
        diff = self.output_max - self.output_min

        # increasing features
        if len(self.idx_inc) > 0:
            zeros = torch.zeros(len(self.idx_inc), 1,
                                    device=self.weight_inc.device)
            weights = torch.cumsum(self.softmax(self.weight_inc), dim=1)
            weights = torch.cat((zeros, weights), 1).T
            keypoints_y = diff * weights + self.output_min
            ret[:,self.idx_inc] *= self._interpolate(keypoints_y,
                                                     x[:,self.idx_inc])

        # decreasing features
        if len(self.idx_dec) > 0:
            zeros = torch.zeros(len(self.idx_dec), 1,
                                    device=self.weight_inc.device)
            weights = torch.cumsum(self.softmax(self.weight_dec), dim=1)
            weights = torch.cat((zeros, weights), 1).T
            keypoints_y = -diff * weights + self.output_max
            ret[:,self.idx_dec] *= self._interpolate(keypoints_y,
                                                     x[:,self.idx_dec])

        # transform features
        if len(self.idx_tra) > 0:
            keypoints_y = self.sigmoid(self.weight_tra).T
            ret[:,self.idx_tra] *= self._interpolate(keypoints_y,
                                                     x[:,self.idx_tra])

        return ret
