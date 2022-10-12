import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pmlayer.torch import mlp
from pmlayer.common import util


class MultiLinearInterpolation:
    '''
    Multilinear interpolation

    self.coef : Tensor (dtype=torch.long)
        self.coef.shape = [ # monotone columns ]
    self.mesh_size : Tensor (dtype=torch.long)
        self.coef.mesh_size = [ # monotone columns ]
    '''

    def __init__(self, mesh_size):
        '''
        Compute self.coef based on mesh_size

        Parameters
        ----------
        mesh_size : list of integer
            mesh_size specifies the size of each dimension
        '''

        self.mesh_size = torch.tensor(mesh_size, dtype=torch.long)
        coef = []
        for i in range(len(mesh_size)):
            coef.append(torch.prod(self.mesh_size[i+1:]))
        self.coef = torch.tensor(coef, dtype=torch.long)

    def get_index(self, coordinates):
        '''
        Parameters
        ----------
        coordinates : Tensor
            coordinates.shape = [batchsize, len(self.mesh_size)]
            Each number in coordinates must be in range [0,1)

        Returns
        -------
        coordinates_int : Tensor
            coordinates_int.shape = coordinates.shape
            Integer coordinates scaled by self.mesh_size

        coordinates_frac : Tensor
            coordinates_frac.shape = coordinates.shape
            Fractional coordinates scaled by self.mesh_size
        '''

        coordinates = coordinates * (self.mesh_size-1)
        coordinates_int = coordinates.to(torch.long)
        coordinates_int = torch.clamp(coordinates_int, min=0)
        coordinates_frac = coordinates - coordinates_int
        return coordinates_int, coordinates_frac

    def interpolate(self, coordinates, mesh_pred):
        '''
        Interpolate mesh_pred values at coordinates

        Parameters
        ----------
        coordinates : Tensor
            coordinates.shape = [ batch_size, # monotone columns ]

        mesh_pred : Tensor
            mesh_pred.shape = [ batch_size, volume of lattice ]

        Returns
        -------
        ret : long
        '''

        coordinates_int, coordinates_frac = self.get_index(coordinates)
        l = [0,1]
        pred = torch.zeros(coordinates.shape[0])
        for d in itertools.product(l, repeat=coordinates.shape[1]):
            d = torch.tensor(d, dtype=torch.long)
            index = self.coordinate2index(coordinates_int + d).view(-1,1)
            value = torch.gather(mesh_pred, 1, index).view(-1)
            temp = (1-d) - coordinates_frac * (1-d*2)
            pred += value * temp.prod(axis=1)
        return pred

    def coordinate2index(self, coordinate):
        '''
        Convert coordinate into index

        Parameters
        ----------
        coordinate : Tensor (dtype=torch.long)
            coordinate.shape = [ batch_size, # monotone column ]

        Returns
        -------
        ret : Tensor (dtype=torch.long)
            ret.shape = [ batch_size ]
        '''

        if coordinate.shape[0] == 0:
            return None

        return torch.matmul(coordinate, self.coef)


class LUMap:
    '''
    Vertex and its upper and lower sets.
    Coordinates of vertices are represented as tuples.

    self.coordindate : coordinate of a vertex
    self.lower_set : set of vertices that are dominated by this vertex
    self.upper_set : set of vertices that dominate this vertex
    '''

    def __init__(self, coordinate):
        self.coordinate = coordinate
        self.lower_set = []
        self.upper_set = []

    def __str__(self):
        ret = str(self.coordinate)
        ret += ' lower:'+str(self.lower_set)
        ret += ' upper:'+str(self.upper_set)
        return ret

class HLattice(nn.Module):
    '''
    Hierarchical lattice layer
    
    @note Current implementation does not support decreasing function
    '''

    def __init__(self,
                 lattice_sizes,
                 monotonicities,
                 neural_network=None):
        super().__init__()
        '''
        Parameters
        ----------
        lattice_sizes : list of integer
            Specifies the granularity of lattice for each input feature
            Each number must be at least 2
            Numbers corresponding to non-monotone features are ignored

        monotonicities : list of integer or str
            Specifies the monotonicity of each input feature
        '''

        # create map table
        indices = util.parse_monotonicities(monotonicities,
                                            len(lattice_sizes))
        if len(indices[1]) > 0:  # if len(idx_dec) > 0
            message = 'HLattice does not support decreasing function'
            raise ValueError(message, monotonicities)
        input_len = len(indices[2])  # len(idx_none)
        mesh_size = []
        output_len = 1  # @note can be computed by math.prod
        cols_monotone = [ False for i in range(len(monotonicities)) ]
        for idx in indices[0]:  # for idx in idx_inc:
            cols_monotone[idx] = True
            size = lattice_sizes[idx]
            output_len *= size
            mesh_size.append(size)
        self.mli = MultiLinearInterpolation(mesh_size)
        self.map_table = self._create_map_table()

        # initialize neural network
        if input_len > 0:
            if neural_network is None:
                self.nn = mlp.MLP(input_len, output_len)
            else:
                self.nn = neural_network
        else:
            var = torch.sqrt(torch.full((output_len,), 2.0 / output_len))
            initial_b = torch.normal(0.0, var)
            self.b = nn.Parameter(initial_b)

        # set monotonicity
        self.cols_monotone = torch.tensor(cols_monotone)
        self.cols_non_monotone = torch.logical_not(self.cols_monotone)

    def _create_map_table(self):
        '''
        Create ret (list of LUMap) based on self.mli.mesh_size
        '''

        # create LUMap
        max_hamming_distance = torch.sum(self.mli.mesh_size).item()
        max_hamming_distance -= self.mli.mesh_size.shape[0]
        ret = []
        bt = util.create_skewed_tree(max_hamming_distance+2)
        tree_preorder = util.traverse_preorder(bt)
        mesh_size = self.mli.mesh_size.tolist()
        for tree_node in tree_preorder:
            ret.extend(self._create_map_table_sub([],
                                                  mesh_size,
                                                  tree_node[1]-1,
                                                  tree_node))

        # convert coordinates in LUMap into long values
        for node in ret:
            coordinate = torch.tensor(node.coordinate, dtype=torch.long)
            node.index = self.mli.coordinate2index(coordinate)
            ls = torch.tensor(list(node.lower_set), dtype=torch.long)
            node.lower_index = self.mli.coordinate2index(ls)
            us = torch.tensor(list(node.upper_set), dtype=torch.long)
            node.upper_index = self.mli.coordinate2index(us)
        return ret

    def _create_map_table_sub(self, coordinate, cols_max, residual,
                               tree_node):
        '''
        Parameters
        ----------
        coordinate : list of integer
            Coordinates determined so far

        cols_max : list of integer
            Maximum of each dimension of coordinates

        residual : integer
            Residual that can be used to fill coordinates

        tree_node : tuple of three integers
            tree_node = (left, value, right)
        '''

        cols_index = len(coordinate)
        if cols_index >= len(cols_max):
            # do nothing if coordinate is invalid
            if residual > 0:
                return []

            # create lumap based on coordinate
            coordinate = tuple(coordinate)
            lumap = LUMap(coordinate)
            diff_u = tree_node[2] - tree_node[1]
            lumap.upper_set = self._enumerate_upper_set([],
                                                        coordinate,
                                                        cols_max,
                                                        diff_u)
            diff_l = tree_node[1] - tree_node[0]
            lumap.lower_set = self._enumerate_lower_set([],
                                                        coordinate,
                                                        cols_max,
                                                        diff_l)
            return [lumap]

        ret = []
        for i in range(min(cols_max[cols_index],residual+1)):
            coordinate.append(i)
            mt = self._create_map_table_sub(coordinate, cols_max,
                                            residual-i, tree_node)
            ret.extend(mt)
            coordinate.pop()
        return ret

    def _enumerate_upper_set(self, coordinate, cols_base, cols_max,
                             residual):
        cols_index = len(coordinate)
        if cols_index >= len(cols_max):
            if residual == 0:
                return [tuple(np.array(cols_base) + np.array(coordinate))]
            else:
                return []

        ret = []
        num = min(cols_max[cols_index]-cols_base[cols_index],residual+1)
        for i in range(num):
            coordinate.append(i)
            ret.extend(self._enumerate_upper_set(coordinate, cols_base,
                                                 cols_max, residual-i))
            coordinate.pop()
        return ret

    def _enumerate_lower_set(self, coordinate, cols_base, cols_max,
                             residual):
        cols_index = len(coordinate)
        if cols_index >= len(cols_max):
            if residual == 0:
                return [tuple(np.array(cols_base) - np.array(coordinate))]
            else:
                return []

        ret = []
        for i in range(min(cols_base[cols_index]+1,residual+1)):
            coordinate.append(i)
            ret.extend(self._enumerate_lower_set(coordinate, cols_base,
                                                 cols_max, residual-i))
            coordinate.pop()
        return ret

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            x.shape = [batch_size, len(lattice_sizes)]

        Returns
        -------
        ret : Tensor
            ret.shape = [batch_size, 1]
        '''

        # predict values associated with lattice vertices
        xn = x[:,self.cols_non_monotone]
        if xn.shape[1] > 0:
            xn = self.nn(xn)
        else:
            b = torch.sigmoid(self.b)
            xn = torch.tile(b, (xn.shape[0],1))
        if xn.shape[1] == 1:  # all inputs are non-monotone
            return xn

        # transform tree structure into estimated grid values
        out = torch.Tensor(xn.shape)
        for item in self.map_table:
            if item.lower_index is None:
                lb = torch.zeros(xn.shape[0])
            else:
                lb = torch.index_select(out, 1, item.lower_index)
                lb, _ = torch.max(lb, 1)
                lb = lb.view(-1)
            if item.upper_index is None:
                ub = torch.ones(xn.shape[0])
            else:
                ub = torch.index_select(out, 1, item.upper_index)
                ub, _ = torch.min(ub, 1)
                ub = ub.view(-1)
            out[:,item.index] = torch.lerp(lb, ub, xn[:,item.index])

        # interpolate by using the output of the neural network
        monotone_inputs = x[:,self.cols_monotone]
        ret = self.mli.interpolate(monotone_inputs, out)
        return ret.view(-1,1)
