import torch.nn as nn
import torch

from utils import make_MLP

# neural network for query and key encoder
class Encoder(nn.Module):

    def __init__(self, dim_in: tuple, 
                       dim_out: int,
                       n_layers: int, 
                       n_kernels: int, 
                       ksize=3,
                       mlp_hidden_dims = (), 
                       hidden_act=nn.ReLU, 
                       out_act=None):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        layers = []
        layers.append(nn.Conv2d(dim_in[0],n_kernels,ksize))
        layers.append(hidden_act())

        for l in range(n_layers -1):
            layers.append(nn.Conv2d(n_kernels,n_kernels,ksize))
            layers.append(hidden_act())

        self.cnn = nn.Sequential(*layers)
        
        # get flattened size through a forward
        with torch.no_grad():
            _test = torch.zeros(self.dim_in)
            out = self.cnn(_test)
            self.v_shape = out.flatten().shape[0]

        self.mlp = make_MLP(self.v_shape, self.dim_out, mlp_hidden_dims, out_act)

    def forward(self, x):
        v = torch.flatten(self.cnn(x))
        return self.mlp(v)

# the whole encoder
class FeatureEncoder(nn.Module):
    def __init__(self, obs_shape: tuple, 
                       a_shape:   tuple,
                       ):
        super().__init__()

