import torch.nn as nn
import torch

from utils import make_MLP

# neural network for query and key encoder
class Encoder(nn.Module):

    def __init__(self, dim_in: tuple, 
                       dim_out: int,
                       n_layers = 2, 
                       n_kernels = 32, 
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
    def __init__(self, s_shape: tuple, # state shape (already stacked, preprocessed)
                       a_shape: int, # action shape
                       s_dim:   int,   # latent state space dimension
                       a_dim:   int,   # latent action space dimension
                       a_hidden_dims = (50,50,), # action embedding MLP hidden layers
                       a_out_act = nn.Tanh,      # action embedding output activation
                       fdm_hidden_dims = (50,50,), # fwd dynamics MLP hidden layers
                       fdm_out_act = None,         # fwd dynamics MLP output activation
                       ):
        super().__init__()

        self.s_shape = s_shape
        self.a_shape = a_shape
        self.s_dim = s_dim

        self.query_encoder = Encoder(s_shape,s_dim)

        self.key_encoder = Encoder(s_shape,s_dim)
        for param in self.key_encoder.parameters(): 
            param.requires_grad = False # disable gradient computation for target network

        self.action_encoder = make_MLP(a_shape, a_dim, a_hidden_dims, out_act=a_out_act)
        self.fdm = make_MLP(s_dim+a_dim,s_dim,fdm_hidden_dims,out_act=fdm_out_act) 

        self.W = nn.Parameter(torch.rand((s_dim,s_dim))) # for bilinear product
        self.sim_metrics = { # similarity metrics for contrastive loss
            # do the dot product on every pair: (k' k)
            "dot": lambda x,y: torch.einsum("ij, kj -> ik", x,y), 

            # do the bilinear product on every pair: (k' W  k)
            "bilinear": lambda x,y: torch.einsum("ij, kj, jj -> ik", x,y,self.W),

            # temperature ...
        }

    def encode(self, s: torch.Tensor)-> torch.Tensor:
        # just encode a state to be passed to SAC
        with torch.no_grad():
            return self.query_encoder(s)
    
    def compute_loss(self, s, a, r, d, sp):
        # what is proposed in the paper is contrastive loss
        # between new states and predicted new state

        # what is actually implemented in https://github.com/thanhkaist/CCFDM1
        # is a combination of this loss and the CURL loss, which is between
        # current states and does not involve the dynamics model
        raise NotImplementedError()