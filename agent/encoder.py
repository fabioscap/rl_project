import torch.nn as nn
import torch
from math import exp
from utils import make_MLP, infoNCE, soft_update_params, copy_params
from torchvision import transforms
# neural network for query and key encoder
class Encoder(nn.Module):

    def __init__(self, dim_in: tuple, 
                       dim_out: int,
                       n_layers = 2, 
                       n_kernels = 32, 
                       ksize=3,
                       mlp_hidden_dims = (), 
                       hidden_act=nn.ReLU, 
                       out_act=None,
                       device="cpu"
                       ):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        layers = []
        layers.append(nn.Conv2d(dim_in[0],n_kernels,ksize))
        layers.append(hidden_act())

        for l in range(n_layers -1):
            layers.append(nn.Conv2d(n_kernels,n_kernels,ksize))
            layers.append(hidden_act())

        self.cnn = nn.Sequential(*layers).to(device)
        
        # get flattened size through a forward
        with torch.no_grad():
            _test = torch.zeros(self.dim_in).to(device)
            out = self.cnn(_test)
            self.v_shape = out.flatten().shape[0]

        self.mlp = make_MLP(self.v_shape, self.dim_out, mlp_hidden_dims, out_act).to(device)

    def forward(self, x): # x has shape (b,c,w,h)
        v = torch.flatten(self.cnn(x), start_dim = 1)
        return self.mlp(v)

# the whole encoder
class FeatureEncoder(nn.Module):
    def __init__(self, s_shape: tuple, # state shape (already stacked, preprocessed)
                       s_dim:   int,   # latent state space dimension
                       tau = 0.005, # target update speed
                       C = 0.2,      # intrinsic weight
                       gamma = 2e-5, # intrinsic decay
                       device="cpu",
                       ):
        super().__init__()

        self.device = device

        self.s_shape = s_shape

        self.s_dim = s_dim

        self.tau = tau
        self.C = C
        self.gamma = gamma

        self.query_encoder = Encoder(s_shape,s_dim, device=device)

        self.key_encoder = Encoder(s_shape,s_dim, device=device)
        for param in self.key_encoder.parameters(): 
            param.requires_grad = False
        copy_params(self.query_encoder, self.key_encoder)

        self.W = nn.Parameter(torch.rand((s_dim,s_dim))).to(device) # for bilinear product
        self.sim_metrics = { # similarity metrics for contrastive loss
            # do the dot product on every pair: (k' k)
            "dot": lambda x,y: torch.einsum("ij, kj -> ik", x,y), 

            # do the bilinear product on every pair: (k' W  k)
            "bilinear": lambda x,y: torch.einsum("ij, kj, jj -> ik", x,y,self.W),

            # temperature ...
        }


    def encode(self, s: torch.Tensor, 
                     target=False, 
                     grad=True,    
                     )-> torch.Tensor:
        # just encode a state to be passed to SAC
        
    
        if target:

            return self.key_encoder(s)
        else:
            if grad:
                return self.query_encoder(s)
            else:
                with torch.no_grad():
                    return self.query_encoder(s)
    

    def compute_contrastive_loss(self, q, k, sim_metric="dot"):
        fdm_contrastive_loss = infoNCE(q,k,self.sim_metrics[sim_metric], self.device)
    
        return fdm_contrastive_loss


    def update_key_network(self):
        soft_update_params(self.query_encoder, self.key_encoder, self.tau)
    
    def encode_reward_loss(self, obs, pos, sim_metric="dot"):

        q = self.encode(obs, grad=True) # encode state with key encoder with grad
                                      # in order to update the network through SAC loss
        
        k = self.encode(pos, target=True, grad=False) # encode keys with target
                                                      # compute no gradient

        l = self.compute_contrastive_loss(q, k, sim_metric)


        return q, l