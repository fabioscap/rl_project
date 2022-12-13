import torch.nn as nn
import torch

from utils import make_MLP, infoNCE, soft_update_params, copy_params

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
                       tau = 0.005, # target update speed
                       C = 0.2,      # intrinsic weight
                       gamma = 2e-5, # intrinsic decay
                       ):
        super().__init__()

        self.s_shape = s_shape
        self.a_shape = a_shape
        self.s_dim = s_dim

        self.tau = tau
        self.C = C
        self.gamma = gamma

        self.query_encoder = Encoder(s_shape,s_dim)

        self.key_encoder = Encoder(s_shape,s_dim)
        for param in self.key_encoder.parameters(): 
            param.requires_grad = False # disable gradient computation for target network
        # copy params at start ?
        copy_params(self.query_encoder, self.key_encoder)

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
        self.max_intrinsic = 0 # the maximum intrinsic reward (for normalization)

    def encode(self, s: torch.Tensor)-> torch.Tensor:
        # just encode a state to be passed to SAC
        with torch.no_grad():
            return self.query_encoder(s)
    
    def predict(self,s,a)-> torch.Tensor:
        # generate a prediciton for the new state
        q = self.query_encoder(s) # encode the state
        ae = self.action_encoder(a) # encode the action
        qp = self.fdm(torch.cat((q,ae),dim=1)) # predict new state

        return qp

    def compute_contrastive_loss(self, s, a, sp, sim_metric="dot"):
        # what is proposed in the paper is contrastive loss
        # between new states and predicted new state

        # what is actually implemented in https://github.com/thanhkaist/CCFDM1
        # is a combination of this loss and the CURL loss, which is between
        # current states and does not involve the dynamics model.
        # this modification is not reported in the paper.

        # fdm loss
        qp = self.predict(s,a)

        # encode the next states in the transition
        kp = self.key_encoder(sp)

        fdm_contrastive_loss = infoNCE(qp,kp,self.sim_metrics[sim_metric])

        # TODO: curl loss ?
        # TODO: what happens when done is true?

        return fdm_contrastive_loss


    def compute_intrinsic_reward(self, s, a, sp, step, max_reward, sim_metric="dot"):
        with torch.no_grad(): # don't backprop through intrinsic reward
            qp = self.predict(s,a)
            kp = self.key_encoder(sp)

            # we try to use MSE as a dissimilarity metric
            pred_error = (qp-kp).pow(2).sum(dim=1).sqrt()

            max_error = max(pred_error)
            if max_error > self.max_intrinsic:
                self.max_intrinsic = max_error
            
            ri = self.C*torch.exp(-self.gamma*step)* pred_error * max_reward / self.max_intrinsic

            # TODO: what happens when done is true?

            return ri

    def update_key_network(self):
        soft_update_params(self.query_encoder, self.key_encoder, self.tau)
        