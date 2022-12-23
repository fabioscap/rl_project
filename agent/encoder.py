import torch.nn as nn
import torch
from math import exp
from utils import make_MLP, infoNCE, soft_update_params, copy_params, random_crop, center_crop_images
from torchvision import transforms
# neural network for query and key encoder
class Encoder(nn.Module):

    def __init__(self, dim_in: tuple, 
                       dim_out: int,
                       n_layers = 4, 
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
        layers.append(nn.Conv2d(dim_in[0],16,ksize))
        layers.append(hidden_act())
        layers.append(nn.Conv2d(16,n_kernels,ksize))
        layers.append(hidden_act())

        for l in range(n_layers -1):
            # layers.append(nn.MaxPool2d(kernel_size=(2,2)))
            layers.append(nn.Conv2d(n_kernels,n_kernels,ksize))
            layers.append(hidden_act())

        self.cnn = nn.Sequential(*layers).to(device)
        
        # get flattened size through a forward
        with torch.no_grad():
            _test = torch.zeros(self.dim_in).to(device)
            out = self.cnn(_test)
            self.v_shape = out.flatten().shape[0]

        self.mlp = make_MLP(self.v_shape, self.dim_out, mlp_hidden_dims, out_act).to(device)
        self.layer_norm = nn.LayerNorm(self.dim_out).to(device)

    def forward(self, x): # x has shape (b,c,w,h)
        v = torch.flatten(self.cnn(x), start_dim = 1)
        return self.layer_norm(self.mlp(v))

# the whole encoder
class FeatureEncoder(nn.Module):
    def __init__(self, s_crop_shape: tuple,
                       a_shape: int, # action shape
                       s_dim:   int,   # latent state space dimension
                       a_dim:   int,   # latent action space dimension
                       a_hidden_dims = (50,50,), # action embedding MLP hidden layers
                       a_out_act = None,      # action embedding output activation
                       fdm_hidden_dims = (50,50,), # fwd dynamics MLP hidden layers
                       fdm_out_act = None,         # fwd dynamics MLP output activation
                       tau = 0.002, # target update speed
                       C = 0.2,      # intrinsic weight
                       gamma = 2e-5, # intrinsic decay
                       device="cpu",
                       ):
        super().__init__()

        self.device = device

        self.s_crop_shape = s_crop_shape
        self.a_shape = a_shape
        self.s_dim = s_dim

        self.tau = tau
        self.C = C
        self.gamma = gamma

        self.query_encoder = Encoder(s_crop_shape,s_dim, device=device)
        self.key_encoder = Encoder(s_crop_shape,s_dim, device=device)
        for param in self.key_encoder.parameters(): 
            param.requires_grad = False # disable gradient computation for target network
        # copy params at start ?
        copy_params(self.query_encoder, self.key_encoder)

        self.action_encoder = make_MLP(a_shape[0], a_dim, a_hidden_dims, out_act=a_out_act).to(device)
        self.fdm = make_MLP(s_dim+a_dim,s_dim,fdm_hidden_dims,out_act=fdm_out_act).to(device)

        self.W = nn.Parameter(torch.rand((s_dim,s_dim))).to(device) # for bilinear product
        self.sim_metrics = { # similarity metrics for contrastive loss
            # do the dot product on every pair: (k' k)
            "dot": lambda x,y: torch.einsum("ij, kj -> ik", x,y), 

            # do the bilinear product on every pair: (k' W  k)
            "bilinear": lambda x,y: torch.einsum("ij, kj, jj -> ik", x,y,self.W),

            # temperature ...
        }
        self.max_intrinsic = 1e-8 # the maximum intrinsic reward (for normalization)

    def encode(self, s: torch.Tensor, 
                     target=False, # whether to use key network
                     grad=True,    # enable/disable gradient flow
                     center_crop=False)-> torch.Tensor:
        
        if center_crop:
            cropped = center_crop_images(s, self.s_crop_shape[-1]) # s is a single state
        else:
            cropped = random_crop(s, self.s_crop_shape[-1]) # s is a batch of states
        cropped = cropped.to(self.device)
        if target:
            return self.key_encoder(cropped)
        else:
            if grad:
                return self.query_encoder(cropped)
            else:
                with torch.no_grad():
                    return self.query_encoder(cropped)

    def compute_contrastive_loss(self, qp, kp, sim_metric="dot"):
        # what is proposed in the paper is contrastive loss
        # between new states and predicted new state

        # what is actually implemented in https://github.com/thanhkaist/CCFDM1
        # is a combination of this loss and the CURL loss, which is between
        # current states and does not involve the dynamics model.
        # this modification is not reported in the paper.

        return infoNCE(qp,kp,self.sim_metrics[sim_metric], self.device)
      

    def compute_mse_loss(self, qp, kp):
        
        mse_loss = nn.functional.mse_loss(qp,kp)
        
        return mse_loss

    def compute_intrinsic_reward(self, qp, kp, step, max_reward):
        with torch.no_grad(): #  make sure you don't backprop through intrinsic reward

            # we try to use MSE as a dissimilarity metric
            pred_error = (qp-kp).pow(2).sum(dim=1).sqrt()

            max_error = max(pred_error)
            if max_error > self.max_intrinsic:
                self.max_intrinsic = max_error

            ri = self.C*exp(-self.gamma*step)* pred_error * (max_reward / self.max_intrinsic)

            return ri.reshape(-1,1)

    def update_key_network(self):
        soft_update_params(self.query_encoder, self.key_encoder, self.tau)
    
    def encode_reward_loss(self, s, a, sp, step, max_reward, sim_metric="bilinear"):
        q = self.encode(s, target=False, grad=True, center_crop=False) # encode state with key encoder with grad
                                      # in order to update the network through SAC loss
        k_anch = self.encode(s.clone(), target=True, center_crop=False) # for CURL
        
        curl_loss = self.compute_contrastive_loss(q,k_anch,sim_metric)

        #ae = self.action_encoder(a)  
        #qp = self.fdm(torch.cat((q,ae),dim=1)) # for FDM loss

        #kp = self.encode(sp, target=True, grad=False, center_crop=False) # encode keys with target
                                                      # compute no gradient

        #fdm_loss = self.compute_contrastive_loss(qp, kp, sim_metric)

        #ri = self.compute_intrinsic_reward(qp.detach(), kp.detach(), step, max_reward)
        ri = 0
        weight = 0.2

        return q, ri, curl_loss #+ (1-weight)*fdm_loss