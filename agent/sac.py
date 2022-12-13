import torch.nn as nn
from utils import make_MLP, copy_params

# soft actor critic for vector states
class SAC(nn.Module):
    def __init__(self,
                s_dim,
                a_dim,
                value_hidden_dims: tuple,
                Q_hidden_dims: tuple,
                policy_hidden_dims: tuple,

                ):
        super().__init__()

        self.V_network = make_MLP(s_dim,1,value_hidden_dims)
        self.Q_network = make_MLP(s_dim,a_dim,Q_hidden_dims)
        self.Q_target = make_MLP(s_dim,a_dim,Q_hidden_dims)
        for param in self.Q_target.parameters(): 
            param.requires_grad = False # disable gradient computation for target network
        # copy params at start ?
        copy_params(self.Q_network, self.Q_target)

        self.policy_network = make_MLP(s_dim, 2*a_dim, policy_hidden_dims) 
                                              # half for the mean
                                              # and half for the (log) std
        ########## AOOO#######