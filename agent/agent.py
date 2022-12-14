from _sac import SacAgent
from encoder import FeatureEncoder
class Agent():
    def __init__(self,
                 obs_shape: tuple,
                 a_shape: tuple,
                 n_frames: int, # how many frames to stack
                 s_dim: int, # state representation dimension
                 a_dim: int, # action representation dimension
                 device = "cpu"
                ):
        self.obs_shape = obs_shape
        self.n_frames = n_frames
        
        self.s_shape = (self.obs_shape[0]*n_frames, *self.obs_shape[1:])
        self.feature_encoder = FeatureEncoder(self.s_shape, a_shape, s_dim, a_dim)
        self.sac = SacAgent(s_dim, a_dim, device=device)

        # optimizers ...

        # move the SAC update methods here, we want to backprop also through the encoder

    def sample_action(self, state):
        raise NotImplementedError()

