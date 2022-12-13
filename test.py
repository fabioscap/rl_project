from agent.encoder import FeatureEncoder
from utils import infoNCE
import torch
# test 
net = FeatureEncoder((4,64,64),2,4,12)

dot = net.sim_metrics["dot"]
bilinear = net.sim_metrics["bilinear"]

a = torch.rand((3,4))
b = torch.rand((3,4))

a = torch.arange(3*4, dtype=torch.float32).reshape(3,4)
b = torch.arange(3*4, dtype=torch.float32).reshape(3,4)


print(infoNCE(a,b,dot))
