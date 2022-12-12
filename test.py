from agent.encoder import Encoder

# test 
net = Encoder((3,64,64),dim_out=5,n_layers=2,n_kernels=32)
print(net.v_shape)
