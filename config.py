# hyperparameters
embedding_dim = 32
n_layers = 2
clip = 10
learning_rate = 1e-2
eps = 1e-10
drop_prob = 0.1
ffn_hidden = 16
n_head = 8
# When comm_rate is 0, the best results are achieved in terms of performance, 
# whereas when comm_rate is 1, the best efficiency is obtained.
comm_rate=0
center_flag=True