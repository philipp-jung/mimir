# 0 for training SeqGAN, 1 for repairing part, 2 for doing it simultaneously
flag = 2
# Number of model execution (in alternating orders)
runs = 1
# Order, 1 for positive order, 0 for negative order
order = 1
# Insert errors into the data (1 yes, 0 no)
insert_errors = 0

# batch size,Default=32
batch_size = 32
# Max length of sentence,Default=25
max_length = 20

# Generator embedding size
g_e = 64
# Generator LSTM hidden size
g_h = 64

# Discriminator embedding and Highway network sizes
d_e = 64
# Discriminator LSTM hidden size
d_h = 64

# Number of Monte Calro Search
n_sample=16
# Number of generated sentences,Default=500,20000
generate_samples = 500

# Pretraining parameters,g_pre_epochs_Default=20,d=3
g_pre_epochs= 5
d_pre_epochs = 2

g_lr = 1e-5

# Discriminator dropout ratio
d_dropout = 0.0
d_lr = 1e-6

# Pretraining parameters
g_pre_lr = 1e-2
d_pre_lr = 1e-4


# filter sizes for CNNs
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
# num of filters for CNNs
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
