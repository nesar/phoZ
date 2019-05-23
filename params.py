######################## PARAMETERS ##########################

original_dim = 6#   # DES - 8, COSMOs - 6
#intermediate_dim3 = 1600
# intermediate_dim2 = 1024
intermediate_dim1 = 32
intermediate_dim0 = 16
intermediate_dim = 8
latent_dim = 4


num_train = 784
num_test = 128
num_para = 1

batch_size = 16
num_epochs =  10 # 20  #200 # 7500 # 200  #110 #50
epsilon_mean = 0.0 # 1.0
epsilon_std = 1.0 #1e-1 ## original = 1.0, smaller the better 1e-4
learning_rate =  1e-4
decay_rate = 1.0

noise_factor = 0.0 # 0.0 necessary

######################## I/O #################################
MainDir = './data/'

DataDir = MainDir+'Data/'
PlotsDir = MainDir+'Plots/'
ModelDir = MainDir+'Model/'

fileOut = 'P'+str(num_para)+'Model_tot' + str(num_train) + '_batch' + str(batch_size) + '_lr' + str(
    learning_rate) + '_decay' + str(decay_rate) + '_z' + str(latent_dim) + '_epoch' + str(
    num_epochs)

##############################################################
