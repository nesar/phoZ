# import matplotlib as mpl
# mpl.use('Agg')

import math
import numpy as np
import string
from datetime import datetime
import os
from astropy.table import Table
import matplotlib.pyplot as plt;
import random


import matplotlib.axes as axes;
from matplotlib.patches import Ellipse
import seaborn as sns;
from scipy import stats
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub

tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

np.random.seed(12)


print(30*'=~')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
# writer = tf.summary.FileWriter('./log_dir', sess.graph)
# tf.contrib.summary.create_file_writer('./log_dir', sess.graph)
print(30*'=~')

def ReadCosmosDraw_UM(path_program = '../../Data/fromGalaxev/photozs/datasets/'):

    fileIn = path_program + 'Training_data_UM_random/all_finite_col_mag_sdss.npy'
    #fileInColors = path_program + 'new_cosmos_sdss/all_col_sdss.npy'

    TrainfilesColors = np.load(fileIn)
    #TrainfilesMagI = np.load(fileInMagI)
    print('Train files shape', TrainfilesColors.shape)

    min_col = -5
    max_col = 5
    max_max = 25
    for ii in range(TrainfilesColors.shape[1]):
        aa = np.alltrue(np.isfinite(TrainfilesColors[:, ii, :]), axis=1)
        bb = (TrainfilesColors[:,ii,-1] < max_max) & (aa == True)
        cc = np.alltrue(TrainfilesColors[:, ii, :-1] < max_col, axis=1) & (bb == True)
        mask = np.alltrue(TrainfilesColors[:, ii, :-1] > min_col, axis=1)  & (cc == True)

    TrainfilesColors = TrainfilesColors[mask]
    print(TrainfilesColors.shape)


    #magI_low = 15
    #magI_high = 23

    fileInZ = path_program + 'Training_data_UM_random/redshifts.npy'
    TrainZ = np.load(fileInZ)

    # print(TrainfilesCol.shape, TrainZ.shape)
    
    # Trainfiles = np.append(TrainfilesCol, TrainZ[:, None], axis=1) 

    Trainfiles = np.zeros(shape=(TrainfilesColors.shape[0]*TrainfilesColors.shape[1], TrainfilesColors.shape[2] + 1))

    for galID in range(TrainfilesColors.shape[0]):

    #     TrainfilesMagI[galID, :, 1][TrainfilesMagI[galID, :, 1] < magI_low] = magI_low
    #     TrainfilesMagI[galID, :, 0][TrainfilesMagI[galID, :, 0] > magI_high] = magI_high

    #     imag = np.random.uniform(low=TrainfilesMagI[galID, :, 0], high=TrainfilesMagI[galID, :, 1], size=(num_magI_draws, np.shape(TrainfilesMagI[galID, :, 1])[0])).T

        # for mag_degen in range(num_magI_draws):
            # colors_mag = np.append(TrainfilesColors[galID, :, :], imag[:, mag_degen][:, None], axis=1)
            trainfiles100 = np.append(TrainfilesColors[galID, :, :] , TrainZ[:, None], axis=1)

            train_ind_start = galID*TrainfilesColors.shape[1]
            train_ind_end = galID*TrainfilesColors.shape[1] + TrainfilesColors.shape[1]

            # print(train_ind_start, train_ind_end)

            Trainfiles[train_ind_start: train_ind_end] = trainfiles100

    print('Train files shape (with z)', Trainfiles.shape)


    TrainshuffleOrder = np.arange(Trainfiles.shape[0])
    np.random.shuffle(TrainshuffleOrder)
    Trainfiles = Trainfiles[TrainshuffleOrder]

    Test_VAL = False
    if Test_VAL: 

        fileIn = path_program + 'new_cosmos_sdss/SDSS_val.npy'
        Testfiles = np.load(fileIn)
        print('Test files shape:', Testfiles.shape)


        # min_col = -5
        # max_col = 5
        # max_max = 25
        # for ii in range(Testfiles.shape[1]):
        #     aa = np.alltrue(np.isfinite(Testfiles[:, ii, :]), axis=1)
        #     bb = (Testfiles[:,ii,-1] < max_max) & (aa == True)
        #     cc = np.alltrue(Testfiles[:, ii, :-1] < max_col, axis=1) & (bb == True)
        #     mask = np.alltrue(Testfiles[:, ii, :-1] > min_col, axis=1)  & (cc == True)


        TestshuffleOrder = np.arange(Testfiles.shape[0])
        np.random.shuffle(TestshuffleOrder)

        Testfiles = Testfiles[TestshuffleOrder]
        X_train = Trainfiles[:num_train, :-1]  # color mag
        X_test = Testfiles[:num_test, 1:]  # color mag
        
        y_train = Trainfiles[:num_train, -1]  # spec z
        y_test = Testfiles[:num_test, 0] # spec z

    # ############## THINGS ARE SAME AFTER THIS ###########
    #
    # ## rescaling xmax/xmin
    # xmax = np.max([np.max(X_train, axis=0), np.max(X_test, axis=0)], axis=0)
    # xmin = np.min([np.min(X_train, axis=0), np.min(X_test, axis=0)], axis=0)
    #
    # X_train = (X_train - xmin) / (xmax - xmin)
    # X_test = (X_test - xmin) / (xmax - xmin)
    #
    # #### RESCALING X_train, X_test NOT done yet -- (g-i), (r-i) ... and i mag -->> Color/Mag issue
    #
    # ymax = np.max([y_train.max(), y_test.max()])
    # ymin = np.min([y_train.min(), y_test.min()])
    #
    # y_train = (y_train - ymin) / (ymax - ymin)
    # y_test = (y_test - ymin) / (ymax - ymin)
    #
    # return X_train, y_train, X_test, y_test, ymax, ymin, xmax, xmin
    #
    # ############# THINGS ARE SAME AFTER THIS ###########

    TestSynth = False

    if TestSynth:

        X_train = Trainfiles[:num_train, :-1]  # color mag
        X_test = Trainfiles[num_train + 1: num_train + num_test, :-1]  # color mag


        y_train = Trainfiles[:num_train, -1]   # spec z
        y_test = Trainfiles[num_train + 1: num_train + num_test, -1]  # spec z


    ##################################################
    ##################################################

    TestSDSS = True

    if TestSDSS:

        #     fileIn = path_program + 'new_cosmos_sdss/SDSS_val.npy'
        fileIn = path_program + 'Data_from_observations_new/SDSS_cols.npy'
        TestfilesColors = np.load(fileIn)
        fileIn = path_program + 'Data_from_observations_new/SDSS_iband.npy'
        TestfilesMag = np.load(fileIn)   
        
        Testfiles = np.append(TestfilesColors, TestfilesMag[:, None], axis=1)


        # TrainshuffleOrder = np.arange(Trainfiles.shape[0])
        # np.random.shuffle(TrainshuffleOrder)

        # Trainfiles = Trainfiles[TrainshuffleOrder]

        TestshuffleOrder = np.arange(Testfiles.shape[0])
        np.random.shuffle(TestshuffleOrder)

        Testfiles = Testfiles[TestshuffleOrder]

        X_train = Trainfiles[:num_train, :-1]  # color mag
        X_test = Testfiles[:num_test, 1:]  # color mag
        y_train = Trainfiles[:num_train, -1]  # spec z
        y_test = Testfiles[:num_test, 0] # spec z


    ############################################################
    ############## THINGS ARE SAME AFTER THIS ###########



    ## rescaling xmax/xmin
    xmax = np.max([np.max(X_train, axis=0), np.max(X_test, axis=0)], axis=0)
    xmin = np.min([np.min(X_train, axis=0), np.min(X_test, axis=0)], axis=0)

    X_train = (X_train - xmin) / (xmax - xmin)
    X_test = (X_test - xmin) / (xmax - xmin)

    #### RESCALING X_train, X_test NOT done yet -- (g-i), (r-i) ... and i mag -->> Color/Mag issue

    ymax = np.max([y_train.max(), y_test.max()])
    ymin = np.min([y_train.min(), y_test.min()])

    y_train = (y_train - ymin) / (ymax - ymin)
    y_test = (y_test - ymin) / (ymax - ymin)

    return X_train, y_train, X_test, y_test, ymax, ymin, xmax, xmin

def evaluate(tensors):
    """Evaluates Tensor or EagerTensor to Numpy `ndarray`s.
    Args:
    tensors: Object of `Tensor` or EagerTensor`s; can be `list`, `tuple`,
      `namedtuple` or combinations thereof.

    Returns:
      ndarrays: Object with same structure as `tensors` except with `Tensor` or
        `EagerTensor`s replaced by Numpy `ndarray`s.
    """
    if tf.executing_eagerly():
        return tf.contrib.framework.nest.pack_sequence_as(
            tensors,
            [t.numpy() if tf.contrib.framework.is_tensor(t) else t
             for t in tf.contrib.framework.nest.flatten(tensors)])
    return sess.run(tensors)

def plot_normal_mix(pis, mus, sigmas, ax, label='', comp=True):
  """Plots the mixture of Normal models to axis=ax comp=True plots all
  components of mixture model
  """
  # x = np.linspace(-10.5, 10.5, 250)
  x = np.linspace(-0.1, 1.1, 250)
  final = np.zeros_like(x)
  for i, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
    temp = stats.norm.pdf(x, mu_mix, sigma_mix) * weight_mix
    final = final + temp
    if comp:
      ax.plot(x, temp, label='Normal ' + str(i))
  ax.plot(x, final, label='Mixture of Normals ' + label)
  ax.legend(fontsize=13)
  return final

def neural_network_mod():
    """
    loc, scale, logits = NN(x; theta)

    Args:
      X: Input Tensor containing input data for the MDN
    Returns:
      locs: The means of the normal distributions that our data is divided into.
      scales: The scales of the normal distributions that our data is divided
        into.
      logits: The probabilities of ou categorical distribution that decides
        which normal distribution our data points most probably belong to.
    """
    X = tf.placeholder(tf.float64,name='X',shape=(None,D))
    # 2 hidden layers with 15 hidden units
    net = tf.layers.dense(X, 64, activation=tf.nn.relu)
    net = tf.layers.dense(X, 32, activation=tf.nn.relu)
    net = tf.layers.dense(net, 16, activation=tf.nn.relu)
    net = tf.layers.dense(net, 8, activation=tf.nn.relu)
    locs = tf.layers.dense(net, K, activation=None)
    scales = tf.layers.dense(net, K, activation=tf.exp)
    logits = tf.layers.dense(net, K, activation=None)
    outdict= {'locs':locs, 'scales':scales, 'logits':logits}
    hub.add_signature(inputs=X,outputs=outdict)

    return locs, scales, logits

def mixture_model(X,Y,learning_rate=1e-3,decay_rate=.95,step=1000,train=True):
    if train:
        dict = neural_network(tf.convert_to_tensor(X),as_dict=True)
    else:
        dict = neural_network_t(tf.convert_to_tensor(X),as_dict=True)
    locs = dict['locs'] ; scales = dict['scales'] ; logits = dict['logits']
    cat = tfd.Categorical(logits=logits)
    components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                  in zip(tf.unstack(tf.transpose(locs)),
                         tf.unstack(tf.transpose(scales)))]

    y = tfd.Mixture(cat=cat, components=components)
    #define loss function
    log_likelihood = y.log_prob(Y)
    # log_likelihood = -tf.reduce_sum(log_likelihood/(1. + y_train)**2 )
    y_mean = np.median(Y)
    log_likelihood = -tf.reduce_sum(log_likelihood)
    #log_likelihood = -tf.reduce_sum(log_likelihood*(y_mean-y_train)**4 )
    if train:
        global_step = tf.Variable(0, trainable=False)
        decayed_lr = tf.train.exponential_decay(learning_rate,
                                        global_step, step,
                                        decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(decayed_lr)
        train_op = optimizer.minimize(log_likelihood)
        evaluate(tf.global_variables_initializer())
        return log_likelihood, train_op, logits, locs, scales
    else:
        evaluate(tf.global_variables_initializer())
        return log_likelihood, logits, locs, scales

def train(log_likelihood,train_op,n_epoch):
    train_loss = np.zeros(n_epoch)
    test_loss = np.zeros(n_epoch)
    for i in range(n_epoch):
        _, loss_value = evaluate([train_op, log_likelihood])
        # summary, loss_value = evaluate([train_op, log_likelihood])

        train_loss[i] = loss_value
        # writer.add_summary(summary, i)
    plt.plot(np.arange(n_epoch), -train_loss / len(X_train), label='Train Loss')
    # plt.savefig('../Plots/T_loss_function.pdf')
    return train_loss

def get_predictions(logits,locs,scales):
    pred_weights, pred_means, pred_std = evaluate([tf.nn.softmax(logits), locs, scales])
    return pred_weights, pred_means, pred_std

def plot_pdfs(pred_means,pred_weights,pred_std,num=6,train=True):
    if train:
        obj = [random.randint(0,num_train-1) for x in range(num)]
    else:
        obj = [random.randint(0,num_test-1) for x in range(num)]
    #obj = [93, 402, 120,789,231,4,985]
    print(obj)
    fig, axes = plt.subplots(nrows=num, ncols=1, sharex = True, figsize=(8, 7))
    allfs = []
    for i in range(len(obj)):
        fs = plot_normal_mix(pred_weights[obj][i], pred_means[obj][i],
                    pred_std[obj][i], axes[i], comp=False)
        allfs.append(fs)
        axes[i].axvline(x=y_train[obj][i], color='black', alpha=0.5)
        axes[i].text(0.3, 4.0, 'ID: ' +str(obj[i]), horizontalalignment='center',
        verticalalignment='center')

    plt.xlabel(r' rescaled[$z_{pred}]$', fontsize = 19)
    # plt.savefig('../Plots/T_pdfs.pdf')
    plt.show()

def plot_pred_mean(pred_means,pred_weights,pred_std,ymax,ymin,y_train,select='no'):
    y_pred = np.sum(pred_means*pred_weights, axis = 1)
    y_pred_std = np.sum(pred_std*pred_weights, axis = 1)

    plt.figure(22, figsize=(9,8))

    #ymax=1
    #ymin=0
    # if select == 'yes':
    #     y_pred = y_pred[obj]
    #     y_train = y_train[obj]
    #     y_pred_std = y_pred_std[obj]

    # plt.scatter(y_test, y_pred, facecolors='k', s = 1)

    plt.errorbar( (ymax - ymin)*(y_train)+ymin, (ymax - ymin)*(y_pred)+ymin, yerr= (ymax - ymin)*(y_pred_std), fmt='bo', ecolor='r', ms = 2, alpha = 0.1)

    #switched
    #plt.errorbar(  (ymax - ymin)*(y_pred)+ymin, (ymax - ymin)*(y_train)+ymin, yerr= (ymax - ymin)*(y_pred_std), fmt='bo', ecolor='r', ms = 2, alpha = 0.1)

    #plt.text(0.2, 0.9, train_datafile + ' trained', horizontalalignment='center', verticalalignment='center')
    plt.plot((ymax - ymin)*(y_train)+ymin, (ymax - ymin)*( y_train)+ymin, 'k')

    plt.ylabel(r'$z_{pred}$', fontsize = 19)
    plt.xlabel(r'$z_{true}$', fontsize = 19)
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    plt.title('weight x mean')
    plt.tight_layout()
    # plt.savefig('../Plots/T_pred_mean.pdf')
    plt.show()

def plot_pred_peak(pred_means,pred_weights,pred_std,ymax,ymin,y_train,select='no'):
    def peak(weight,sigma):
        return weight/np.sqrt(2*np.pi*sigma**2)

    peak_max = np.argmax(peak(pred_weights,pred_std),axis=1)
    y_pred = np.array([pred_means[i,peak_max[i]] for i in range(len(y_train))])
    y_pred_std = np.array([pred_std[i,peak_max[i]] for i in range(len(y_train))])
    plt.figure(24, figsize=(9, 8))
    # if select == 'yes':
    #     y_pred = y_pred[obj]
    #     y_train = y_train[obj]
    #     y_pred_std = y_pred_std[obj]
    # plt.scatter(y_test, y_pred, facecolors='k', s = 1)
    plt.errorbar((ymax - ymin)*(y_train)+ymin, (ymax - ymin)*(y_pred)+ymin, yerr= (ymax - ymin)*(
      y_pred_std), fmt='bo', ecolor='r', ms = 2, alpha = 0.1)
    #plt.text(0.2, 0.9, train_datafile + ' trained', horizontalalignment='center', verticalalignment='center')
    plt.plot((ymax - ymin)*(y_test)+ymin, (ymax - ymin)*(y_test)+ymin, 'k')
    plt.ylabel(r'$z_{pred}$', fontsize = 19)
    plt.xlabel(r'$z_{true}$', fontsize = 19)
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    plt.title('highest peak')
    plt.tight_layout()
    plt.show()

def plot_pred_weight(pred_means,pred_weights,pred_std,ymax,ymin,y_train,select='no'):
    weight_max = np.argmax(pred_weights, axis = 1)  ## argmax or max???

    y_pred = np.array([pred_means[i,weight_max[i]] for i in range(len(y_train))])
    y_pred_std = np.array([pred_std[i,weight_max[i]] for i in range(len(y_train))])

    plt.figure(29, figsize=(9, 8))
    # if select == 'yes':
    #     y_pred = y_pred[obj]
    #     y_train = y_train[obj]
    #     y_pred_std = y_pred_std[obj]

    # plt.scatter(y_test, y_pred, facecolors='k', s = 1)
    plt.errorbar((ymax - ymin)*(y_train)+ymin, (ymax - ymin)*(y_pred)+ymin, yerr= (ymax - ymin)*(
      y_pred_std), fmt='bo', ecolor='r', ms = 2, alpha = 0.1)

    #plt.text(0.2, 0.9, train_datafile + ' trained', horizontalalignment='center', verticalalignment='center')
    plt.plot((ymax - ymin)*(y_test)+ymin, (ymax - ymin)*(y_test)+ymin, 'k')
    plt.ylabel(r'$z_{pred}$', fontsize = 19)
    plt.xlabel(r'$z_{true}$', fontsize = 19)
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    plt.title('highest weight')
    plt.tight_layout()
    plt.show()

def per_stats(pred_means,pred_weights,pred_std,ymax,ymin,y_train):
    y_pred = np.sum(pred_means*pred_weights, axis = 1)
    y_pred_std = np.sum(pred_std*pred_weights, axis = 1)
    y_pred = (ymax - ymin)*(y_pred)+ymin
    y_pred_std = (ymax - ymin)*(y_pred_std)
    y_train = (ymax - ymin)*(y_train)+ymin
    diff = y_pred-y_train
    mean_diff = np.mean(diff)
    med_diff = np.median(diff)
    std_diff = np.std(diff)
    mean_sigma = np.mean(y_pred_std)
    med_sigma = np.median(y_pred_std)
    std_sigma = np.std(y_pred_std)
    return mean_diff, med_diff, std_diff, mean_sigma, med_sigma, std_sigma

def testing(X_test,y_test):

    log_likelihood,  logits, locs, scales = mixture_model(X_test,y_test,train=False)
    #_, loss_value = evaluate([train_op, log_likelihood])
    pred_weights, pred_means, pred_std = get_predictions(logits,locs,scales)
    return pred_weights, pred_means, pred_std

def plot_cum_sigma(pred_weights,pred_std,ymax,ymin):
    #y_pred_std = np.sum(pred_std*pred_weights, axis = 1)

    weight_max = np.argmax(pred_weights, axis = 1)  ## argmax or max???
    y_pred_std = np.array([pred_std[i,weight_max[i]] for i in range(len(pred_weights[0]))])
    y_pred_std = (ymax - ymin)*(y_pred_std)
    plt.figure(222)
    plt.hist(y_pred_std,100, density=True, histtype='step',
                           cumulative=True,color='k')
    plt.xlabel('Sigma')
    plt.show()


n_epochs = 302 #3030030 #000 #20000 #100000 #1000 #20000 #20000
# N = 4000  # number of data points  -- replaced by num_trai
D = 5 #6  # number of features  (8 for DES, 6 for COSMOS)
K = 3 # number of mixture components

learning_rate = 1e-3
decay_rate= 0.01 #0.0
step=1000

num_train = 1100000 #2900000 #000#00 #800000 #12000000 #800000
num_test = 500 #5000 #params.num_test # 32


syntheticTrain = True # True # (sim_obs_combine) True -- train using GalaxyPy, False -- train using

save_mod = 'saved_hubs/'+'sdss_colmagUM_synthetic_'+str(syntheticTrain)+'_lr_'+str(learning_rate)+'_dr'+str(decay_rate)+'_step'+str(step)+'_ne'+str(n_epochs)+'_k'+str(K)+'_nt'+str(num_train)



############training

# X_train, y_train, X_test, y_test, ymax, ymin, xmax, xmin = ReadGalaxPy(path_program = '../../Data/fromGalaxev/photozs/datasets/', sim_obs_combine = syntheticTrain)
X_train, y_train, X_test, y_test, ymax, ymin, xmax, xmin = ReadCosmosDraw_UM(path_program = '../../Data/fromGalaxev/photozs/datasets/')


print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))



net_spec = hub.create_module_spec(neural_network_mod)
neural_network = hub.Module(net_spec,name='neural_network',trainable=True)

log_likelihood, train_op, logits, locs, scales  = mixture_model(X_train,y_train,learning_rate=learning_rate,decay_rate=decay_rate)

train_loss = train(log_likelihood,train_op,n_epochs)



#save network
neural_network.export(save_mod,sess)

pred_weights, pred_means, pred_std = get_predictions(logits, locs, scales)
print(pred_means)

plot_pdfs(pred_means,pred_weights,pred_std)

plot_pred_mean(pred_means,pred_weights,pred_std,ymax,ymin,y_train)

mean_diff, med_diff, std_diff, mean_sigma, med_sigma, std_sigma = per_stats(pred_means,pred_weights,pred_std,ymax,ymin,y_train)

plot_cum_sigma(pred_weights,pred_std,ymax,ymin)


plot_pred_peak(pred_means,pred_weights,pred_std,ymax,ymin,y_train)
plot_pred_weight(pred_means,pred_weights,pred_std,ymax,ymin,y_train)

#load network
neural_network_t = hub.Module(save_mod)


##testing

print(20*'='+' Testing ' + 20*'=')
test_weights, test_means, test_std = testing(X_test,y_test)
plot_pdfs(test_means,test_weights,test_std,train=False)

plot_pred_mean(test_means,test_weights,test_std,ymax,ymin,y_test)

plot_cum_sigma(test_weights,test_std,ymax,ymin)

test_mean_diff, test_med_diff, test_std_diff, test_mean_sigma, test_med_sigma, test_std_sigma = per_stats(test_means,test_weights,test_std,ymax,ymin,y_test)

plot_pred_peak(test_means,test_weights,test_std,ymax,ymin,y_test)
plot_pred_weight(test_means,test_weights,test_std,ymax,ymin,y_test)

print(20*'=')
plt.figure(1222)
plt.plot(train_loss)
plt.savefig(save_mod + '/loss.png')
plt.show()
