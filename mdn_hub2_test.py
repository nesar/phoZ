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

print(20*'=~')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
print(20*'=~')

# if datafile == 'GalaxPy':

def ReadGalaxPy(path_program = '../../Data/fromGalaxev/photozs/datasets/', sim_obs_combine = True):
    import os
    import sys
    import glob
    from astropy.table import Table

    # path_program = '../../Data/fromGalaxev/photozs/datasets/'


    class Curated_sample():
        ''' Class to store the redshift and colors of observed galaxies,
            and the redshift, Mpeak, colors, and "weights" of simulated
            galaxies whose colors are compatible with those of observed
            galaxies.

            The observed sample include galaxies from SDSS
            (SDSS+BOSS+eBOSS), DEEP2, and VIPERS.

            The simulated sample was created by sampling the parameter of
            GALAXPY using a LH.

            The weights of simulated galaxies are related to the number
            density of observed galaxies in the same region of the color
            space.

            You only have to care about the method load_structure. '''

        def __init__(self):
            self.arr_c = []
            self.arr_z = []
            self.arr_m = []
            self.arr_w = []

        def append(self, c, z, m, w):
            self.arr_c.append(c)
            self.arr_z.append(z)
            self.arr_m.append(m)
            self.arr_w.append(w)

        def ndarray(self):
            self.arr_c = np.concatenate(self.arr_c)
            self.arr_z = np.concatenate(self.arr_z)
            self.arr_m = np.concatenate(self.arr_m)
            self.arr_w = np.concatenate(self.arr_w)

        def save_struct(self, name):
            np.save(name + 'c.npy', self.arr_c)
            np.save(name + 'z.npy', self.arr_z)
            np.save(name + 'm.npy', self.arr_m)
            np.save(name + 'w.npy', self.arr_w)

        def load_struct(self, name):
            self.arr_c = np.load(name + 'c.npy')
            self.arr_z = np.load(name + 'z.npy')
            self.arr_m = np.load(name + 'm.npy')
            self.arr_w = np.load(name + 'w.npy')

        def duplicate_data(self, zrange):
            aa = np.where((self.arr_w > 50)
                          & (self.arr_z >= zrange[0])
                          & (self.arr_z < zrange[1]))[0]
            print(aa.shape)
            cc = np.repeat(aa, self.arr_w[aa].astype(int))
            self.arr_cn = self.arr_c[cc, :]
            self.arr_zn = self.arr_z[cc]
            self.arr_mn = self.arr_m[cc]


    def read_curated_data():
        run_path = path_program + 'runs/run_z3/'

        sim_q = Curated_sample()  # simulated colors quenched galaxies
        sim_s = Curated_sample()  # simulated colors star-forming galaxies
        obs_q = Curated_sample()  # observed colors quenched galaxies
        obs_s = Curated_sample()  # observed colors star-forming galaxies

        obs_q.load_struct(run_path + 'str_obs_q')
        obs_s.load_struct(run_path + 'str_obs_s')
        sim_q.load_struct(run_path + 'str_sim_q')
        sim_s.load_struct(run_path + 'str_sim_s')

        print(sim_q.arr_c.shape)
        print(sim_s.arr_c.shape)
        print(obs_q.arr_c.shape)
        print(obs_s.arr_c.shape)

        return sim_q, sim_s, obs_q, obs_s


    sim_q, sim_s, obs_q, obs_s = read_curated_data()

    if sim_obs_combine:
        train_datafile = 'GalaxPy'

        # 2.0 ####### TRAIN USING SIMULATION, TEST OBSERVATION ####

        Trainfiles = np.append(sim_q.arr_c, sim_s.arr_c, axis=0)
        TrainZ = np.append(sim_q.arr_z, sim_s.arr_z, axis=0)

        Trainfiles = np.delete(Trainfiles, (4), axis=1)  ## deleting z-Y

        Testfiles = np.append(obs_q.arr_c, obs_s.arr_c, axis=0)
        TestZ = np.append(obs_q.arr_z, obs_s.arr_z, axis=0)

        TrainshuffleOrder = np.arange(Trainfiles.shape[0])
        np.random.shuffle(TrainshuffleOrder)

        Trainfiles = Trainfiles[TrainshuffleOrder]
        TrainZ = TrainZ[TrainshuffleOrder]

        TestshuffleOrder = np.arange(Testfiles.shape[0])
        np.random.shuffle(TestshuffleOrder)

        Testfiles = Testfiles[TestshuffleOrder]
        TestZ = TestZ[TestshuffleOrder]

        X_train = Trainfiles[:num_train]  # color mag
        X_test = Trainfiles[:num_test]  # color mag

        y_train = TrainZ[:num_train]  # spec z
        y_test = TrainZ[:num_test]  # spec z

    else:
        train_datafile = 'SDSS'
        # 1.1 ####### SIMULATED: QUENCHED ONLY ############
        # Trainfiles = sim_q.arr_c
        # TrainZ = sim_q.arr_z

        # 1.2 ### SIMULATED: QUENCHED + STAR FORMATION ####

        # Trainfiles =np.append( sim_q.arr_c, sim_s.arr_c, axis = 0)
        # TrainZ = np.append( sim_q.arr_z, sim_s.arr_z, axis = 0)

        # 1.3 ####### OBSERVED: QUENCHED + STAR FORMATION ####

        Trainfiles = np.append(obs_q.arr_c, obs_s.arr_c, axis=0)
        TrainZ = np.append(obs_q.arr_z, obs_s.arr_z, axis=0)

        TrainshuffleOrder = np.arange(Trainfiles.shape[0])
        np.random.shuffle(TrainshuffleOrder)

        Trainfiles = Trainfiles[TrainshuffleOrder]
        TrainZ = TrainZ[TrainshuffleOrder]

        # 1 #################################

        X_train = Trainfiles[:num_train]  # color mag
        X_test = Trainfiles[num_train + 1: num_train + num_test]  # color mag

        X_train = Trainfiles[:num_train]  # color mag
        X_test = Trainfiles[num_train + 1: num_train + num_test]  # color mag

        y_train = TrainZ[:num_train]  # spec z
        y_test = TrainZ[num_train + 1: num_train + num_test]  # spec z

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
        train_loss[i] = loss_value
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
    y_pred_std = np.array( [pred_std[i,weight_max[i] ] for i in range(len(pred_weights[0]))])
    y_pred_std = (ymax - ymin)*(y_pred_std)
    plt.figure(222)
    plt.hist(y_pred_std,100, density=True, histtype='step',
                           cumulative=True,color='k')
    plt.xlabel('Sigma')
    plt.show()



n_epochs = 2000 #1000 #20000 #20000
# N = 4000  # number of data points  -- replaced by num_trai
D = 5 #6  # number of features  (8 for DES, 6 for COSMOS)
K = 3 # number of mixture components


learning_rate = 5e-3
decay_rate= 0.0
step=100


num_train = 800000 #800000
num_test = 5000 #10000 #params.num_test # 32


save_mod = 'hub_mod_lr_1'+str(learning_rate)+'_dr'+str(decay_rate)+'_step'+str(step)+'_ne'+str(n_epochs)+'_k'+str(K)+'_nt'+str(num_train)



n_epochs = 1000 #1000 #20000 #20000
# N = 4000  # number of data points  -- replaced by num_trai
D = 5 #6  # number of features  (8 for DES, 6 for COSMOS)
K = 3 # number of mixture components


learning_rate = 5e-3
decay_rate= 0.0
step=100


num_train = 100000 #800000
num_test = 5000 #10000 #params.num_test # 32


save_mod = 'hub_mod_lr_1'+str(learning_rate)+'_dr'+str(decay_rate)+'_step'+str(step)+'_ne'+str(n_epochs)+'_k'+str(K)+'_nt'+str(num_train)




############training

X_train, y_train, X_test, y_test, ymax, ymin, xmax, xmin = ReadGalaxPy(path_program = '../../Data/fromGalaxev/photozs/datasets/', sim_obs_combine = True)

print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))



net_spec = hub.create_module_spec(neural_network_mod)
neural_network = hub.Module(net_spec,name='neural_network',trainable=True)

log_likelihood, train_op, logits, locs, scales  = mixture_model(X_train,y_train,learning_rate=learning_rate,decay_rate=decay_rate)

# train_loss = train(log_likelihood,train_op,n_epochs)
#save network
# neural_network.export(save_mod,sess)

# pred_weights, pred_means, pred_std = get_predictions(logits, locs, scales)
# print(pred_means)
#
# plot_pdfs(pred_means,pred_weights,pred_std)
#
# plot_pred_mean(pred_means,pred_weights,pred_std,ymax,ymin,y_train)
#
# mean_diff, med_diff, std_diff, mean_sigma, med_sigma, std_sigma = per_stats(pred_means,pred_weights,pred_std,ymax,ymin,y_train)
#
# plot_cum_sigma(pred_weights,pred_std,ymax,ymin)
#
#
# plot_pred_peak(pred_means,pred_weights,pred_std,ymax,ymin,y_train)
# plot_pred_weight(pred_means,pred_weights,pred_std,ymax,ymin,y_train)

#load network
neural_network_t = hub.Module(save_mod)


##testing


test_weights, test_means, test_std = testing(X_test,y_test)
plot_pdfs(test_means,test_weights,test_std,train=False)

plot_pred_mean(test_means,test_weights,test_std,ymax,ymin,y_test)

plot_cum_sigma(test_weights,test_std,ymax,ymin)

test_mean_diff, test_med_diff, test_std_diff, test_mean_sigma, test_med_sigma, test_std_sigma = per_stats(test_means,test_weights,test_std,ymax,ymin,y_test)

plot_pred_peak(test_means,test_weights,test_std,ymax,ymin,y_test)
plot_pred_weight(test_means,test_weights,test_std,ymax,ymin,y_test)

