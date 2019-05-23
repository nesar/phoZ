from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style(style ="white")
import tensorflow as tf

from edward.models import Categorical, Mixture, Normal
from scipy import stats
from sklearn.model_selection import train_test_split


# import params
# https://github.com/cbonnett/MDN_Edward_Keras_TF/blob/master/MDN_Edward_Keras_TF.ipynb
# http://cbonnett.github.io/MDN.html


import SetPub
SetPub.set_pub()



################################################################


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

def sample_from_mixture(x, pred_weights, pred_means, pred_std, amount):
  """Draws samples from mixture model.

  Returns 2 d array with input X and sample from prediction of mixture model.
  """
  samples = np.zeros((amount, 2))
  n_mix = len(pred_weights[0])
  to_choose_from = np.arange(n_mix)
  for j, (weights, means, std_devs) in enumerate(
          zip(pred_weights, pred_means, pred_std)):
    index = np.random.choice(to_choose_from, p=weights)
    samples[j, 1] = np.random.normal(means[index], std_devs[index], size=1)
    samples[j, 0] = x[j]
    if j == amount - 1:
      break
  return samples



######################################################



# n_epoch = 10000
# # N = 4000  # number of data points  -- replaced by num_trai
# D = 8  # number of features  (8 for DES, 6 for COSMOS)
# K = 3 # number of mixture components
#
#
# learning_rate = 1e-3
#
# num_train = 20000 #params.num_train # 512
# num_test = 10000 #params.num_test # 32

############## i/o ###########################################


n_epoch = 10000 #20000 #20000
# N = 4000  # number of data points  -- replaced by num_trai
D = 5 #6  # number of features  (8 for DES, 6 for COSMOS)
K = 3 # number of mixture components


learning_rate = 5e-3

## galacticus ##
num_train = 800000#400000 #400000 # #params.num_train # 512
num_test = 750 #10000 #params.num_test # 32
#


#
datafile = ['DES', 'COSMOS', 'Galacticus', 'GalaxPy', 'DC2'][4]  ############ added dc2 changes



############ model 1 ########################################


modelFile = 'GalaxPy'  ############ added dc2 changes


sim_obs_combine = False

if sim_obs_combine: ModelName = './Model/Edward_posterior_' + modelFile + '_nComp' + str(K) + \
                                '_ntrain' + str(num_train) + '_nepoch' + str(n_epoch) + '_lr' + \
                                str(learning_rate) + '_sim_obs_combine'
else: ModelName = './Model/Edward_posterior_' + modelFile + '_nComp' + str(K) + '_ntrain' + str(
    num_train) + '_nepoch' + str(n_epoch)  + '_lr' + str(learning_rate)  + '_obs_only'




np.random.seed(7643)




if datafile == 'GalaxPy':
    import numpy as np
    import os
    import sys
    import glob
    from astropy.table import Table

    # path_program = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
    path_program = '../../Data/fromGalaxev/photozs/datasets/'

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

    # 2.0 ####### TRAIN USING SIMULATION, TEST OBSERVATION ####

        Trainfiles =np.append( sim_q.arr_c, sim_s.arr_c, axis = 0)
        TrainZ = np.append( sim_q.arr_z, sim_s.arr_z, axis = 0)

        Trainfiles = np.delete(Trainfiles,(4), axis=1)   ## deleting z-Y

        Testfiles =np.append( obs_q.arr_c, obs_s.arr_c, axis = 0)
        TestZ = np.append( obs_q.arr_z, obs_s.arr_z, axis = 0)


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
        # 1.1 ####### SIMULATED: QUENCHED ONLY ############
        # Trainfiles = sim_q.arr_c
        # TrainZ = sim_q.arr_z

        # 1.2 ### SIMULATED: QUENCHED + STAR FORMATION ####

        # Trainfiles =np.append( sim_q.arr_c, sim_s.arr_c, axis = 0)
        # TrainZ = np.append( sim_q.arr_z, sim_s.arr_z, axis = 0)


        # 1.3 ####### OBSERVED: QUENCHED + STAR FORMATION ####

        Trainfiles =np.append( obs_q.arr_c, obs_s.arr_c, axis = 0)
        TrainZ = np.append( obs_q.arr_z, obs_s.arr_z, axis = 0)

        TrainshuffleOrder = np.arange(Trainfiles.shape[0])
        np.random.shuffle(TrainshuffleOrder)

        Trainfiles = Trainfiles[TrainshuffleOrder]
        TrainZ = TrainZ[TrainshuffleOrder]

        # 1 #################################

        X_train = Trainfiles[:num_train]  # color mag
        X_test = Trainfiles[num_train + 1: num_train + num_test + 1]  # color mag

        X_train = Trainfiles[:num_train]  # color mag
        X_test = Trainfiles[num_train + 1: num_train + num_test + 1]  # color mag

        y_train = TrainZ[:num_train]  # spec z
        y_test = TrainZ[num_train + 1: num_train + num_test + 1]  # spec z



    ############## THINGS ARE SAME AFTER THIS ###########

    ## rescaling xmax/xmin
    xmax = np.max([np.max(X_train, axis = 0), np.max(X_test, axis=0)], axis = 0)
    xmin = np.min([np.min(X_train, axis = 0), np.min(X_test, axis=0)], axis = 0)

    X_train = (X_train - xmin) / (xmax - xmin)
    X_test = (X_test - xmin) / (xmax - xmin)

    #### RESCALING X_train, X_test NOT done yet -- (g-i), (r-i) ... and i mag -->> Color/Mag issue

    ymax = np.max([y_train.max(), y_test.max()])
    ymin = np.min([y_train.min(), y_test.min()])

    y_train = (y_train - ymin) / (ymax - ymin)
    y_test = (y_test - ymin) / (ymax - ymin)



############ added dc2 changes

if datafile == 'DC2':
    ## DC2 params
    num_train_dc2 = 7000  # 400000 #400000 # #params.num_train # 512
    num_test_dc2 = 750  # 10000 #params.num_test # 32
    #

    import numpy as np
    import os
    import sys
    import glob
    from astropy.table import Table

    # path_program = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
    path_program = '../../Data/fromGalaxev/photozs/datasets/'

    fname = [path_program + 'DC2/z_0_1.step_all.healpix_9944.hdf5']


    for fileName in fname:
        print(fname)

        ugrizy = np.array(Table.read(fileName, path='data')['mag_u_obs', 'mag_g_obs', 'mag_r_obs', 'mag_i_obs', 'mag_z_obs', 'mag_y_obs', 'redshift'])

        # ugrizy = np.array(sample['mag_u_obs', 'mag_g_obs', 'mag_r_obs', 'mag_i_obs', 'mag_z_obs', 'mag_y_obs'])

        Trainfiles = np.array([ugrizy['mag_u_obs'] - ugrizy['mag_g_obs'], ugrizy['mag_g_obs'] - ugrizy['mag_r_obs'], ugrizy['mag_r_obs'] - ugrizy['mag_i_obs'], ugrizy['mag_i_obs'] - ugrizy['mag_z_obs'], ugrizy['mag_i_obs'] ]).T
        TrainZ = ugrizy['redshift']

        TrainshuffleOrder = np.arange(Trainfiles.shape[0])
        np.random.shuffle(TrainshuffleOrder)

        Trainfiles = Trainfiles[TrainshuffleOrder]
        TrainZ = TrainZ[TrainshuffleOrder]

        # 1 #################################

        X_train = Trainfiles[:num_train_dc2, :]  # color mag
        X_test = Trainfiles[num_train_dc2 + 1: num_train_dc2 + num_test_dc2, :]  # color mag
        #
        # X_train = Trainfiles[:num_train]  # color mag
        # X_test = Trainfiles[num_train + 1: num_train + num_test]  # color mag

        y_train = TrainZ[:num_train_dc2]  # spec z
        y_test = TrainZ[num_train_dc2 + 1: num_train_dc2 + num_test_dc2]  # spec z



    ############## THINGS ARE SAME AFTER THIS ###########

    ## rescaling xmax/xmin
    xmax = np.max([np.max(X_train, axis = 0), np.max(X_test, axis=0)], axis = 0)
    xmin = np.min([np.min(X_train, axis = 0), np.min(X_test, axis=0)], axis = 0)

    X_train = (X_train - xmin) / (xmax - xmin)
    X_test = (X_test - xmin) / (xmax - xmin)

    #### RESCALING X_train, X_test NOT done yet -- (g-i), (r-i) ... and i mag -->> Color/Mag issue

    ymax = np.max([y_train.max(), y_test.max()])
    ymin = np.min([y_train.min(), y_test.min()])

    y_train = (y_train - ymin) / (ymax - ymin)
    y_test = (y_test - ymin) / (ymax - ymin)






print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))




##########  TESTING SCRIPT ################


X_ph_new = tf.placeholder(tf.float32, [None, D])
y_ph_new = tf.placeholder(tf.float32, [None])


def neural_network(X):
  """loc, scale, logits = NN(x; theta)"""
  # 2 hidden layers with 15 hidden units
  net = tf.layers.dense(X, 15, activation=tf.nn.relu)
  net = tf.layers.dense(net, 15, activation=tf.nn.relu)
  locs = tf.layers.dense(net, K, activation=None)
  scales = tf.layers.dense(net, K, activation=tf.exp)
  logits = tf.layers.dense(net, K, activation=None)
  return locs, scales, logits


locs_new, scales_new, logits_new = neural_network(X_ph_new)

cat_new = Categorical(logits=logits_new)
components_new = [Normal(loc=loc, scale=scale) for loc, scale
              in zip(tf.unstack(tf.transpose(locs_new)),
                     tf.unstack(tf.transpose(scales_new)))]
y_new = Mixture(cat=cat_new, components=components_new, value=tf.zeros_like(y_ph_new))
## Note: A bug exists in Mixture which prevents samples from it to have
## a shape of [None]. For now fix it using the value argument, as
## sampling is not necessary for MAP estimation anyways.

######################### inference ##############################

# There are no latent variables to infer. Thus inference is concerned
# with only training model parameters, which are baked into how we
# specify the neural networks.



inference_new = ed.MAP(data={y_new: y_ph_new})
optimizer_new = tf.train.AdamOptimizer(learning_rate=learning_rate)
inference_new.initialize(optimizer=optimizer_new, var_list=tf.trainable_variables())


new_saver = tf.train.Saver()
# new_saver = tf.train.import_meta_graph(ModelName+'.meta')

sess_load = ed.get_session()
# tf.global_variables_initializer().run()


# new_saver = tf.train.Saver()

new_saver.restore(sess_load, ModelName)
print("Model restored.")


pred_weights_new, pred_means_new, pred_std_new = sess_load.run(
    [tf.nn.softmax(logits_new), locs_new, scales_new], feed_dict={X_ph_new: X_test})





######### PLOT testing #############



obj = [93, 142, 120]

PlotMix = True

if PlotMix:
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex = True, figsize=(8, 7))

    plot_normal_mix(pred_weights_new[obj][0], pred_means_new[obj][0],
                    pred_std_new[obj][0], axes[0], comp=False, label='Obs only')
    axes[0].axvline(x=y_test[obj][0], color='black', alpha=0.5)
    axes[0].text(0.3, 4.0, 'ID: ' +str(obj[0]), horizontalalignment='center',
                 verticalalignment='center')


    plot_normal_mix(pred_weights_new[obj][1], pred_means_new[obj][1],
                    pred_std_new[obj][1], axes[1], comp=False)
    axes[1].axvline(x=y_test[obj][1], color='black', alpha=0.5)
    axes[1].text(0.3, 4.0, 'ID: ' +str(obj[1]), horizontalalignment='center',
                 verticalalignment='center')

    plot_normal_mix(pred_weights_new[obj][2], pred_means_new[obj][2],
                    pred_std_new[obj][2], axes[2], comp=False)
    axes[2].axvline(x=y_test[obj][2], color='black', alpha=0.5)
    axes[2].text(0.3, 4.0, 'ID: ' +str(obj[2]), horizontalalignment='center',
                 verticalalignment='center')

    plt.xlabel(r' rescaled[$z_{pred}]$', fontsize = 19)

    plt.show()





    a = sample_from_mixture(X_test[:,1], pred_weights_new, pred_means_new,
                            pred_std_new, amount=len(X_test))
    sns.jointplot(a[:, 0], a[:, 1], kind="hex", color="#4CB391")
    plt.show()

# PlotMix(obj)
#############################################################################3


sess_load.close()
tf.reset_default_graph()



####################### model 1 ends ###########################



############ model 2 ########################################


sim_obs_combine = True

if sim_obs_combine: ModelName = './Model/Edward_posterior_' + modelFile + '_nComp' + str(K) + \
                                '_ntrain' + str(num_train) + '_nepoch' + str(n_epoch) + '_lr' + \
                                str(learning_rate) + '_sim_obs_combine'
else: ModelName = './Model/Edward_posterior_' + modelFile + '_nComp' + str(K) + '_ntrain' + str(
    num_train) + '_nepoch' + str(n_epoch)  + '_lr' + str(learning_rate)  + '_obs_only'




np.random.seed(42)


## dc2 changes

#
# datafile = 'GalaxPy'  ############ added dc2 changes
# if datafile == 'GalaxPy':
#     import numpy as np
#     import os
#     import sys
#     import glob
#     from astropy.table import Table
#
#     # path_program = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
#     path_program = '../../Data/fromGalaxev/photozs/datasets/'
#
#     class Curated_sample():
#         ''' Class to store the redshift and colors of observed galaxies,
#             and the redshift, Mpeak, colors, and "weights" of simulated
#             galaxies whose colors are compatible with those of observed
#             galaxies.
#
#             The observed sample include galaxies from SDSS
#             (SDSS+BOSS+eBOSS), DEEP2, and VIPERS.
#
#             The simulated sample was created by sampling the parameter of
#             GALAXPY using a LH.
#
#             The weights of simulated galaxies are related to the number
#             density of observed galaxies in the same region of the color
#             space.
#
#             You only have to care about the method load_structure. '''
#
#         def __init__(self):
#             self.arr_c = []
#             self.arr_z = []
#             self.arr_m = []
#             self.arr_w = []
#
#         def append(self, c, z, m, w):
#             self.arr_c.append(c)
#             self.arr_z.append(z)
#             self.arr_m.append(m)
#             self.arr_w.append(w)
#
#         def ndarray(self):
#             self.arr_c = np.concatenate(self.arr_c)
#             self.arr_z = np.concatenate(self.arr_z)
#             self.arr_m = np.concatenate(self.arr_m)
#             self.arr_w = np.concatenate(self.arr_w)
#
#         def save_struct(self, name):
#             np.save(name + 'c.npy', self.arr_c)
#             np.save(name + 'z.npy', self.arr_z)
#             np.save(name + 'm.npy', self.arr_m)
#             np.save(name + 'w.npy', self.arr_w)
#
#         def load_struct(self, name):
#             self.arr_c = np.load(name + 'c.npy')
#             self.arr_z = np.load(name + 'z.npy')
#             self.arr_m = np.load(name + 'm.npy')
#             self.arr_w = np.load(name + 'w.npy')
#
#         def duplicate_data(self, zrange):
#             aa = np.where((self.arr_w > 50)
#                           & (self.arr_z >= zrange[0])
#                           & (self.arr_z < zrange[1]))[0]
#             print(aa.shape)
#             cc = np.repeat(aa, self.arr_w[aa].astype(int))
#             self.arr_cn = self.arr_c[cc, :]
#             self.arr_zn = self.arr_z[cc]
#             self.arr_mn = self.arr_m[cc]
#
#
#     def read_curated_data():
#         run_path = path_program + 'runs/run_z3/'
#
#         sim_q = Curated_sample()  # simulated colors quenched galaxies
#         sim_s = Curated_sample()  # simulated colors star-forming galaxies
#         obs_q = Curated_sample()  # observed colors quenched galaxies
#         obs_s = Curated_sample()  # observed colors star-forming galaxies
#
#         obs_q.load_struct(run_path + 'str_obs_q')
#         obs_s.load_struct(run_path + 'str_obs_s')
#         sim_q.load_struct(run_path + 'str_sim_q')
#         sim_s.load_struct(run_path + 'str_sim_s')
#
#         print(sim_q.arr_c.shape)
#         print(sim_s.arr_c.shape)
#         print(obs_q.arr_c.shape)
#         print(obs_s.arr_c.shape)
#
#         return sim_q, sim_s, obs_q, obs_s
#
#
#     sim_q, sim_s, obs_q, obs_s = read_curated_data()
#
#
#     if sim_obs_combine:
#
#     # 2.0 ####### TRAIN USING SIMULATION, TEST OBSERVATION ####
#
#         Trainfiles =np.append( sim_q.arr_c, sim_s.arr_c, axis = 0)
#         TrainZ = np.append( sim_q.arr_z, sim_s.arr_z, axis = 0)
#
#         Trainfiles = np.delete(Trainfiles,(4), axis=1)   ## deleting z-Y
#
#         Testfiles =np.append( obs_q.arr_c, obs_s.arr_c, axis = 0)
#         TestZ = np.append( obs_q.arr_z, obs_s.arr_z, axis = 0)
#
#
#         TrainshuffleOrder = np.arange(Trainfiles.shape[0])
#         np.random.shuffle(TrainshuffleOrder)
#
#         Trainfiles = Trainfiles[TrainshuffleOrder]
#         TrainZ = TrainZ[TrainshuffleOrder]
#
#         TestshuffleOrder = np.arange(Testfiles.shape[0])
#         np.random.shuffle(TestshuffleOrder)
#
#         Testfiles = Testfiles[TestshuffleOrder]
#         TestZ = TestZ[TestshuffleOrder]
#
#
#         X_train = Trainfiles[:num_train]  # color mag
#         X_test = Trainfiles[:num_test]  # color mag
#
#         y_train = TrainZ[:num_train]  # spec z
#         y_test2 = TrainZ[:num_test]  # spec z
#
#     else:
#         # 1.1 ####### SIMULATED: QUENCHED ONLY ############
#         # Trainfiles = sim_q.arr_c
#         # TrainZ = sim_q.arr_z
#
#         # 1.2 ### SIMULATED: QUENCHED + STAR FORMATION ####
#
#         # Trainfiles =np.append( sim_q.arr_c, sim_s.arr_c, axis = 0)
#         # TrainZ = np.append( sim_q.arr_z, sim_s.arr_z, axis = 0)
#
#
#         # 1.3 ####### OBSERVED: QUENCHED + STAR FORMATION ####
#
#         Trainfiles =np.append( obs_q.arr_c, obs_s.arr_c, axis = 0)
#         TrainZ = np.append( obs_q.arr_z, obs_s.arr_z, axis = 0)
#
#         TrainshuffleOrder = np.arange(Trainfiles.shape[0])
#         np.random.shuffle(TrainshuffleOrder)
#
#         Trainfiles = Trainfiles[TrainshuffleOrder]
#         TrainZ = TrainZ[TrainshuffleOrder]
#
#         # 1 #################################
#
#         X_train = Trainfiles[:num_train]  # color mag
#         X_test = Trainfiles[num_train + 1: num_train + num_test + 1]  # color mag
#
#         X_train = Trainfiles[:num_train]  # color mag
#         X_test = Trainfiles[num_train + 1: num_train + num_test + 1]  # color mag
#
#         y_train = TrainZ[:num_train]  # spec z
#         y_test2 = TrainZ[num_train + 1: num_train + num_test + 1]  # spec z
#
#
#
#     ############## THINGS ARE SAME AFTER THIS ###########
#
#     ## rescaling xmax/xmin
#     xmax = np.max([np.max(X_train, axis = 0), np.max(X_test, axis=0)], axis = 0)
#     xmin = np.min([np.min(X_train, axis = 0), np.min(X_test, axis=0)], axis = 0)
#
#     X_train = (X_train - xmin) / (xmax - xmin)
#     X_test = (X_test - xmin) / (xmax - xmin)
#
#     #### RESCALING X_train, X_test NOT done yet -- (g-i), (r-i) ... and i mag -->> Color/Mag issue
#
#     ymax2 = np.max([y_train.max(), y_test2.max()])
#     ymin2 = np.min([y_train.min(), y_test2.min()])
#
#     y_train = (y_train - ymin2) / (ymax2 - ymin2)
#     y_test2 = (y_test2 - ymin2) / (ymax2 - ymin2)
#

y_test2 = y_test
ymax2 = ymax
ymin2 = ymin



print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test2.shape))




##########  TESTING SCRIPT ################


X_ph_new2 = tf.placeholder(tf.float32, [None, D])
y_ph_new2 = tf.placeholder(tf.float32, [None])


def neural_network(X):
  """loc, scale, logits = NN(x; theta)"""
  # 2 hidden layers with 15 hidden units
  net = tf.layers.dense(X, 15, activation=tf.nn.relu)
  net = tf.layers.dense(net, 15, activation=tf.nn.relu)
  locs = tf.layers.dense(net, K, activation=None)
  scales = tf.layers.dense(net, K, activation=tf.exp)
  logits = tf.layers.dense(net, K, activation=None)
  return locs, scales, logits


locs_new2, scales_new2, logits_new2 = neural_network(X_ph_new2)

cat_new2 = Categorical(logits=logits_new2)
components_new2 = [Normal(loc=loc, scale=scale) for loc, scale
              in zip(tf.unstack(tf.transpose(locs_new2)),
                     tf.unstack(tf.transpose(scales_new2)))]
y_new2 = Mixture(cat=cat_new2, components=components_new2, value=tf.zeros_like(y_ph_new2))
## Note: A bug exists in Mixture which prevents samples from it to have
## a shape of [None]. For now fix it using the value argument, as
## sampling is not necessary for MAP estimation anyways.

######################### inference ##############################

# There are no latent variables to infer. Thus inference is concerned
# with only training model parameters, which are baked into how we
# specify the neural networks.



inference_new2 = ed.MAP(data={y_new2: y_ph_new2})
optimizer_new2 = tf.train.AdamOptimizer(learning_rate=learning_rate)
inference_new2.initialize(optimizer=optimizer_new2, var_list=tf.trainable_variables())


new_saver2 = tf.train.Saver()
# new_saver = tf.train.import_meta_graph(ModelName+'.meta')

sess_load2 = ed.get_session()
# tf.global_variables_initializer().run()


# new_saver = tf.train.Saver()

new_saver2.restore(sess_load2, ModelName)
print("Model restored.")


pred_weights_new2, pred_means_new2, pred_std_new2 = sess_load2.run(
    [tf.nn.softmax(logits_new2), locs_new2, scales_new2], feed_dict={X_ph_new2: X_test})





######### PLOT testing #############



PlotMix = True

if PlotMix:
    # fig, axes = plt.subplots(nrows=3, ncols=1, sharex = True, figsize=(8, 7))

    plot_normal_mix(pred_weights_new2[obj][0], pred_means_new2[obj][0],
                    pred_std_new2[obj][0], axes[0], comp=False, label='Combine')
    axes[0].axvline(x=y_test2[obj][0], color='black', alpha=0.5)
    axes[0].text(0.3, 4.0, 'ID: ' +str(obj[0]), horizontalalignment='center',
                 verticalalignment='center')


    plot_normal_mix(pred_weights_new2[obj][1], pred_means_new2[obj][1],
                    pred_std_new2[obj][1], axes[1], comp=False)
    axes[1].axvline(x=y_test2[obj][1], color='black', alpha=0.5)
    axes[1].text(0.3, 4.0, 'ID: ' +str(obj[1]), horizontalalignment='center',
                 verticalalignment='center')

    plot_normal_mix(pred_weights_new2[obj][2], pred_means_new2[obj][2],
                    pred_std_new2[obj][2], axes[2], comp=False)
    axes[2].axvline(x=y_test2[obj][2], color='black', alpha=0.5)
    axes[2].text(0.3, 4.0, 'ID: ' +str(obj[2]), horizontalalignment='center',
                 verticalalignment='center')

    plt.xlabel(r' rescaled[$z_{pred}]$', fontsize = 19)

    plt.show()





    a = sample_from_mixture(X_test[:,1], pred_weights_new2, pred_means_new2,
                            pred_std_new2, amount=len(X_test))
    sns.jointplot(a[:, 0], a[:, 1], kind="hex", color="#4CB391")
    plt.show()

# PlotMix(obj)
#############################################################################3



## Overall mean --- weight * mean
ifPlotWeighted = True

plt.figure(22, figsize=(7, 7))

if ifPlotWeighted:

    y_pred = np.sum(pred_means_new*pred_weights_new, axis = 1)
    y_pred_std_new = np.sum(pred_std_new*pred_weights_new, axis = 1)

    # # plt.scatter(y_test, y_pred, facecolors='k', s = 1)
    # plt.errorbar((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred), yerr= (ymax - ymin)*(
    #   ymin + y_pred_std_new), fmt='bo', ecolor='r', ms = 2, alpha = 0.1)
    #
    plt.errorbar((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred), yerr= (ymax - ymin)*(
      ymin + y_pred_std_new), fmt='ro', ecolor='r', ms = 3, alpha = 0.3, label = 'SDSS trained '
                                                                                 'model')



    # plt.text(0.8, 2.0, datafile, horizontalalignment='center', verticalalignment='center')
    # plt.ylabel(r'$z_{pred}$', fontsize = 19)
    # plt.xlabel(r'$z_{true}$', fontsize = 19)

    # plt.title('weight x mean')
    # plt.tight_layout()
    # plt.show()



## Overall mean --- weight * mean
ifPlotWeighted = True


if ifPlotWeighted:

    y_pred_new2 = np.sum(pred_means_new2*pred_weights_new2, axis = 1)
    y_pred_std_new2 = np.sum(pred_std_new2*pred_weights_new2, axis = 1)

    plt.figure(22, figsize=(6,6))


    # # plt.scatter(y_test2, y_pred, facecolors='k', s = 1)
    # plt.errorbar((ymax2 - ymin2)*(ymin2 + y_test2), (ymax2 - ymin2)*(ymin2 + y_pred), yerr= (ymax2 - ymin2)*(
    #   ymin2 + y_pred_std_new2), fmt='bo', ecolor='r', ms = 2, alpha = 0.1)
    #
    plt.errorbar((ymax2 - ymin2)*(ymin2 + y_test2), (ymax2 - ymin2)*(ymin2 + y_pred_new2), yerr= (ymax2 - ymin2)*(
      ymin2 + y_pred_std_new2), fmt='bo', ecolor='b', ms = 3, alpha = 0.3, label = 'GALAXPY '
                                                                                   'trained model')



    # plt.text(0.8, 2.0, datafile, horizontalalignment='center', verticalalignment='center')


plt.plot(423)
sns.jointplot(x=(ymax2 - ymin2)*(ymin2 + y_test2), y=(ymax2 - ymin2)*(ymin2 + y_pred_new2),
              kind="kde")
sns.jointplot(x=(ymax - ymin)*(ymin + y_test), y=(ymax - ymin)*(ymin + y_pred),kind="kde")

# plt.hist2d((ymax - ymin)*(ymin + y_test),(ymax - ymin)*(ymin + y_pred),100,cmap='jet')


# sess_load.close()




def sigmaNMAD(z_spec, z_pho):
    return 1.48*np.median( np.abs( z_pho - z_spec)/(1 + z_spec))
    # else: return 1.48*np.median( np.abs( z_pho - z_spec)/(1 + z_spec),)


def outlierFrac(z_spec, z_pho, threshold = 0.15):
    outliers = z_pho[ (np.abs(z_spec - z_pho)) >= threshold*z_spec ]
    return 100.0*len(outliers)/np.shape(z_pho)[0]



sigmaNMAD_obs_all =  sigmaNMAD( (ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin +
                                                                                      y_pred) )

sigmaNMAD_combine_all =  sigmaNMAD( (ymax2 - ymin2)*(ymin2 + y_test2), (ymax2 - ymin2)*(ymin2 +
                                                                                      y_pred_new2) )


print( 'sigmaNMAD_combine_all ', sigmaNMAD_combine_all  ) #
# combine
print( 'sigmaNMAD_obs_all', sigmaNMAD_obs_all ) # obs only


outFr_obs_all = outlierFrac( (ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred), 0.15 )

outFr_combine_all =  outlierFrac( (ymax2 - ymin2)*(ymin2 + y_test2), (ymax2 - ymin2)*(ymin2 + y_pred_new2) , 0.15)


print( 'outFr_combine_all ', outFr_combine_all  ) ## combine
print( 'outFr_obs_all', outFr_obs_all ) # obs only



####################### model 2 ends ###########################




fig = plt.figure(22)

plt.xlim(0, 1)
plt.ylim(0, 1)

# plt.text(0.1, 0.9, r'$\sigma_{NMAD}$ = %.3f'%sigmaNMAD_obs_all, color = 'red' , size = 20)
# plt.text(0.1, 0.85, r'$\sigma_{NMAD}$ = %.3f'%sigmaNMAD_combine_all, color = 'blue' , size = 20)

# plt.text(0.1, 0.9, r'GalaxPy training', color = 'red' , size = 20)
# plt.text(0.1, 0.85, r'SDSS training', color = 'blue' , size = 20)

plt.plot([0, 1], [0, 1], 'k')
plt.plot([0, 1], 0.85*np.array([0, 1]), 'k-.')
plt.plot([0, 1], 1.15*np.array([0, 1]), 'k-.')

plt.ylabel(r'$z_{phot}$', fontsize=19)
plt.xlabel(r'$z_{spec}$', fontsize=19)
#
# plt.ylabel(r'Photometric redshift', fontsize=19)
# plt.xlabel(r'Spectroscopic redshift', fontsize=19)
plt.xlim(0.0, 1)
plt.ylim(0.0, 1)

# plt.legend(fontsize = 'large', markerscale=3., numpoints=3)
# plt.title('weight x mean')
plt.tight_layout()

plt.axes().set_aspect('equal')

# plt.xscale('log')
# plt.yscale('log')

leg = plt.legend(fontsize = 'large', markerscale=2., numpoints=2)

for artist, text in zip(leg.legendHandles, leg.get_texts()):
    col = artist.get_color()
    if isinstance(col, np.ndarray):
        col = col[0]
    text.set_color(col)
    text.set_alpha(1.0)


plt.savefig('phoz_compare.png', bbox_inches='tight')

plt.show()



def weightedDiff(z_spec, z_pho):
    return ( z_pho - z_spec)/(1 + z_spec)



# fig.add_subplot(212, adjustable='box', aspect=0.3)
# plt.plot( (ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred), 'ro')
# plt.plot( (ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred), 'bo')


plt.show()

#######################################

bins = np.linspace(0, 1, 10)
# z_spec_bin = np.histogram( (ymax2 - ymin2)*(ymin2 + y_test2), bins)[0]
# z_phot_bin = np.histogram( (ymax2 - ymin2)*(ymin2 + y_pred_new2), bins)[0]


z_spec2 = (ymax2 - ymin2)*(ymin2 + y_test2)
z_phot2 = (ymax2 - ymin2)*(ymin2 + y_pred_new2)

z_spec_digitize2 = np.digitize( z_spec2, bins)

# for n in range(z_spec.size):
#     print(bins[z_spec_digitize[n]-1], "<=", z_spec[n], "<", bins[z_spec_digitize[n]])


sigmaNMAD_combine = np.zeros(shape=10)
outFr_combine = np.zeros(shape=10)


for ind in range(bins.shape[0]):
    z_spec2_bin_z2 =  z_spec2[ z_spec_digitize2  == ind + 1]
    z_phot2_bin_z2 =  z_phot2[ z_spec_digitize2  == ind + 1]
    sigmaNMAD_combine[ind] =  sigmaNMAD(z_spec2_bin_z2, z_phot2_bin_z2)
    outFr_combine[ind] = outlierFrac(z_spec2_bin_z2, z_phot2_bin_z2, 0.15)


#####################


z_spec = (ymax - ymin)*(ymin + y_test)
z_phot = (ymax - ymin)*(ymin + y_pred)

z_spec_digitize = np.digitize( z_spec, bins)

# for n in range(z_spec.size):
#     print(bins[z_spec_digitize[n]-1], "<=", z_spec[n], "<", bins[z_spec_digitize[n]])


sigmaNMAD_obs = np.zeros(shape=10)
outFr_obs = np.zeros(shape=10)


for ind in range(bins.shape[0]):
    z_spec_bin_z =  z_spec[ np.where(z_spec_digitize  == ind + 1) ]
    z_phot_bin_z =  z_phot[ np.where(z_spec_digitize  == ind + 1)]
    sigmaNMAD_obs[ind] =  sigmaNMAD(z_spec_bin_z, z_phot_bin_z)
    outFr_obs[ind] = outlierFrac(z_spec_bin_z, z_phot_bin_z, 0.15)


plt.figure(5232)

bincenter = (bins[1:] + bins[:-1]) / 2.

plt.plot(bincenter, sigmaNMAD_obs[:9], 'ro--', label = 'SDSS training')
plt.plot(bincenter, sigmaNMAD_combine[:9] , 'bo--', label = 'GalaxPy training')
plt.xscale('log')
plt.legend()





plt.figure(5233)

bincenter = (bins[1:] + bins[:-1]) / 2.

plt.plot(bincenter, outFr_obs[:9], 'ro--', label = 'SDSS training')
plt.plot(bincenter, outFr_combine[:9] , 'bo--', label = 'GalaxPy training')
plt.xscale('log')
plt.title('outlier fraction')
plt.legend()




#######################################


# sigmaNMAD(z_spec_bin, z_phot_bin)
#
# outlierFrac(z_spec, z_pho):
#     len(z_pho[z_pho>=1.3])
