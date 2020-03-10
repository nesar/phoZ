import matplotlib as mpl
mpl.use('Agg')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pylab as plt
np.random.seed(10)

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
tfd = tfp.distributions

# Activate TF2 behavior:
from tensorflow.python import tf2
if not tf2.enabled():
  import tensorflow.compat.v2 as tf
  tf.enable_v2_behavior()
  assert tf2.enabled()


# %%
def ReadGalaxPy(path_program = '../../Data/fromGalaxev/photozs/datasets/', sim_obs_combine = True):    
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

        # X_train = Trainfiles[:num_train]  # color mag
        # X_test = Trainfiles[:num_test]  # color mag

        # y_train = TrainZ[:num_train]  # spec z
        # y_test = TrainZ[:num_test]  # spec z

        #### THIS PART WAS WRONG above --- incorrect test and train -- BOTH WERE SAME ####

        X_train = Trainfiles[:num_train]  # color mag
        X_test = Testfiles[:num_test]  # color mag

        y_train = TrainZ[:num_train]  # spec z
        y_test = TestZ[:num_test]  # spec z

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


# %%
# def ReadGalaxPy(path_program = '../../Data/fromGalaxev/photozs/datasets/', sim_obs_combine = True):    
#     class Curated_sample():
#         ''' Class to store the redshift and colors of observed galaxies,
#             and the redshift, Mpeak, colors, and "weights" of simulated
#             galaxies whose colors are compatible with those of observed
#             galaxies.

#             The observed sample include galaxies from SDSS
#             (SDSS+BOSS+eBOSS), DEEP2, and VIPERS.

#             The simulated sample was created by sampling the parameter of
#             GALAXPY using a LH.

#             The weights of simulated galaxies are related to the number
#             density of observed galaxies in the same region of the color
#             space.

#             You only have to care about the method load_structure. '''

#         def __init__(self):
#             self.arr_c = []
#             self.arr_z = []
#             self.arr_m = []
#             self.arr_w = []

#         def append(self, c, z, m, w):
#             self.arr_c.append(c)
#             self.arr_z.append(z)
#             self.arr_m.append(m)
#             self.arr_w.append(w)

#         def ndarray(self):
#             self.arr_c = np.concatenate(self.arr_c)
#             self.arr_z = np.concatenate(self.arr_z)
#             self.arr_m = np.concatenate(self.arr_m)
#             self.arr_w = np.concatenate(self.arr_w)

#         def save_struct(self, name):
#             np.save(name + 'c.npy', self.arr_c)
#             np.save(name + 'z.npy', self.arr_z)
#             np.save(name + 'm.npy', self.arr_m)
#             np.save(name + 'w.npy', self.arr_w)

#         def load_struct(self, name):
#             self.arr_c = np.load(name + 'c.npy')
#             self.arr_z = np.load(name + 'z.npy')
#             self.arr_m = np.load(name + 'm.npy')
#             self.arr_w = np.load(name + 'w.npy')

#         def duplicate_data(self, zrange):
#             aa = np.where((self.arr_w > 50)
#                           & (self.arr_z >= zrange[0])
#                           & (self.arr_z < zrange[1]))[0]
#             print(aa.shape)
#             cc = np.repeat(aa, self.arr_w[aa].astype(int))
#             self.arr_cn = self.arr_c[cc, :]
#             self.arr_zn = self.arr_z[cc]
#             self.arr_mn = self.arr_m[cc]


#     def read_curated_data():
#         run_path = path_program + 'runs/run_z3/'

#         sim_q = Curated_sample()  # simulated colors quenched galaxies
#         sim_s = Curated_sample()  # simulated colors star-forming galaxies
#         obs_q = Curated_sample()  # observed colors quenched galaxies
#         obs_s = Curated_sample()  # observed colors star-forming galaxies

#         obs_q.load_struct(run_path + 'str_obs_q')
#         obs_s.load_struct(run_path + 'str_obs_s')
#         sim_q.load_struct(run_path + 'str_sim_q')
#         sim_s.load_struct(run_path + 'str_sim_s')

#         print(sim_q.arr_c.shape)
#         print(sim_s.arr_c.shape)
#         print(obs_q.arr_c.shape)
#         print(obs_s.arr_c.shape)

#         return sim_q, sim_s, obs_q, obs_s


#     sim_q, sim_s, obs_q, obs_s = read_curated_data()

#     if sim_obs_combine:
#         train_datafile = 'GalaxPy'

#         # 2.0 ####### TRAIN USING SIMULATION, TEST OBSERVATION ####

#         Trainfiles = np.append(sim_q.arr_c, sim_s.arr_c, axis=0)
#         TrainZ = np.append(sim_q.arr_z, sim_s.arr_z, axis=0)

#         Trainfiles = np.delete(Trainfiles, (4), axis=1)  ## deleting z-Y

#         Testfiles = np.append(obs_q.arr_c, obs_s.arr_c, axis=0)
#         TestZ = np.append(obs_q.arr_z, obs_s.arr_z, axis=0)

#         np.argsort()

#         TrainshuffleOrder = np.arange(Trainfiles.shape[0])
#         np.random.shuffle(TrainshuffleOrder)

#         Trainfiles = Trainfiles[TrainshuffleOrder]
#         TrainZ = TrainZ[TrainshuffleOrder]

#         TestshuffleOrder = np.arange(Testfiles.shape[0])
#         np.random.shuffle(TestshuffleOrder)

#         Testfiles = Testfiles[TestshuffleOrder]
#         TestZ = TestZ[TestshuffleOrder]
        

#         # X_train = Trainfiles[:num_train]  # color mag
#         # X_test = Trainfiles[:num_test]  # color mag

#         # y_train = TrainZ[:num_train]  # spec z
#         # y_test = TrainZ[:num_test]  # spec z

#         #### THIS PART WAS WRONG above --- incorrect test and train -- BOTH WERE SAME ####

#         X_train = Trainfiles[:num_train]  # color mag
#         X_test = Testfiles[:num_test]  # color mag

#         y_train = TrainZ[:num_train]  # spec z
#         y_test = TestZ[:num_test]  # spec z

#     else:
#         train_datafile = 'SDSS'
#         # 1.1 ####### SIMULATED: QUENCHED ONLY ############
#         # Trainfiles = sim_q.arr_c
#         # TrainZ = sim_q.arr_z

#         # 1.2 ### SIMULATED: QUENCHED + STAR FORMATION ####

#         # Trainfiles =np.append( sim_q.arr_c, sim_s.arr_c, axis = 0)
#         # TrainZ = np.append( sim_q.arr_z, sim_s.arr_z, axis = 0)

#         # 1.3 ####### OBSERVED: QUENCHED + STAR FORMATION ####

#         Trainfiles = np.append(obs_q.arr_c, obs_s.arr_c, axis=0)
#         TrainZ = np.append(obs_q.arr_z, obs_s.arr_z, axis=0)

#         TrainshuffleOrder = np.arange(Trainfiles.shape[0])
#         np.random.shuffle(TrainshuffleOrder)

#         Trainfiles = Trainfiles[TrainshuffleOrder]
#         TrainZ = TrainZ[TrainshuffleOrder]

#         # 1 #################################

#         X_train = Trainfiles[:num_train]  # color mag
#         X_test = Trainfiles[num_train + 1: num_train + num_test]  # color mag

#         X_train = Trainfiles[:num_train]  # color mag
#         X_test = Trainfiles[num_train + 1: num_train + num_test]  # color mag

#         y_train = TrainZ[:num_train]  # spec z
#         y_test = TrainZ[num_train + 1: num_train + num_test]  # spec z

#     ############## THINGS ARE SAME AFTER THIS ###########

#     ## rescaling xmax/xmin
#     xmax = np.max([np.max(X_train, axis=0), np.max(X_test, axis=0)], axis=0)
#     xmin = np.min([np.min(X_train, axis=0), np.min(X_test, axis=0)], axis=0)

#     X_train = (X_train - xmin) / (xmax - xmin)
#     X_test = (X_test - xmin) / (xmax - xmin)

#     #### RESCALING X_train, X_test NOT done yet -- (g-i), (r-i) ... and i mag -->> Color/Mag issue

#     ymax = np.max([y_train.max(), y_test.max()])
#     ymin = np.min([y_train.min(), y_test.min()])

#     y_train = (y_train - ymin) / (ymax - ymin)
#     y_test = (y_test - ymin) / (ymax - ymin)

#     return X_train, y_train, X_test, y_test, ymax, ymin, xmax, xmin


# %%
n_epochs = 1000 
D = 5 #6  # number of features  (8 for DES, 6 for COSMOS)
K = 32 #16 # number of mixture components

learning_rate = 1e-4
decay_rate= 0.0 
step=1000 #100
batch_size = 1024 #8192


num_train = 12000000 #800000
num_test = 10000 #5000 #params.num_test # 32

syntheticTrain = True # (sim_obs_combine) True -- train using GalaxyPy, False -- train using

save_mod = 'saved_hubs/new_tf2/'+'Synthetic_'+str(syntheticTrain)+'_lr_'+str(learning_rate)+'_dr'+str(decay_rate)+'_step'+str(step)+'_ne'+str(n_epochs)+'_k'+str(K)+'_nt'+str(num_train)


# %%
############training

# X_train, y_train, X_test, y_test, ymax, ymin, xmax, xmin = ReadGalaxPy(path_program = '../../Data/fromGalaxev/photozs/datasets/', sim_obs_combine = syntheticTrain)
X_train, y_train, X_test, y_test, ymax, ymin, xmax, xmin = ReadGalaxPy(path_program = '../../Data/fromGalaxev/photozs/datasets/', sim_obs_combine = syntheticTrain)


print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))


# %%
test_argsort = np.argsort(y_test)

y_test = y_test[test_argsort]
X_test = X_test[test_argsort]


# %%
num_y_test_select = 1000
exp_dist = np.random.exponential(scale=0.01, size=num_test)
# exp_dist = np.random.geometric(p=0.1, size=num_test)
trunc_dist = num_test*(1-exp_dist/(exp_dist.max() - exp_dist.min()) )
int_dist = trunc_dist.astype('int')
int_select = int_dist[:num_y_test_select]


plt.figure(23)
plt.hist(y_test, normed=True, histtype='step', label='original')


y_test1 = y_test[int_select]
X_test1 = X_test[int_select]


plt.hist(y_test1, normed=True, histtype='step', label='resampled')
plt.legend()
plt.show()


# %%
# int_select 


# %%



# %%
output_shape = 1 

model = keras.Sequential([
    keras.layers.Dense(units=128, activation='relu', input_shape=(D,)),
    keras.layers.Dense(units=128, activation='tanh'),
    keras.layers.Dense(units=64, activation='tanh'),
    keras.layers.Dense(units=32, activation='tanh'),
    keras.layers.Dense(tfp.layers.MixtureNormal.params_size(K, output_shape)),
    tfp.layers.MixtureNormal(K, output_shape)])

# model = keras.Sequential([
#     keras.layers.Dense(units=32, activation='relu', input_shape=(D,)),
#     keras.layers.Dense(units=16, activation='tanh'),
#     keras.layers.Dense(units=8, activation='tanh'),
#     keras.layers.Dense(units=4, activation='tanh'),
#     keras.layers.Dense(tfp.layers.MixtureNormal.params_size(K, output_shape)),
#     tfp.layers.MixtureNormal(K, output_shape)])



# model = keras.Sequential([
#     keras.layers.Dense(units=32, activation='relu', input_shape=(D,)),
#     keras.layers.Dense(units=16, activation='tanh'),
#     keras.layers.Dense(units=8, activation='tanh'),
#     # keras.layers.Dense(tfp.layers.MixtureNormal.params_size(K, output_shape)),
#     # tfp.layers.MixtureNormal(K, output_shape)
#     keras.layers.Dense(units=K*3),
#     tfp.layers.DistributionLambda(lambda t: tfd.Independent(
#         tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(
#                 logits=tf.expand_dims(t[..., :K], -2)),
#                 components_distribution=tfd.Beta(1 + tf.nn.softplus(tf.expand_dims(t[..., K:2*K], -2)), 
#                 1 + tf.nn.softplus(tf.expand_dims(t[..., 2*K:],-2)))), 1))])


def negloglik(y_true, y_pred):
    return -y_pred.log_prob(y_true)


callback = tf.keras.callbacks.LearningRateScheduler(lambda e: 0.001 if e < 5 else 0.0001)

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1= decay_rate, amsgrad = True)
model.compile(loss=negloglik, optimizer=opt, metrics=[])

model.summary()


# %%
history = model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=n_epochs, batch_size=batch_size, callbacks=[callback], verbose=2)


# %%
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.xlabel('Epochs', fontsize = 28)
plt.ylabel('Loss', fontsize = 28)


# %%
model.save(save_mod + '.h5')


# %%
y_test = y_test1
X_test = X_test1


y_pred = model(X_test)
y_pred_mean = y_pred.components_distribution.mean() 
y_pred_std = y_pred.components_distribution.stddev()
y_pred_mode = y_pred.components_distribution.mode() 


# %%
# y_pred_mean[:, :, 0].shape


# %%
# y_pred.log_prob(1).shape


# %%
# y_pred.components_distribution.mode()


# %%
# y_pred.components_distribution.distribution.mean()


# %%
def plot_normal_mix(pis, mus, sigmas, ax, label='', color = '', comp=True):
  """Plots the mixture of Normal models to axis=ax comp=True plots all
  components of mixture model
  """

  x = np.linspace(-0.1, 1.1, 250)
  final = np.zeros_like(x)
  for i, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
    temp = stats.norm.pdf(x, mu_mix, sigma_mix) * weight_mix
    final = final + temp
    if comp:
#       ax.plot(x, temp, label='Normal ' + str(i), alpha =0.6)
      ax.plot(x, temp, 'k--', alpha =0.9)

#       ax.plot(x, temp/final.max(), alpha =0.5)

  ax.plot(x, final,label=label, color = color)
#   ax.plot(x, final/final.max(), label=label, color = color)

    
  ax.legend(fontsize=13)
  return final

def plot_pdfs(pred_means,pred_weights,pred_std, y,num=4, label = '', color = '', train=False, comp = False):
    np.random.seed(12)

    if train:
        obj = np.random.randint(0,num_train-1,num)
    else:
        obj = np.random.randint(0,num_test-1,num)
#     obj = [462, 667, 81]
#     obj = [462, 102, 81]
    obj = [4, 18, 81]
    
    print(obj)

    allfs = []
    for i in range(len(obj)):
        print(i)
        if (i==0):
            fs = plot_normal_mix(pred_weights[obj][i], pred_means[obj][i], pred_std[obj][i], axes[i], label = label, color = color, comp=comp)
        else: fs = plot_normal_mix(pred_weights[obj][i], pred_means[obj][i], pred_std[obj][i], axes[i], label = '', color = color, comp=comp)

        axes[i].set_ylabel(r'${\rm PDF}$', fontsize = 22)
        allfs.append(fs)
        axes[i].axvline(x=y[obj][i], color='black', alpha=0.5)
        

    plt.xlabel('Photometric redshift', fontsize = 26)
    


# %%
test_means = np.array(y_pred_mean[:, :, 0] )
test_std = np.array(y_pred_std[:, :, 0] )
test_weights = np.ones_like(test_means)

## plotting pdfs
from scipy import stats

nrows = 3
fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex = True, figsize=(12, nrows*4), num='pdfs')
plot_pdfs(test_means,test_weights,test_std, y_test, num=nrows, label = 'Training with synthetic data', color = 'red', train=False)
# plot_pdfs(test_means_2,test_weights_2,test_std_2, y_test, num=nrows, label = 'Training with observational data', color = 'blue', train=False)


# %%
test_means = np.array(y_pred_mean[:, :, 0] )
test_std = np.array(y_pred_std[:, :, 0] )
test_weights = np.ones_like(test_means)

## plotting pdfs




from scipy import stats

nrows = 3
fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex = True, figsize=(12, nrows*4), num='pdfs')
plot_pdfs(test_means,test_weights,test_std, y_test, num=nrows, label = 'Training with synthetic data', color = 'red', train=False)
# plot_pdfs(test_means_2,test_weights_2,test_std_2, y_test, num=nrows, label = 'Training with observational data', color = 'blue', train=False)


# %%
# y_pred_mean_best


# %%
# fig = plt.figure(22)

ifPlotWeighted = True
# y_pred_mean_best = y_pred_mean[:, 0]
y_pred_mean_best = y_pred_mode[:, 0]
y_pred_std_best = y_pred_std[:, 0]


if ifPlotWeighted:
    plt.figure(22, figsize=(10, 10,))
    plt.scatter((ymax - ymin)*(ymin + y_test),(ymax - ymin)*(ymin + y_pred_mean_best), facecolors='r', s = 1)


plt.xlim(0, 1)
plt.ylim(0, 1)

plt.plot([0, 1], [0, 1], 'k')
plt.plot([0, 1], 0.85*np.array([0, 1]), 'k-.')
plt.plot([0, 1], 1.15*np.array([0, 1]), 'k-.')

# plt.ylabel(r'$z_{\rm phot}$', fontsize=30)
# plt.xlabel(r'$z_{\rm spec}$', fontsize=30)


plt.ylabel(r'Photometric redshift', fontsize=25)
plt.xlabel(r'True redshift', fontsize=25)
plt.xlim(0.0, 1)
plt.ylim(0.0, 1)

# plt.legend(fontsize = 'large', markerscale=3., numpoints=3)
# plt.title('weight x mean')
plt.tight_layout()

plt.axes().set_aspect('equal')

# plt.xscale('log')
# plt.yscale('log')

leg = plt.legend(fontsize = 'xx-large', markerscale=1., numpoints=2)

for artist, text in zip(leg.legendHandles, leg.get_texts()):
    col = artist.get_color()
    if isinstance(col, np.ndarray):
        col = col[0]
    text.set_color(col)
    text.set_alpha(1.0)


plt.savefig('phoz_compare.pdf', bbox_inches='tight')

plt.show()


# %%
# fig = plt.figure(22)

ifPlotWeighted = True
# y_pred_mean_best = y_pred_mean[:, 0]
y_pred_mean_best = y_pred_mode[:, 0]
y_pred_std_best = y_pred_std[:, 0]

## Overall mean --- weight * mean
ifPlotWeighted = True

if ifPlotWeighted:
    plt.figure(22, figsize=(10, 10))

#     y_pred = np.sum(pred_means_new*pred_weights_new, axis = 1)
#     y_pred_std_new = np.sum(pred_std_new*pred_weights_new, axis = 1)

    # # plt.scatter(y_test, y_pred, facecolors='k', s = 1)
#     plt.errorbar((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred), yerr= (ymax - ymin)*(
#       ymin + y_pred_std_new), fmt='bo', ecolor='r', ms = 2, alpha = 0.1)
    
    plt.errorbar((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred_mean_best), yerr= (ymax - ymin)*(
      ymin + y_pred_std_best ), fmt='ro', ecolor='r', ms = 5, alpha = 0.3, label = 'Training with synthetic data')

#     plt.errorbar((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred), yerr= (ymax - ymin)*(
#       ymin + y_pred_std), fmt='wo', ecolor='w', ms = 5, alpha = 0.8, label = 'Training with synthetic data')



    # plt.text(0.8, 2.0, datafile, horizontalalignment='center', verticalalignment='center')
    # plt.ylabel(r'$z_{pred}$', fontsize = 19)
    # plt.xlabel(r'$z_{true}$', fontsize = 19)

    # plt.title('weight x mean')
    # plt.tight_layout()
    # plt.show()



## Overall mean --- weight * mean
# ifPlotWeighted = True


# if ifPlotWeighted:

# #     y_pred_new2 = np.sum(pred_means_new2*pred_weights_new2, axis = 1)
# #     y_pred_std_new2 = np.sum(pred_std_new2*pred_weights_new2, axis = 1)

# #     plt.figure(22, figsize=(6,6))


#     # plt.scatter(y_test2, y_pred, facecolors='k', s = 1)
# #     plt.errorbar((ymax2 - ymin2)*(ymin2 + y_test2), (ymax2 - ymin2)*(ymin2 + y_pred), yerr= (ymax2 - ymin2)*(
# #       ymin2 + y_pred_std_new2), fmt='bo', ecolor='r', ms = 2, alpha = 0.1)
    
#     plt.errorbar((ymax2 - ymin2)*(ymin2 + y_test2), (ymax2 - ymin2)*(ymin2 + y_pred_2), yerr= (ymax2 - ymin2)*(
#       ymin2 + y_pred_std_2), fmt='bo', ecolor='b', ms = 5, alpha = 0.3, label = 'Training with observational data')

# #     plt.errorbar((ymax2 - ymin2)*(ymin2 + y_test2), (ymax2 - ymin2)*(ymin2 + y_pred_2), yerr= (ymax2 - ymin2)*(
# #       ymin2 + y_pred_std_2), fmt='co', ecolor='c', ms = 5, alpha = 0.8, label = 'Training with observational data')



#     # plt.text(0.8, 2.0, datafile, horizontalalignment='center', verticalalignment='center')

plt.xlim(0, 1)
plt.ylim(0, 1)

# # plt.text(0.1, 0.9, r'$\sigma_{NMAD}$ = %.3f'%sigmaNMAD_obs_all, color = 'red' , size = 20)
# # plt.text(0.1, 0.85, r'$\sigma_{NMAD}$ = %.3f'%sigmaNMAD_combine_all, color = 'blue' , size = 20)
# plt.text(0.6, 0.2, r'$\sigma_{NMAD}$ = %.3f'%sigmaNMAD_obs_all, color = 'blue' , size = 20)
# plt.text(0.6, 0.1, r'$\sigma_{NMAD}$ = %.3f'%sigmaNMAD_combine_all, color = 'red' , size = 20)

# plt.text(0.1, 0.9, r'GalaxPy training', color = 'red' , size = 20)
# plt.text(0.1, 0.85, r'SDSS training', color = 'blue' , size = 20)

plt.plot([0, 1], [0, 1], 'k')
plt.plot([0, 1], 0.85*np.array([0, 1]), 'k-.')
plt.plot([0, 1], 1.15*np.array([0, 1]), 'k-.')

# plt.ylabel(r'$z_{\rm phot}$', fontsize=30)
# plt.xlabel(r'$z_{\rm spec}$', fontsize=30)


plt.ylabel(r'Photometric redshift', fontsize=25)
plt.xlabel(r'True redshift', fontsize=25)
plt.xlim(0.0, 1)
plt.ylim(0.0, 1)

# plt.legend(fontsize = 'large', markerscale=3., numpoints=3)
# plt.title('weight x mean')
plt.tight_layout()

plt.axes().set_aspect('equal')

# plt.xscale('log')
# plt.yscale('log')

leg = plt.legend(fontsize = 'xx-large', markerscale=1., numpoints=2)

for artist, text in zip(leg.legendHandles, leg.get_texts()):
    col = artist.get_color()
    if isinstance(col, np.ndarray):
        col = col[0]
    text.set_color(col)
    text.set_alpha(1.0)


plt.savefig('phoz_compare.pdf', bbox_inches='tight')

plt.show()


# %%
import scipy.stats
# hist = np.histogram(scaler.inverse_transform(Y_train), 64)
nbins = 1000

hist = np.histogram(y_train, nbins)
prior = scipy.stats.rv_histogram(hist)


# %%
# plt.hist(scaler.inverse_transform(Y_train), 100, normed=True);
plt.hist(y_train, nbins, normed=True)
x = np.linspace(0, 1, nbins)
plt.plot(x, prior.pdf(x))
plt.yscale('log')
plt.xlabel(r'$z$')
plt.show()


# %%
# This returns the distribution q(M | x) for all clusters
outputs = model(X_test)
# xt = scaler.transform(x.reshape((-1,1)))
xt = x
logps = []

for i in range(len(x)):
    logps.append(outputs.log_prob(xt[i]).numpy())
logps = np.stack(logps)


# %%
for i in range(10):
  plt.figure()
  plt.plot(x, np.exp(logps[:,-i]), label='posterior under training prior')
  plt.plot(x, np.exp(logps[:,-i])/prior.pdf(x), label='posterior under flat prior')
  plt.axvline(y_test[-i], color='k', linestyle = 'dashed', label='True value')
  plt.xscale('log')
  plt.xlabel(r'$z$')
  plt.legend()


# %%
# fig = plt.figure(22)

ifPlotWeighted = True
# y_pred_mean_best = y_pred_mean[:, 0]
y_pred_mean_best = y_pred_mode[:, 0]
y_pred_std_best = y_pred_std[:, 0]


if ifPlotWeighted:
    plt.figure(22, figsize=(10, 10,))


    # # plt.scatter(y_test, y_pred, facecolors='k', s = 1)
#     plt.errorbar((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred), yerr= (ymax - ymin)*(
#       ymin + y_pred_std_new), fmt='bo', ecolor='r', ms = 2, alpha = 0.1)
    
    plt.errorbar((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred_mean_best), yerr= (ymax - ymin)*(
      ymin + y_pred_std_best), fmt='ro', ecolor='r', ms = 5, alpha = 0.3, label = 'Training with synthetic data')

#     plt.errorbar((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred), yerr= (ymax - ymin)*(
#       ymin + y_pred_std), fmt='wo', ecolor='w', ms = 5, alpha = 0.8, label = 'Training with synthetic data')





plt.xlim(0, 1)
plt.ylim(0, 1)



plt.plot([0, 1], [0, 1], 'k')
plt.plot([0, 1], 0.85*np.array([0, 1]), 'k-.')
plt.plot([0, 1], 1.15*np.array([0, 1]), 'k-.')




plt.ylabel(r'Photometric redshift', fontsize=25)
plt.xlabel(r'True redshift', fontsize=25)
plt.xlim(0.0, 1)
plt.ylim(0.0, 1)


plt.tight_layout()

plt.axes().set_aspect('equal')


leg = plt.legend(fontsize = 'xx-large', markerscale=1., numpoints=2)

for artist, text in zip(leg.legendHandles, leg.get_texts()):
    col = artist.get_color()
    if isinstance(col, np.ndarray):
        col = col[0]
    text.set_color(col)
    text.set_alpha(1.0)


plt.savefig('phoz_compare.pdf', bbox_inches='tight')

plt.show()


# %%
from scipy.integrate import simps
corrected_posterior = np.exp(logps)/(prior.pdf(x).reshape((-1,1)))
# corrected_posterior = np.exp(logps)/(prior.pdf(x))
y_pred_prior_mean = simps(x.reshape((-1,1))*corrected_posterior, x,axis=0)/simps(corrected_posterior,x,axis=0 )


# %%
simps(corrected_posterior,x, axis=0)


# %%
plt.figure(22, figsize=(10, 10,))


# plt.errorbar((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred_prior_mean), yerr= (ymax - ymin)*(ymin + y_pred_std_best), fmt='ro', ecolor='r', ms = 5, alpha = 0.3, label = 'Training with synthetic data')

plt.scatter((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred_prior_mean), s = 10, alpha = 0.9, label = 'corrected posterior')


plt.xlim(0, 1)
plt.ylim(0, 1)

plt.plot([0, 1], [0, 1], 'k')
plt.plot([0, 1], 0.85*np.array([0, 1]), 'k-.')
plt.plot([0, 1], 1.15*np.array([0, 1]), 'k-.')

plt.ylabel(r'Photometric redshift', fontsize=25)
plt.xlabel(r'True redshift', fontsize=25)
plt.xlim(0.0, 1)
plt.ylim(0.0, 1)

plt.tight_layout()

plt.axes().set_aspect('equal')


leg = plt.legend(fontsize = 'xx-large', markerscale=1., numpoints=2)

# for artist, text in zip(leg.legendHandles, leg.get_texts()):
#     col = artist.get_color()
#     if isinstance(col, np.ndarray):
#         col = col[0]
#     text.set_color(col)
#     text.set_alpha(1.0)


plt.savefig('phoz_compare1.pdf', bbox_inches='tight')

plt.show()


# # %%
# y_test, 
#                 y_pred_prior_mean


# %%
def binned_plot(X, Y, n=10, percentiles=[35, 50], ax=None, **kwargs):
    # Calculation
    calc_percent = []
    for p in percentiles:
        if p < 50:
            calc_percent.append(50-p)
            calc_percent.append(50+p)
        elif p == 50:
            calc_percent.append(50)
        else:
            raise Exception('Percentile > 50')

    bin_edges = np.linspace(X.min()*0.9999, X.max()*1.0001, n+1)

    dtype = [(str(i), 'f') for i in calc_percent]
    bin_data = np.zeros(shape=(n,), dtype=dtype)

    for i in range(n):
        y = Y[(X >= bin_edges[i]) & (X < bin_edges[i+1])]

        if len(y) == 0:
            continue

        y_p = np.percentile(y, calc_percent)

        bin_data[i] = tuple(y_p)

    # Plotting
    if ax is None:
        f, ax = plt.subplots()

    bin_centers = [np.mean(bin_edges[i:i+2]) for i in range(n)]
    for p in percentiles:
        if p == 50:
            ax.plot(bin_centers, bin_data[str(p)], **kwargs)
        else:
            ax.fill_between(bin_centers,
                            bin_data[str(50-p)],
                            bin_data[str(50+p)],
                            alpha=0.2,
                            **kwargs)

    return bin_data, bin_edges


# %%
# replace y_true by y_test

import matplotlib as mpl
f = plt.figure(figsize=(6,8))
gs = mpl.gridspec.GridSpec(2,1,height_ratios=[3,1], hspace=0)

ax1 = f.add_subplot(gs[0,0])

ax1.plot(np.arange(13,16,0.1),np.arange(13,16,0.1),'k--')
_ = binned_plot(y_test, 
                y_pred_prior_mean, 
                n=20, percentiles=[35,45,50], 
                color='b', ax=ax1)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xticks([])
ax1.set_yticks(ax1.get_yticks()[1:])
ax1.set_ylabel(r'$\log_{10}\left[M_\mathrm{pred}\ (M_\odot h^{-1})\right]$', fontsize=14)


ax2 = f.add_subplot(gs[1,0])

# ax2.plot(np.arange(13.,16,0.1),[0]*30,'k--')
_ = binned_plot(y_test, 
                y_pred_prior_mean - y_test, 
                n=20, percentiles=[35,45,50], 
                color='b', ax=ax2)

ax2.set_xlim(0, 1)
ax2.set_ylim(-0.5,0.5)
ax2.set_xlabel(r'$\log_{10}\left[M_\mathrm{true}\ (M_\odot h^{-1})\right]$', fontsize=14)
ax2.set_ylabel(r'$\epsilon$', fontsize=14)


# %%


