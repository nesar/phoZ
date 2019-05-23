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


n_epoch = 20000
# N = 4000  # number of data points  -- replaced by num_trai
D = 6  # number of features  (8 for DES, 6 for COSMOS)
K = 3 # number of mixture components


learning_rate = 5e-3


num_train = 10000 # 100000 #params.num_train # 512
num_test = 1000 #10000 #params.num_test # 32
#

np.random.seed(42)

datafile = ['DES', 'COSMOS', 'Galacticus'][1]


if datafile == 'DES' :
  dirIn = '../data/'
  allfiles = ['DES.train.dat', './DES5yr.nfits.dat']

  Trainfiles = np.loadtxt(dirIn + allfiles[0])
  Testfiles = np.loadtxt(dirIn + allfiles[1])

  TrainshuffleOrder = np.arange(Trainfiles.shape[0])
  np.random.shuffle(TrainshuffleOrder)

  TestshuffleOrder = np.arange(Testfiles.shape[0])
  np.random.shuffle(TestshuffleOrder)

  Trainfiles = Trainfiles[TrainshuffleOrder]
  Testfiles = Testfiles[TestshuffleOrder]

  X_train = Trainfiles[:num_train, 2:10]  # color mag
  X_test = Testfiles[:num_test, 2:10]  # color mag

  xmax = np.max( [X_train.max(), X_test.max()] )
  xmin = np.min( [X_train.min(), X_test.min()] )

  X_train = (X_train - xmin)/(xmax - xmin)
  X_test = (X_test - xmin)/(xmax - xmin)


  y_train = Trainfiles[:num_train, 0]  # spec z
  y_test = Testfiles[:num_test, 0]  # spec z


  ymax = np.max( [y_train.max(), y_test.max()] )
  ymin = np.min( [y_train.min(), y_test.min()] )

  y_train = (y_train - ymin)/(ymax - ymin)
  y_test = (y_test - ymin)/(ymax - ymin)



if datafile == 'COSMOS' :
  dirIn = '../../Data/fromJonas/'
  allfiles = ['catalog_v0.txt', 'catalog_v1.txt', 'catalog_v2a.txt', 'catalog_v2.txt',
              'catalog_v2b.txt', 'catalog_v3.txt'][3]


  Trainfiles = np.loadtxt(dirIn + allfiles)

  TrainshuffleOrder = np.arange(Trainfiles.shape[0])
  np.random.shuffle(TrainshuffleOrder)

  Trainfiles = Trainfiles[TrainshuffleOrder]

  X_train = Trainfiles[:num_train, 2:8]  # color mag
  X_test = Trainfiles[num_train + 1: num_train + num_test, 2:8]  # color mag

  ifFlux = True
  if ifFlux:
      X_train = -2.5*np.log(X_train)
      X_test = -2.5*np.log(X_test)

  xmax = np.max( [X_train.max(), X_test.max()] )
  xmin = np.min( [X_train.min(), X_test.min()] )

  X_train = (X_train - xmin)/(xmax - xmin)
  X_test = (X_test - xmin)/(xmax - xmin)

  y_train = Trainfiles[:num_train, 0]  # spec z
  y_test = Trainfiles[num_train + 1: num_train + num_test, 0]  # spec z

  ymax = np.max( [y_train.max(), y_test.max()] )
  ymin = np.min( [y_train.min(), y_test.min()] )

  y_train = (y_train - ymin)/(ymax - ymin)
  y_test = (y_test - ymin)/(ymax - ymin)





print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))




sns.regplot(X_train[:,0], y_train, fit_reg=False, color='red', scatter_kws={'s':8})
sns.regplot(X_train[:,1], y_train, fit_reg=False, color='blue', scatter_kws={'s':8})
sns.regplot(X_train[:,2], y_train, fit_reg=False, color='green', scatter_kws={'s':8})
sns.regplot(X_train[:,3], y_train, fit_reg=False, color='black', scatter_kws={'s':8})
sns.regplot(X_train[:,4], y_train, fit_reg=False, color='orange', scatter_kws={'s':8})
sns.regplot(X_train[:,5], y_train, fit_reg=False, color='gray', scatter_kws={'s':8})

plt.show()



######################### network arch ##############################


X_ph = tf.placeholder(tf.float32, [None, D])
y_ph = tf.placeholder(tf.float32, [None])


def neural_network(X):
  """loc, scale, logits = NN(x; theta)"""
  # 2 hidden layers with 15 hidden units
  net = tf.layers.dense(X, 15, activation=tf.nn.relu)
  net = tf.layers.dense(net, 15, activation=tf.nn.relu)
  locs = tf.layers.dense(net, K, activation=None)
  scales = tf.layers.dense(net, K, activation=tf.exp)
  logits = tf.layers.dense(net, K, activation=None)
  return locs, scales, logits


locs, scales, logits = neural_network(X_ph)
cat = Categorical(logits=logits)
components = [Normal(loc=loc, scale=scale) for loc, scale
              in zip(tf.unstack(tf.transpose(locs)),
                     tf.unstack(tf.transpose(scales)))]
y = Mixture(cat=cat, components=components, value=tf.zeros_like(y_ph))
# Note: A bug exists in Mixture which prevents samples from it to have
# a shape of [None]. For now fix it using the value argument, as
# sampling is not necessary for MAP estimation anyways.

######################### inference ##############################

# There are no latent variables to infer. Thus inference is concerned
# with only training model parameters, which are baked into how we
# specify the neural networks.
inference = ed.MAP(data={y: y_ph})
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
inference.initialize(optimizer=optimizer, var_list=tf.trainable_variables())



##################################################################

sess = ed.get_session()
tf.global_variables_initializer().run()


train_loss = np.zeros(n_epoch)
test_loss = np.zeros(n_epoch)
for i in range(n_epoch):
  info_dict = inference.update(feed_dict={X_ph: X_train, y_ph: y_train})
  train_loss[i] = info_dict['loss']
  test_loss[i] = sess.run(inference.loss,
                          feed_dict={X_ph: X_test, y_ph: y_test})
  inference.print_progress(info_dict)


###################################################################
### Inference


pred_weights, pred_means, pred_std = sess.run(
    [tf.nn.softmax(logits), locs, scales], feed_dict={X_ph: X_test})



###################################################################
## Plot log likelihood or loss

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 3.5))
plt.plot(np.arange(n_epoch), -test_loss / len(X_test), label='Test')
plt.plot(np.arange(n_epoch), -train_loss / len(X_train), label='Train')
plt.legend(fontsize=20)
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Log-likelihood', fontsize=15)
plt.show()


###################################################################





obj = [93, 602, 120]
fig, axes = plt.subplots(nrows=3, ncols=1, sharex = True, figsize=(8, 7))

plot_normal_mix(pred_weights[obj][0], pred_means[obj][0],
                pred_std[obj][0], axes[0], comp=False)
axes[0].axvline(x=y_test[obj][0], color='black', alpha=0.5)
axes[0].text(0.3, 4.0, 'ID: ' +str(obj[0]), horizontalalignment='center',
             verticalalignment='center')


plot_normal_mix(pred_weights[obj][1], pred_means[obj][1],
                pred_std[obj][1], axes[1], comp=False)
axes[1].axvline(x=y_test[obj][1], color='black', alpha=0.5)
axes[1].text(0.3, 4.0, 'ID: ' +str(obj[1]), horizontalalignment='center',
             verticalalignment='center')

plot_normal_mix(pred_weights[obj][2], pred_means[obj][2],
                pred_std[obj][2], axes[2], comp=False)
axes[2].axvline(x=y_test[obj][2], color='black', alpha=0.5)
axes[2].text(0.3, 4.0, 'ID: ' +str(obj[2]), horizontalalignment='center',
             verticalalignment='center')

plt.xlabel(r' rescaled[$z_{pred}]$', fontsize = 19)

plt.show()





a = sample_from_mixture(X_test[:,1], pred_weights, pred_means,
                        pred_std, amount=len(X_test))
sns.jointplot(a[:, 0], a[:, 1], kind="hex", color="#4CB391")
plt.show()




#############################################################################3

import SetPub
SetPub.set_pub()


## Overall mean --- weight * mean

y_pred = np.sum(pred_means*pred_weights, axis = 1)
y_pred_std = np.sum(pred_std*pred_weights, axis = 1)

plt.figure(22)


# plt.scatter(y_test, y_pred, facecolors='k', s = 1)
plt.errorbar((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred), yerr= (ymax - ymin)*(
  ymin + y_pred_std), fmt='bo', ecolor='r', ms = 2, alpha = 0.05)


plt.text(0.8, 2.0, datafile, horizontalalignment='center', verticalalignment='center')
plt.plot((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_test), 'k')
plt.ylabel(r'$z_{pred}$', fontsize = 19)
plt.xlabel(r'$z_{true}$', fontsize = 19)
plt.tight_layout()
plt.show()

#############################################################################3
##  mean --- highest weight



weight_max = np.argmax(pred_weights, axis = 1)
from keras.utils import np_utils
weight_max = np_utils.to_categorical(weight_max)


y_pred = np.max(pred_weights*weight_max*pred_means, axis=1)
y_pred_std = np.max(pred_weights*weight_max*pred_std, axis = 1)

# y_pred = pred_weights[weight_max]*pred_means[weight_max]
# y_pred_std = pred_weights[weight_max]*pred_std[weight_max]

plt.figure(24)


# plt.scatter(y_test, y_pred, facecolors='k', s = 1)
plt.errorbar((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred), yerr= (ymax - ymin)*(
  ymin + y_pred_std), fmt='bo', ecolor='r', ms = 2, alpha = 0.05)


plt.text(0.8, 2.0, datafile, horizontalalignment='center', verticalalignment='center')
plt.plot((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_test), 'k')
plt.ylabel(r'$z_{pred}$', fontsize = 19)
plt.xlabel(r'$z_{true}$', fontsize = 19)
plt.tight_layout()
plt.show()