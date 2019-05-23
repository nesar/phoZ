from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style(style ="white")

from scipy import stats



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


num_train = 150000 #params.num_train # 512
#

num_samples = 1 # Number of realizations per template

np.random.seed(42)

datafile = ['DES', 'COSMOS', 'Galacticus'][1]


if datafile == 'COSMOS' :
  dirIn = '../../Data/fromJonas/'
  allfiles = ['catalog_v0.txt', 'catalog_v1.txt', 'catalog_v2a.txt', 'catalog_v2.txt',
              'catalog_v3.txt'][3]


  Trainfiles = np.loadtxt(dirIn + allfiles)

  TrainshuffleOrder = np.arange(Trainfiles.shape[0])
  np.random.shuffle(TrainshuffleOrder)

  Trainfiles = Trainfiles[TrainshuffleOrder]


  specz = Trainfiles[:num_train, 0]  # spec z
  templateID = Trainfiles[:num_train, 1]  # template id

  color_flux = Trainfiles[:num_train, 2:8]  # color mag
  color_flux_err = Trainfiles[:num_train, 8:14]  # color mag




  new_cat = np.zeros(shape=(num_train * num_samples, 2 + color_flux.shape[1]))

  for i in np.arange(num_train):
      for j in np.arange(num_samples):
          new_cat[i + j, :] = np.hstack(
              [specz[i], templateID[i], np.random.normal(color_flux[i], color_flux_err[i])[::]])


          # new_cat[i,:], = np.hstack( [specz[i], templateID[i], np.random.normal(color_flux[i],
      #                                                                  color_flux_err[i])[::]] )



np.savetxt(dirIn + 'catalog_v2b.txt',  new_cat)


def colorPlot(x, specz, Flux = True):
    plt.figure()
    if Flux:

        sns.regplot(np.log(x[:,0]), specz, fit_reg=False, color='red', scatter_kws={'s':8})
        # sns.regplot(np.log(x[:,1]), specz, fit_reg=False, color='blue', scatter_kws={'s':8})
        # sns.regplot(np.log(x[:,2]), specz, fit_reg=False, color='green', scatter_kws={'s':8})
        # sns.regplot(np.log(x[:,3]), specz, fit_reg=False, color='black', scatter_kws={'s':8})
        # sns.regplot(np.log(x[:,4]), specz, fit_reg=False, color='orange', scatter_kws={'s':8})
        # sns.regplot(np.log(x[:,5]), specz, fit_reg=False, color='gray', scatter_kws={'s':8})



colorPlot(x = color_flux, specz = specz, Flux= True)
colorPlot(x = new_cat[:, 2:8], specz = new_cat[:,0], Flux= True)
plt.show()
