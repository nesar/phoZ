import keras
import numpy as np
import matplotlib.pylab as plt

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, merge, Reshape
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Convolution2D
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import *


## For DES

batch_size = 1000
epochs = 500
learning_rate = 1e-2
decay_rate = 0.0
D = 8  # number of features  (8 for DES, 6 for COSMOS)

num_train = 20000 #params.num_train # 512
num_test = 10000 #params.num_test # 32

## For COSMOS

batch_size = 10000
epochs = 10000
learning_rate = 2e-3
decay_rate = 0.001
D = 6  # number of features  (8 for DES, 6 for COSMOS)

num_train = 100000 #params.num_train # 512
num_test = 10000 #params.num_test # 32



batch_size = 10000
epochs = 100000
learning_rate = 1e-2
decay_rate = 0.001
D = 6  # number of features  (8 for DES, 6 for COSMOS)

num_train = 100000 #params.num_train # 512
num_test = 10000 #params.num_test # 32


fileOut =  'tot' + str(num_train) + '_batch' + str(batch_size) + '_lr' + str(
    learning_rate) + '_decay' + str(decay_rate) + '_ '+ '_epoch' + str(epochs)

modelfile = './Model/saved_model' + fileOut + '.h5'



############## i/o ###########################################


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
              'catalog_v2b.txt',
              'catalog_v3.txt'][4]


  Trainfiles = np.loadtxt(dirIn + allfiles)


  Trainfiles = Trainfiles[Trainfiles.min(axis=1) >= 0, :] ## Remove rows with any negative values

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


######################### network arch ##############################


print('Building model...')
model = Sequential()
model.add(Dense(64, input_shape=(D,)))
model.add(Activation('relu'))
model.add(Dropout(0.01))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# # add a dense layer with 10 nodes
# model = Sequential()
# model.add(Dense(100, input_dim=D))
# # activation sigmoid
# model.add(Activation('linear'))
# # layer to output
# model.add(Dense(30, activation = 'linear'))
# model.add(Dense(1, activation = 'sigmoid'))
# print("Model defined...")

# model = Sequential()
# model.add(Dense(4, input_shape=(D,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.01))
# model.add(Dense(2))
# model.add(Activation('relu'))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))


adam = Adam(lr = learning_rate, decay = decay_rate)

model.compile(loss='mean_squared_error', optimizer=adam)


ModelFit = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2)
score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)

model.save(modelfile)

print('---------------------------')


plotLoss = True
if plotLoss:
    import matplotlib.pylab as plt

    train_loss= ModelFit.history['loss']
    val_loss= ModelFit.history['val_loss']
    epoch_array = range(1, epochs+1)


    fig, ax = plt.subplots(1,1, sharex= True, figsize = (7,5))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax.plot(epoch_array,train_loss)
    ax.plot(epoch_array,val_loss)
    ax.set_ylabel('loss')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax.legend(['train_loss','val_loss'])

    plt.show()

#--------------Testing----------------------------


import SetPub
SetPub.set_pub()

model = keras.models.load_model(modelfile)

y_pred = model.predict(X_test)

plt.figure(22)
# plt.scatter((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred), c=X_test[:,4] , s = 1)
plt.errorbar((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_pred), fmt='o', ms = 1,
             alpha = 0.2)

plt.text(0.5, 2.5, datafile, horizontalalignment='center', verticalalignment='center')
plt.plot((ymax - ymin)*(ymin + y_test), (ymax - ymin)*(ymin + y_test), 'k')
plt.ylabel(r'$z_{pred}$', fontsize = 19)
plt.xlabel(r'$z_{true}$', fontsize = 19)
plt.tight_layout()
plt.show()

