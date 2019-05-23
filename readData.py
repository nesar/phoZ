import numpy as np
import matplotlib.pylab as plt
import pandas as pd
# from mayavi import mlab
from scipy import stats


datafile = ['DES', 'COSMOS', 'Galacticus', 'Galaxev'][3]


if datafile == 'DES' :
    dirIn = '../data/'
    allfiles = ['DES.train.dat', './DES5yr.nfits.dat'][0]


    #Column 1: Spectroscopic redshift
    #Column 2: SED type
    #Columns 3-11: ugrizYJHKs magnitudes
    #Columns 12-20: ugrizYJHKs magnitude erros

    allData = np.loadtxt(dirIn + allfiles)

    specZ = allData[:, 0]
    sedType = allData[:, 1]

    color_mag = allData[:, 2:10]
    color_mag_err = allData[:, 11:]

    bb_color = ['u', 'g', 'r', 'i', 'z']


    print(color_mag.shape, color_mag_err.shape)

if datafile == 'COSMOS' :
    dirIn = '../../Data/fromJonas/'
    allfiles = ['catalog_v0.txt', 'catalog_v3.txt'][0]

    # Column 1: Spectroscopic redshift
    # Column 2: template
    # Columns 3-8: ugrizY magnitudes
    # Columns 12-20: ugrizYJHKs magnitude erros

    allData = np.loadtxt(dirIn + allfiles)

    specZ = allData[:, 0]
    sedType = allData[:, 1]

    color_mag = allData[:, 2:8]
    # color_mag_err = allData[:, 11:]

    bb_color = ['u', 'g', 'r', 'i', 'z', 'y']

    print(color_mag.shape)


if datafile == 'Galaxev':
    dirIn = '../../Data/fromGalaxev/'
    allfiles = ['photozSim2.txt'][0]

    # "metallicity" "mass" "tV" "eV" "redshift" "b_u" "b_g" "b_r" "b_i" "b_z"

    allData = np.loadtxt(dirIn + allfiles, skiprows=1)
    input_params = allData[:,0:4]
    specZ = allData[:, 4]
    color_mag = allData[:, 5:]
    bb_color = ['u', 'g', 'r', 'i', 'z']
    sedType = np.ones_like(specZ) ## All BlUE right now

#######################################################################
plt.figure(132)

for i in range(np.shape(bb_color)[0]):

    plt.plot(specZ, color_mag[:,i], 'o', label = bb_color[i], ms = 2, alpha = 0.1)


plt.xlabel('spec-z')
plt.ylabel('mag')
plt.legend()
plt.show()


plt.figure(342)
plt.scatter(specZ, color_mag[:,1], c = sedType, s = 1, alpha = 0.4)
plt.colorbar( label = 'SED type')


plt.xlabel('spec-z')
plt.ylabel('g mag')



plt.figure(333)
plt.scatter(color_mag[:,2], color_mag[:,3], c = specZ, s = 1, alpha = 0.4)
plt.colorbar( label = 'spec-z')


plt.xlabel('g')
plt.ylabel('r')

plt.show()



plt.figure(3344)
plt.scatter(color_mag[:,0], color_mag[:,1], c=specZ, cmap=plt.cm.get_cmap('jet', 6), s=3, alpha=0.8)
plt.colorbar(label='redshift')
plt.xlabel('u (mag)')
plt.ylabel('g (mag)')




plt.figure(3345)
plt.scatter(color_mag[:,0] - color_mag[:,1], color_mag[:,2] - color_mag[:,3],  c=specZ,
            cmap=plt.cm.get_cmap('jet', 6), s=3,
            alpha=0.8)
plt.colorbar(label='redshift')
plt.xlabel('u - g (mag)')
plt.ylabel('r - i (mag)')



plt.figure(3346)
plt.scatter( input_params[:,1], color_mag[:,1] - color_mag[:,2],  c=specZ,
            cmap=plt.cm.get_cmap('jet', 6), s=3,
            alpha=0.8)

plt.xscale('log')
plt.colorbar(label='redshift')
plt.ylabel('u - g (mag)')
plt.xlabel('mass')



plt.figure(3346)
plt.scatter( input_params[:,1], specZ,  c=specZ,
            cmap=plt.cm.get_cmap('jet', 6), s=3,
            alpha=0.8)
plt.colorbar(label='redshift')
plt.ylabel('u - g (mag)')
plt.xlabel('mass')


############# ############# ############# ############# #############

#############  joint probability distribution calculation using Gaussian kde #############

#############  known (u,g,z) -> rho(u',g',z') for any u', g', z'


n_val = 200
data = np.vstack( [color_mag[:n_val, 1:3].T, specZ[:n_val] ] ).T
# data = color_mag[:n_val, 0:3]
values = data.T


from scipy import stats

kde = stats.gaussian_kde(values)

density = kde(values)



# Plotting contours of rho(u,g,z) for known u,g,z

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
x, y, z = values
ax.scatter(x, y, z, c=density)
plt.show()


# Plotting contours of rho(u',g',z')


# Create a regular 3D grid with 50 points in each dimension
xmin, ymin, zmin = values.min(axis=1)
xmax, ymax, zmax = values.max(axis=1)
xi, yi, zi = np.mgrid[xmin:xmax:50j, ymin:ymax:50j, zmin:zmax:50j]

# Evaluate the KDE on a regular grid...
coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
density = kde(coords).reshape(xi.shape)

# Visualize the density estimate as isosurfaces
mlab.contour3d(xi, yi, zi, density, opacity=0.5)
mlab.axes()
mlab.show()


############# ############# ############# ############# #############

#############  joint probability distribution calculation using Gaussian kde #############

#############  known (u,z) -> rho(u', z') for any u'


band = 1


n_val = 1000
data = np.vstack( [color_mag[:n_val, band].T, specZ[:n_val] ] ).T
# data = color_mag[:n_val, 0:3]
values = data.T

# https://stackoverflow.com/questions/21918529/multivariate-kernel-density-estimation-in-python
# https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/



kde = stats.gaussian_kde(values)

density = kde(values)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
x, z = values
# ax.scatter(x, z, density)

# ax.plot_wireframe(x, z, density)
ax.plot_trisurf(x, z, density, alpha = 0.8)
plt.xlabel(bb_color[band])
plt.ylabel('specz')
plt.title('P(' + bb_color[band] + ', specz)')
plt.show()


# plt.scatter(x, z, c = density)


#
# from mayavi import mlab
#
# # Create a regular 3D grid with 50 points in each dimension
# xmin, ymin, zmin = values.min(axis=1)
# xmax, ymax, zmax = values.max(axis=1)
# xi, yi, zi = np.mgrid[xmin:xmax:50j, ymin:ymax:50j, zmin:zmax:50j]
#
# # Evaluate the KDE on a regular grid...
# coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
# density = kde(coords).reshape(xi.shape)
#
# # Visualize the density estimate as isosurfaces
# mlab.contour3d(xi, yi, zi, density, opacity=0.5)
# mlab.axes()
# mlab.show()

# mag = -2.5*np.log(Flux)
# Flux_u = 10**(-0.4101)


#########################################
#########################################


##### Full joint probability distribution #######

n_val = 10000
data = np.vstack( [color_mag[:n_val, :5].T, specZ[:n_val] ] ).T
values = data.T


kde = stats.gaussian_kde(values)
density = kde(values)



# Create a regular 3D grid with 50 points in each dimension
x0min, x1min, x2min, x3min, x4min, zmin = values.min(axis=1)
x0max, x1max, x2max, x3max, x4max, zmax = values.max(axis=1)
xi, yi, zi = np.mgrid[x1min:x1max:50j, x2min:x2max:50j, zmin:zmax:50j]

x0val = 24.9*np.ones_like(xi)
x3val = 22.7*np.ones_like(xi)
x4val = 22.3*np.ones_like(xi)

# Evaluate the KDE on a regular grid P(c1,c2, z)

coords = np.vstack([item.ravel() for item in [x0val, xi, yi, x3val, x4val, zi]])
density = kde(coords).reshape(xi.shape)

# Visualize the density estimate as isosurfaces
mlab.contour3d(xi, yi, zi, density, opacity=0.5)
mlab.axes()
mlab.show()

# Conditional probability p(z| c1, c2 = X, c3 = Y, ...)

band = 2
print('minmax', values.min(axis=1)[band], values.max(axis=1)[band])

band_value = values.mean(axis=1)[band]

# band_value = 2.5


zi = np.linspace(values.min(axis=1)[-1], values.max(axis=1)[-1], 100)


xi = np.linspace(values.min(axis=1)[band], values.max(axis=1)[band], 100)
# xi = np.linspace( 0.9*band_value, 1.1*band_value, 100)
# xi = np.linspace( band_value-0.5, band_value+0.5, 100)



allItem = np.empty(shape=(6, xi.shape[0]))

for noband in range(5):
    if (noband != band):
        x_noband = (values.mean(axis=1)[noband])*np.ones_like(xi)
        allItem[noband] = x_noband


allItem[band] = xi
allItem[-1] = zi
# allItem


# Evaluate the KDE on a regular grid P(c1,c2, z)

coords = np.vstack([item.ravel() for item in allItem])
density = kde(coords).reshape(xi.shape)

plt.plot(zi, density)

plt.xlabel('spec-z')
plt.ylabel('P(z,u`,g`,r`,i`,z`)')

# plt.title()