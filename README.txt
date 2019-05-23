export PATH=/home/nes/anaconda3/bin:$PATH

requires edward - pip install


http://edwardlib.org/tutorials/bayesian-neural-network


issue with edward -- tensorflow old version work.


conda create --name tf_old python=3.7 jupyter -y
conda activate tf_old
conda install --name tf_old tensorflow=1.2.0 --channel conda-forge -y

pip install edward


conda activate tf_gpu
