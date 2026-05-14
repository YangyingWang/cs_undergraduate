from keras.layers import Conv2D, Dense, Flatten, LeakyReLU, Reshape, Conv2DTranspose, BatchNormalization, Input, Dropout
from keras.models import Sequential, Model, load_model
from keras.optimizers import adam_v2
from keras.utils import plot_model
import numpy as np
import tensorflow as tf

def uniform_sampling(n_sample, dim):
     return np.random.uniform(0, 1, size=(n_sample, dim))
def normal_sampling(n_sample, dim):
       return np.random.randn(n_sample, dim)
d_model = Sequential()
d_model.add(BatchNormalization())
d_model.add(Dropout(0.3))
d_model.add(Conv2D(64, (3, 3), padding='same', input_shape=(28, 28, 1)))
d_model.add(LeakyReLU(0.2))
d_model.add(Dropout(0.3))
d_model.add(Conv2D(128, (3, 3) , padding='same'))  
d_model.add(LeakyReLU(0.2))
d_model.add(Dropout(0.3))
d_model.add(Conv2D(256, (3, 3), padding='same'))
d_model.add(LeakyReLU(0.2))
d_model.add(Dropout(0.3))
d_model.add(Conv2D(512, (3, 3), padding='same'))
d_model.add(LeakyReLU(0.2))
d_model.add(Flatten())
d_model.add(Dropout(0.3))
d_model.add(Dense(1, activation='sigmoid'))

g_model = Sequential()
g_model.add(BatchNormalization())
g_model.add(Dense(7 * 7 * 256, activation='relu', input_dim=100))
g_model.add(Reshape((7, 7, 256)))
g_model.add(BatchNormalization())
g_model.add(Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'))
g_model.add(Conv2DTranspose(64, 3, strides=2, padding='same', ))
g_model.add(Conv2DTranspose(32, 3, strides=1, padding='same',))

g_model.add(Conv2DTranspose(1, 3, strides=1, padding='same', activation='tanh'))
class DCGAN:
    def __init__(self, d_model, g_model,
                 input_dim=784, g_dim=100,
                 max_step=100, sample_size=256, d_iter=3, kind='normal'):
        self.input_dim = input_dim #12 
        self.g_dim = g_dim  # 13.
        self.max_step =max_step  # 14. 
        self.sample_size = sample_size
        self.d_iter = d_iter
        self.kind = kind
        self.d_model = d_model  # 
        self.g_model = g_model  # 
        self.m_model = self.merge_model()  # 
        self.optimizer = adam_v2.Adam(lr=0.002, beta_1=0.5)

