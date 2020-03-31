import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, BatchNormalization, Conv3D, Dense, Flatten, Reshape, Conv3DTranspose
from tensorflow.keras.models import Model

input_shape = (1, 32, 32, 32)
z_dim = 128

def get_model():
    enc_in = Input(shape = input_shape)

    enc_idx = Input(shape = (1,), dtype = 'int32')

    enc_conv1 = BatchNormalization()(
        Conv3D(
            filters = 32,
            kernel_size = (4, 4, 4),
            strides = (2, 2, 2),
            padding = 'same',
            activation = 'relu',
            data_format = 'channels_first')(enc_in))
    enc_conv2 = BatchNormalization()(
        Conv3D(
            filters = 16,
            kernel_size = (4, 4, 4),
            strides = (2, 2, 2),
            padding = 'same',
            activation = 'relu',
            data_format = 'channels_first')(enc_conv1))
    enc_conv3 = BatchNormalization()(
        Conv3D(
            filters = 8,
            kernel_size = (4, 4, 4),
            strides = (2, 2, 2),
            padding = 'same',
            activation = 'relu',
            data_format = 'channels_first')(enc_conv2))

    z = Dense(
        units = z_dim,
        activation = 'relu')(Flatten()(enc_conv3))

    encoder = Model(enc_in, z)

    dec_in = Input(shape = (z_dim, ))

    dec_fc1 = Dense(
        units = 512,
        activation = 'relu')(dec_in)
    dec_unflatten = Reshape(
        target_shape = (8, 4, 4, 4))(dec_fc1)

    dec_conv1 = BatchNormalization()(
        Conv3DTranspose(
            filters = 16,
            kernel_size = (4, 4, 4),
            strides = (2, 2, 2),
            padding = 'same',
            activation = 'relu',
            data_format = 'channels_first')(dec_unflatten))
    dec_conv2 = BatchNormalization()(
        Conv3DTranspose(
            filters = 32,
            kernel_size = (4, 4, 4),
            strides = (2, 2, 2),
            padding = 'same',
            activation = 'relu',
            data_format = 'channels_first')(dec_conv1))
    dec_conv3 = Conv3DTranspose(
        filters = 1,
        kernel_size = (4, 4, 4),
        strides = (2, 2, 2),
        padding = 'same',
        activation = 'sigmoid',
        data_format = 'channels_first')(dec_conv2)

    decoder = Model(dec_in, dec_conv3)

    dec_conv3 = decoder(encoder(enc_in))

    gae = Model([enc_in, enc_idx], dec_conv3)

    return {'inputs': enc_in,
            'indices': enc_idx,
            'outputs': dec_conv3,
            'z': z,
            'encoder': encoder,
            'decoder': decoder,
            'gae': gae}
