import re
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tqdm import tqdm
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from scipy.spatial import distance_matrix
from sklearn.neighbors import KDTree

from GAE import *
from utils import npytar

learning_rate = 0.001
batch_size = 10
epoch_num = 200

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def data_loader(fname):
    x_dic = {}
    reader = npytar.NpyTarReader(fname)
    for ix, (x, name) in enumerate(reader):
        x_dic[name] = x.astype(np.float32)
    reader.reopen()
    xc = np.zeros((reader.length(), ) + input_shape, dtype = np.float32)
    i = 0
    for ik in sorted(x_dic.keys(), key = natural_keys):
        xc[i] = x_dic[ik]
        i += 1
    return xc

def loss_wrapper(data_train, sort_distance_idx, indices):
    def generalized_loss(y_true, y_pred):
        loss = 0
        loss += mean_squared_error(tf.gather(data_train, indices), y_pred)
        for i in range(batch_size):
            curr_idx = indices[i]
            sort_idx = tf.gather(sort_distance_idx, curr_idx)
            data_true = tf.gather(data_train, sort_idx[0, 0])
            for j in range(10):
                curr_data_train = tf.gather(data_train, sort_idx[0, j])
                s = tf.math.exp(-(tf.norm(data_true - curr_data_train) ** 2) / 200)
                loss += s * mean_squared_error(curr_data_train, y_pred[i, :])
        return loss

    return generalized_loss

if __name__ == '__main__':
    if not os.path.exists('results_gae'):
        os.makedirs('results_gae')

    model = get_model()

    inputs = model['inputs']
    indices = model['indices']
    outputs = model['outputs']
    z = model['z']

    encoder = model['encoder']
    decoder = model['decoder']

    plot_model(encoder, to_file = 'results_gae/gae_encoder.pdf', show_shapes = True)
    plot_model(decoder, to_file = 'results_gae/gae_decoder.pdf', show_shapes = True)

    gae = model['gae']

    adam = Adam(lr = learning_rate)

    plot_model(gae, to_file = 'results_gae/gae.pdf', show_shapes = True)

    data_train = data_loader('datasets/chairs_32.tar')

    if not os.path.exists('datasets/chairs_euclidean_distances.npy'):
        data_train_flatten = data_train.reshape((data_train.shape[0], data_train.shape[2] * data_train.shape[3] * data_train.shape[4]))
        euclidean_distances = distance_matrix(data_train_flatten, data_train_flatten)
        euclidean_distances = (euclidean_distances - euclidean_distances.min()) / (euclidean_distances.max() - euclidean_distances.min())
        np.save('datasets/chairs_euclidean_distances.npy', euclidean_distances)
    else:
        euclidean_distances = np.load('datasets/chairs_euclidean_distances.npy')

    if not os.path.exists('datasets/chairs_chamfer_distances.npy'):
        data_train_ones_idx = []
        for i in range(data_train.shape[0]):
            data_train_ones_idx.append(np.argwhere(data_train[i, 0, :] == 1))
        chamfer_distances = np.zeros([data_train.shape[0], data_train.shape[0]])
        for i in range(len(data_train_ones_idx)):
            x = data_train_ones_idx[i]
            tree_x = KDTree(x, leaf_size = x.shape[0] + 1)
            for j in range(len(data_train_ones_idx)):
                y = data_train_ones_idx[j]
                tree_y = KDTree(y, leaf_size = y.shape[0] + 1)
                distances_xy, _ = tree_x.query(y)
                distances_yx, _ = tree_y.query(x)
                chamfer_distances[i, j] = np.sum(distances_xy) + np.sum(distances_yx)
        chamfer_distances = (chamfer_distances - chamfer_distances.min()) / (chamfer_distances.max() - chamfer_distances.min())
        np.save('datasets/chairs_chamfer_distances.npy', chamfer_distances)
    else:
        chamfer_distances = np.load('datasets/chairs_chamfer_distances.npy')

    distances = 0 * euclidean_distances + 1 * chamfer_distances

    sort_distance_idx = []
    for i in range(distances.shape[0]):
        sort_distance_idx.append(np.argsort(distances[i, :]))
    sort_distance_idx = np.asarray(sort_distance_idx)

    gae.compile(optimizer = adam, loss = loss_wrapper(data_train, sort_distance_idx, indices))

    shape_indices = np.array(range(data_train.shape[0]))

    for e in range(1, epoch_num + 1):
        print('Epoch %d' % e)
        gae_loss = 0
        for i in tqdm(range(data_train.shape[0] // batch_size)):
            shape_batch = data_train[i * batch_size: (i + 1) * batch_size]
            indices_batch = shape_indices[i * batch_size: (i + 1) * batch_size]
            gae_loss += gae.train_on_batch([shape_batch, indices_batch], shape_batch)
        print('gae_loss: %f' % np.mean(gae_loss))

    gae.save_weights('results_gae/gae.h5')
