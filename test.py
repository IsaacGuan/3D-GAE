import re
import os
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from GAE import *
from utils import npytar, save_volume

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

if __name__ == '__main__':
    model = get_model()

    inputs = model['inputs']
    indices = model['indices']
    outputs = model['outputs']
    z = model['z']

    encoder = model['encoder']
    decoder = model['decoder']
    gae = model['gae']

    gae.load_weights('results_gae/gae.h5')

    gae.trainable = False

    data_test = data_loader('datasets/chairs_32.tar')

    z_vectors = encoder.predict(data_test)
    np.savetxt('results_gae/z_vectors.csv', z_vectors, delimiter = ',')

    shape_indices = np.array(range(data_test.shape[0]))

    start_time = time.time()
    reconstructions = gae.predict([data_test, shape_indices])
    end_time = time.time()
    reconstructions[reconstructions >= 0.5] = 1
    reconstructions[reconstructions < 0.5] = 0

    data_test[data_test > 0] = 1
    data_test[data_test < 0] = 0

    error_rates = []

    fout = open('results_gae/test.out', 'w')
    sys.stdout = fout

    if not os.path.exists('results_gae/reconstructions'):
        os.makedirs('results_gae/reconstructions')

    for i in range(reconstructions.shape[0]):
        save_volume.save_output(reconstructions[i, 0, :], 32, 'results_gae/reconstructions', i)
        error_rate = np.mean((reconstructions[i, 0, :] - data_test[i, 0, :]) ** 2)
        error_rates.append(error_rate)
        print('Mean squared error of shape {}: {}'.format(i, error_rate))

    error_rate_total = np.mean((reconstructions - data_test) ** 2)
    print('Mean squared error total: {}'.format(error_rate_total))

    print('Prediction time per shape: {}'.format((end_time - start_time) / reconstructions.shape[0]))

    np.savetxt('results_gae/test_loss_gae.csv', error_rates, delimiter = ',')

    fout.close()
