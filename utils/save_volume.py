from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from skimage import measure

def save_output(output_arr, output_size, output_dir, file_idx):
    with_border_arr = np.zeros([output_size + 2, output_size + 2, output_size + 2])

    text_save = np.reshape(output_arr, (output_size * output_size * output_size))
    np.savetxt(output_dir + '/volume' + str(file_idx) + '.txt', text_save)

    output_image = np.reshape(output_arr, (output_size, output_size, output_size)).astype(np.float32)
    with_border_arr = np.pad(output_image, pad_width = 1, mode = 'constant', constant_values = 0)

    if not np.any(with_border_arr):
        verts, faces, normals, values = [], [], [], []
    else:
        verts, faces, normals, values = measure.marching_cubes_lewiner(with_border_arr, level = 0.0, gradient_direction = 'descent')
        faces = faces + 1

    obj_save = open(output_dir + '/volume' + str(file_idx) + '.obj', 'w')
    for item in verts:
        obj_save.write('v {0} {1} {2}\n'.format(item[0], item[1], item[2]))
    for item in normals:
        obj_save.write('vn {0} {1} {2}\n'.format(-item[0], -item[1], -item[2]))
    for item in faces:
        obj_save.write('f {0}//{0} {1}//{1} {2}//{2}\n'.format(item[0], item[2], item[1]))
    obj_save.close()

    output_image = np.rot90(output_image)
    x, y, z = output_image.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x, y, z, zdir = 'z', c = 'red')
    plt.savefig(output_dir + '/volume' + str(file_idx) + '.png')
    plt.close()
