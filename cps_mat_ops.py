"""
Citypersons .mat annotation operations
"""
from scipy.io import loadmat
from pathlib import Path
import numpy as np


def parse_mat(file_path):
    '''
    Read .mat file from file_path, return a dict
    
    :param file_path: where the .mat is.
    :return: a dict, key: imagename, value: an array of n* <lbl, x1 y1 x2 y2>
    '''
    file_path = str(file_path)
    tv = 'train' if 'train' in str(file_path) else 'val'
    k = 'anno_{}_aligned'.format(tv)
    noins_counter = 0
    print('Parsing mat file from {}'.format(file_path))
    mat = loadmat(file_path, mat_dtype=True)[k][0]  # uint8 overflow fix
    name_bbs_dict = {}
    for img_idx in range(len(mat)):
        # each image
        img_anno = mat[img_idx][0, 0]
        city_name = img_anno[0][0]
        img_name_with_ext = img_anno[1][0]
        bbs = img_anno[2]  # n x 10 matrix
        # 10-D: [class_label, x1, y1, w, h, instance_id, x1_vis, y1_vis, w_vis, h_vis]

        # no-instance filter
        if bbs.shape[0] == 0:
            noins_counter += 1
            continue

        bbs = bbs[:, [0, 1, 2, 3, 4]]
        # lbl x1 y1 w h
        # 0   1  2  3 4

        bbs[:, 3] += bbs[:, 1]
        bbs[:, 4] += bbs[:, 2]

        name_bbs_dict[img_name_with_ext] = bbs

    print(' --> {} images contains no instances filtered'.format(noins_counter))
    return name_bbs_dict


if __name__ == '__main__':
    # test

    anno_dir = Path('./data/annotations')
    train_mat = anno_dir / 'anno_train.mat'
    val_mat = anno_dir / 'anno_val.mat'

    d = parse_mat(train_mat)
    print(d.keys())
    print(len(d))
