#!/usr/bin/python3

from cps_mat_ops import parse_mat
from pathlib import Path
import voc_ops

# dir that contains *.mat
citypersons_annotaions_dir = Path('./data/annotations')

# downloaded from: https://www.cityscapes-dataset.com/
# expected orignal name: leftImg8bit
# expected to contain: train,val,test(optional) sub-dirs
cityperson_image_root_dir = Path('./data/leftImg8bit')

# customized devkit output dir:
devkit_output_dir = Path('./citypersons_devkit')

# index-to-string map based on:
# https://bitbucket.org/shanshanzhang/citypersons/src/c13bbdfa986222c7dc9b4b84cc8a24f58d7ab72b/annotations/?at=default
lbl_map = {
    0: 'ignore',
    1: 'ped',
    2: 'rider',
    3: 'sit',
    4: 'other',
    5: 'group'
}

# parse *.mat
train_mat = citypersons_annotaions_dir / 'anno_train.mat'
val_mat = citypersons_annotaions_dir / 'anno_val.mat'

train_dict = parse_mat(train_mat)
val_dict = parse_mat(val_mat)

vf = voc_ops.voc_formatter(cityperson_image_root_dir,
                            devkit_output_dir,
                            train_dict,
                            val_dict,
                            lbl_map,
                            file_exist_handling='ABORT'
                           )
