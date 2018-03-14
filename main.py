#!/usr/bin/python3

from cps_mat_ops import parse_mat
from pathlib import Path
import numpy as np
import voc_ops

# dir that contains *.mat
citypersons_annotaions_dir = Path('./data/annotations')

# downloaded from: https://www.cityscapes-dataset.com/
# expected orignal folder name: leftImg8bit
# expected to contain: train,val,test(optional). 3 sub-dirs
cityperson_image_root_dir = Path('./data/leftImg8bit')

# your customized devkit output dir:
devkit_output_dir = Path('./cps_dvkt_w10hv50_ignore')

# index-to-string map based on:
# https://bitbucket.org/shanshanzhang/citypersons/src/c13bbdfa986222c7dc9b4b84cc8a24f58d7ab72b/annotations/?at=default
lbl_map = {
    0: 'ignore',  # 'ignore', #set to None to remove this class
    1: 'ped',
    2: 'ped',  # 'rider',
    3: 'ignore',  # 'sit',
    4: 'ignore',  # 'other',
    5: 'ignore'  # 'group'
}  # ignore,ped,rider,sit,other,group

# parse *.mat
train_mat = citypersons_annotaions_dir / 'anno_train.mat'
val_mat = citypersons_annotaions_dir / 'anno_val.mat'

train_dict = parse_mat(train_mat)  # , lbl_map, filter=True)
val_dict = parse_mat(val_mat)  # , lbl_map , filter=False)

# convert
vf = voc_ops.voc_formatter(cityperson_image_root_dir,
                           devkit_output_dir,
                           train_dict,
                           val_dict,
                           lbl_map,
                           height_range=None,  # [45, np.inf],
                           width_range=[10, np.inf],
                           vis_range=[0.5, np.inf],
                           enable_train_filter=True,
                           enable_val_filter=False,
                           handle_ignore=True,
                           copy_imgs=False,
                           dir_exist_handling='PROCED')

vf.run()
