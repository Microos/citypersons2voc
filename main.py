#!/usr/bin/python3

from cps_mat_ops import parse_mat
from pathlib import Path
import voc_ops

# dir that contains *.mat
citypersons_annotaions_dir = Path('./data/annotations')

# downloaded from: https://www.cityscapes-dataset.com/
# expected orignal folder name: leftImg8bit
# expected to contain: train,val,test(optional). 3 sub-dirs
cityperson_image_root_dir = Path('./data/leftImg8bit')

# your customized devkit output dir:
devkit_output_dir = Path('./citypersons_devkit_ped')

# index-to-string map based on:
# https://bitbucket.org/shanshanzhang/citypersons/src/c13bbdfa986222c7dc9b4b84cc8a24f58d7ab72b/annotations/?at=default
lbl_map = {
    0: None, #'ignore', #set to None to remove this class
    1: 'ped',
    2: None, #'rider',
    3: None, # 'sit',
    4: None, # 'other',
    5: None, #'group'
}  # ignore,ped,rider,sit,other,group



# parse *.mat
train_mat = citypersons_annotaions_dir / 'anno_train.mat'
val_mat = citypersons_annotaions_dir / 'anno_val.mat'

train_dict = parse_mat(train_mat, lbl_map, filter=True)
val_dict = parse_mat(val_mat, lbl_map , filter=False)

# convert
vf = voc_ops.voc_formatter(cityperson_image_root_dir,
                           devkit_output_dir,
                           train_dict,
                           val_dict,
                           lbl_map,
                           copy_imgs=True,
                           dir_exist_handling='ABORT')


vf.run()
