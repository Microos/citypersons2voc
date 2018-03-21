from pathlib import Path
from PIL import Image
import numpy as np
import shutil
import os
from functools import reduce
import datetime
import pprint

header_tmpl = \
    '''
    <annotation>
        <folder>VOC2007</folder>
        <filename>{}</filename>
        <source>
            <database>My Database</database>
            <annotation>CityPersons</annotation>
            <image>flickr</image>
            <flickrid>NULL</flickrid>
        </source>
        <owner>
            <flickrid>NULL</flickrid>
            <name>facevise</name>
        </owner>
        <size>
            <width>{}</width>
            <height>{}</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
    '''  # .format(img_name_with_ext, img_w, img_h)
object_tmpl = \
    '''
        <object>
            <name>{}</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>{:.3f}</xmin>
                <ymin>{:.3f}</ymin>
                <xmax>{:.3f}</xmax>
                <ymax>{:.3f}</ymax>
            </bndbox>
        </object>
    '''  # .format(lbl, x1, y1, x2, y2)
tail = '</annotation>'


class voc_formatter():
    def __init__(self,
                 img_src_dir,  # cityscapes: expected to have train/test/val dir and sub-dirs with city names
                 des_dir,  # output devkit dir
                 name_bbs_dict_train,
                 name_bbs_dict_val,
                 lbl_map,  # see main.py, set map value to None to discard that class
                 height_range=None,  # e.g.: [50, np.inf] or [50, 75], right-endpoint closed
                 width_range=None,  # e.g.: similar to the `height_range`
                 vis_range=None,
                 enable_train_filter=True,
                 enable_val_filter=False,  # False to disable all the filtering action in val set
                 copy_imgs=True,  # True to copy all the images,
                 handle_ignore=False,  # if Ture, label name 'ignore' will no be seen as a gt box when filtering
                 dir_exist_handling='PROCED'
                 # ABORT: abort the program
                 # PROCED: rm existing file
                 ):

        self.img_src_dir = img_src_dir  # expected to have train/test/val dir and sub-dirs with city names
        self.des_dir = des_dir  # devkit dir
        self.dict_train = name_bbs_dict_train
        self.dict_val = name_bbs_dict_val
        '''
            key: img_name_with_ext
            value: 
                n* <class_label, x1, y1, w, h, instance_id, x1_vis, y1_vis, w_vis, h_vis>
        '''
        self.dir_exist_handling = dir_exist_handling
        self.lbl_map = lbl_map
        if self.dir_exist_handling == 'ABORT':
            self.rm = False
        elif self.dir_exist_handling == 'PROCED':
            self.rm = True
        else:
            raise ValueError('''
            allowed option for "dir_exist_handling":
            ABORT: abort the program
            PROCED: rm existing file
        ''')
        self.handle_ignore = handle_ignore
        self.copy_imgs = copy_imgs

        self.enable_train_filter = enable_train_filter
        self.enable_val_filter = enable_val_filter
        self.height_range = height_range
        self.width_range = width_range
        self.vis_range = vis_range

        self.__prepare()
        self.train_set = []
        self.val_set = []

    def __prepare(self):
        '''
        -. create new dir at dest
        -. check src dir existence 
        # -. check dict dimension
        -. write readme
        :return: 
        '''
        # -. create new dir at dest
        if self.des_dir.exists():
            if self.rm:
                print('remove dir: {}'.format(self.des_dir.absolute()))
                os.system('rm -rf {}'.format(self.des_dir))
            else:
                raise FileExistsError("set dir_exist_handling='PROCED'" +
                                      "to remove the dir at {}".format(self.des_dir.absolute()))
        data_dir = self.des_dir / 'data'
        self.anno_dir = data_dir / 'Annotations'
        self.set_dir = data_dir / 'ImageSets' / 'Main'
        self.img_dir = data_dir / 'JPEGImages'

        self.anno_dir.mkdir(parents=True)
        self.set_dir.mkdir(parents=True)
        if self.copy_imgs:
            self.img_dir.mkdir(parents=True)

        # -. check src dir existence
        self.img_train_dir = self.img_src_dir / 'train'
        self.img_val_dir = self.img_src_dir / 'val'
        if not self.img_train_dir.exists():
            raise FileNotFoundError('{} Not Found'.format(self.img_train_dir))
        if not self.img_val_dir.exists():
            raise FileNotFoundError('{} Not Found'.format(self.img_val_dir))

        # others
        self.drop_cls_id = [k for k, v in self.lbl_map.items() if v is None]
        self.ign_cls_id = [k for k, v in self.lbl_map.items() if v == 'ignore']

    def __write_readme(self, done_str):
        readme_file = self.des_dir / 'README'

        fmt = '''This is a generated dataset.
@Date: {date}

@Statistic: 
{stat}

@Filters:
(a, b]
    - width_range  : {wrng}
    - height_range : {hrng}
    - vis_range    : {vrng}
    
@lbl_map:
{lbl_map_str}

@handle_ignore: {hd_ign}

@filter_train: {ft_tra}

@filter_val: {ft_val}
        '''

        date = datetime.datetime.now().strftime('%Y/%m/%d, %H:%M:%S')
        id_clsname_map = {0: 'ignore', 1: 'pedestrian', 2: 'rider', 3: 'sitting', 4: 'other', 5: 'group'}
        lblm_str = '{\n' + ",\n".join(
            ["{} : '{:>5s}' # {:<10s} ".format(k, str(v), id_clsname_map[k]) for k, v in self.lbl_map.items()]) + "}"
        s = fmt.format(date=date,
                       stat=done_str,
                       wrng=str(self.width_range),
                       hrng=str(self.height_range),
                       vrng=str(self.vis_range),
                       lbl_map_str=lblm_str,
                       hd_ign=self.handle_ignore,
                       ft_tra=self.enable_train_filter,
                       ft_val=self.enable_val_filter)
        with open(str(readme_file), 'w') as f:
            f.write(s)

    def __filter(self, vec: np.ndarray):
        # input n*10d vec
        # output keep_inds + num_dropped
        '''
       Filtering:
           0. no-ins skip
           - cls lbl drop
           1. width_range
           2. height_range
           3. vis_range

           4. no-ins skip
       '''
        #  <class_label, x1, y1, w, h, instance_id, x1_vis, y1_vis, w_vis, h_vis>
        #   0            1   2   3  4  5            6       7       8      9

        wrng = self.width_range
        hrng = self.height_range
        vrng = self.vis_range

        empty_inds = np.array([])
        num_gt = vec.shape[0]

        # 0. no-ins skip
        if num_gt == 0: return empty_inds, num_gt

        # - cls lbl drop
        if len(self.drop_cls_id) != 0:
            lbls = vec[:, 0]
            bool_lbl_inds = np.in1d(lbls, self.drop_cls_id, invert=True)
            lbl_inds = np.arange(num_gt)[bool_lbl_inds]
            if lbl_inds.size == 0: return empty_inds, num_gt
        else:
            lbl_inds = np.arange(num_gt)



        # 1. width range
        wrng_inds = np.arange(num_gt)
        if wrng is not None:
            ws = vec[:, 3]
            wrng_inds = np.where(np.logical_and(ws > wrng[0], ws <= wrng[1]))[0]
            if wrng_inds.size == 0: return empty_inds, num_gt

        # 2. height range
        hrng_inds = np.arange(num_gt)
        if hrng is not None:
            hs = vec[:, 4]
            hrng_inds = np.where(np.logical_and(hs > hrng[0], hs <= hrng[1]))[0]
            if hrng_inds.size == 0: return empty_inds, num_gt

        # 3. vis range
        vrng_inds = np.arange(num_gt)
        if vrng is not None:
            full_area = vec[:, 3] * vec[:, 4]
            vis_area = vec[:, 8] * vec[:, 9]
            vs = vis_area / full_area
            vrng_inds = np.where(np.logical_and(vs > vrng[0], vs <= vrng[1]))[0]
            if vrng_inds.size == 0: return empty_inds, num_gt

        # merge all
        inds_list = (lbl_inds, wrng_inds, hrng_inds, vrng_inds)
        keep_inds = reduce(np.intersect1d, inds_list)

        kept_num = keep_inds.size
        dropped_num = num_gt - kept_num

        return keep_inds, dropped_num

    def __write_xml(self, img_name, w, h, bbs):
        content = header_tmpl.format(img_name, w, h)
        for bb in bbs:
            [lbl, x1, y1, x2, y2] = bb
            content += object_tmpl.format(self.lbl_map[lbl], x1, y1, x2, y2)
        content += tail
        xml_name = img_name.split('.')[0] + '.xml'
        xml_file = self.anno_dir / xml_name
        with open(str(xml_file), 'w') as f:
            f.write(content)

    def __run(self, src_img_dir, name_bbs_dict, name_set, enable_filter):
        total_dropped = 0
        total_gt_num = 0  # total kept number of gt
        '''
        1. for each img:
            @. get img w, h
            @. copy img (if True)
            
            # handle ignore (if True)
            # filtering (if True)

            @. format bbs: 10-d to 5-d
            @. xywh to xyxy
            @. write xml
            @. append name_set
        '''
        tv = 'train' if 'train' in str(src_img_dir) else 'val'
        print('Processing {} ...'.format(tv))
        for k, vec in name_bbs_dict.items():

            city_name = k.split('_')[0]
            img_file = src_img_dir / city_name / k

            # get img w, h
            with Image.open(img_file) as img:
                w, h = img.size

            # copy img
            if self.copy_imgs:
                k_png = k.replace('.png', '.jpg')  # dirty-and-quick, w/o actually tamper image data
                dest_img = self.img_dir / k_png
                shutil.copyfile(str(img_file), str(dest_img))

            # - ignore handle
            #  check: non ignore region num != 0
            if self.handle_ignore and len(self.ign_cls_id) != 0:
                lbls = vec[:, 0]
                bool_non_ign_inds = np.in1d(lbls, self.ign_cls_id, invert=True)
                ign_vec = vec[np.invert(bool_non_ign_inds),:]
                vec = vec[bool_non_ign_inds,:]

            if enable_filter:
                keep_inds, dropped_num = self.__filter(vec)
                total_dropped += dropped_num
                if keep_inds.size == 0: continue
                vec = vec[keep_inds, :]

            total_gt_num += vec.shape[0]
            if self.handle_ignore:
                vec = np.vstack((vec,ign_vec))
            # format bbs: 10-d to 5-d
            bbs = vec[:, [0, 1, 2, 3, 4]]


            # xywh to xyxy
            bbs[:, 3] += bbs[:, 1] - 1  # adding -1
            bbs[:, 4] += bbs[:, 2] - 1  # adding -1
            # write xml
            self.__write_xml(k, w, h, bbs)

            # append name_set
            name_set.append(k.split('.')[0])
        return total_gt_num, total_dropped

    def __write_set_file(self, name_set, tv):
        set_file = self.set_dir / tv
        with open('{}.txt'.format(set_file), 'w') as f:
            f.write('\n'.join(name_set))

    def run(self):
        '''
        1. process train/val
        2. write set file
        '''
        total_gt_num_train, total_dropped_train = \
            self.__run(self.img_train_dir, self.dict_train, self.train_set, self.enable_train_filter)

        total_gt_num_val, total_dropped_val = \
            self.__run(self.img_val_dir, self.dict_val, self.val_set, self.enable_val_filter)

        self.__write_set_file(self.train_set, 'train')
        # self.__write_set_file(self.val_set, 'val')

        # use a pre-calculated val list ordered by val image id
        # this ease the issue at faster-rcnn result output stage
        shutil.copyfile('./precalculated_ordered_val.txt',
                        str(self.set_dir / 'val.txt'))

        num_img_train = len(self.train_set)
        num_img_val = len(self.val_set)
        done_str = '  train: {} images; {:>6} bboxes; {:.2f} boxes/image\n'.format(num_img_train, total_gt_num_train,
                                                                                   float(
                                                                                       total_gt_num_train) / num_img_train)
        done_str += '  val  : {} images; {:>6} bboxes; {:.2f} boxes/image'.format(num_img_val, total_gt_num_val,
                                                                                  float(total_gt_num_val) / num_img_val)
        print('Done:')
        print(done_str)

        if not self.copy_imgs:
            info = '# Note #: Folder JPEGImages is not created. (copy_imgs=False)'
            bar = ''.join(['-' for _ in range(len(info))])
            print(bar)
            print(info)
            print(bar)
        self.__write_readme(done_str)
