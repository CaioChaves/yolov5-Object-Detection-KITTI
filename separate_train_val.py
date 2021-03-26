'''
    This script splits files into training and validation subsets
'''

import os
import shutil

curr_dir = os.getcwd()

names_list = os.listdir(os.path.join(curr_dir,"kitti/train/images_all"))
names_list = sorted(names_list)     # arrange list alphabetically

# Train and validations lists
train_list = open("/home/caiochaves/PFE/data/KITTI/object/lists/trainsplit_chen.txt","r").read().split("\n")
val_list   = open("/home/caiochaves/PFE/data/KITTI/object/lists/valsplit_chen.txt","r").read().split("\n")

for name in names_list:
    if name[0:6] in train_list:
        shutil.copy("/home/caiochaves/PFE/data/KITTI/object/training/image_2/"+name,os.path.join(curr_dir,"kitti/train/images/")+name)
    elif name[0:6] in val_list:
        shutil.copy("/home/caiochaves/PFE/data/KITTI/object/training/image_2/"+name,os.path.join(curr_dir,"kitti/valid/images/")+name)
    elif name[0:6] in val_list and name[0:6] in val_list:
        print('ERROR! Name should not be in both lists')
    else:
        print('ERROR! Name should be in one split list')
