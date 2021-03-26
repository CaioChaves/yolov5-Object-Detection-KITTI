'''
    This script converts the 'N' label files provided with the KITTI dataset
    into an adequate format for yolov5 training (transfer learning)

    The input is N .txt files:
        - one file = one image's label
        - one row per object
        - class(str), trunc(float,[0.0-1.0]), occl(int[1,2,3]), alpha(float[-pi,+pi]) ...
          ... x1,y1,x2,y2 (float, left-top and bottom-rpixels),
          ... dimensions h,w,l (float, meters),
          ... location x,y,z (float, meters),
          ... rotation_y (float[-pi,+pi])

    The output is N .txt files:
        - one file = one image's label
        - one row per object
        - each row is class_id(int) x_center(float) y_center(float) width(float) height(float) format
        - x,y,w,h are normalized between 0.0 and 1.0
        - class number are 0-index
'''

import os

# Image typical resolution
WIDTH = 1242
HEIGHT = 375
kittinames = {'Car':0,
              'Van':1,
              'Truck':2,
              'Pedestrian':3,
              'Person_sitting':4,
              'Cyclist':5,
              'Tram':6,
              'Misc':7,
              }

curr_dir = os.getcwd()

names_list = os.listdir(os.path.join(curr_dir,"kitti/train/images"))
names_list = sorted(names_list) # arrange list alphabetically

# Input files dir
input_dir = "/home/caiochaves/PFE/data/KITTI/object/training/label_2/"

# Output files dir
output_dir_t = "/home/caiochaves/PFE/code/yolov5/kitti/train/labels/"
output_dir_v = "/home/caiochaves/PFE/code/yolov5/kitti/valid/labels/"

# Train and validations lists
train_list = open("/home/caiochaves/PFE/data/KITTI/object/lists/trainsplit_chen.txt","r").read().split("\n")
val_list   = open("/home/caiochaves/PFE/data/KITTI/object/lists/valsplit_chen.txt","r").read().split("\n")

for idx, name in enumerate(names_list):
    name = name[0:6] # eliminate .png

    # Read the input file for this image
    input_label = open(input_dir+name+".txt","r").read().split("\n")
    
    # Create the output file for this image
    if name in train_list:
        output_label = open(output_dir_t+name+".txt","w")
    elif name in val_list:
        output_label = open(output_dir_v+name+".txt","w")
    elif name in train_list and val_list:
        print('ERROR: the same name should not be in both lists')
    else:
        print('ERROR: name should be in at least one list')
    
    for line in input_label:
        if line == '':
            break
        fields = line.split(' ')

        class_str   = fields[0]

        if class_str == 'DontCare':
            break

        x1          = float(fields[4])
        y1          = float(fields[5])
        x2          = float(fields[6])
        y2          = float(fields[7])

        # Write the output file for this image
        class_id    = kittinames[class_str]
        x_center    = round(0.5*(x1+x2)/WIDTH,4)
        y_center    = round(0.5*(y1+y2)/HEIGHT,4)
        width       = round((x2-x1)/WIDTH,4)
        height      = round((y2-y1)/HEIGHT,4)
        new_line    = str(class_id) + str(' ') + \
                      str(x_center) + str(' ') + \
                      str(y_center) + str(' ') + \
                      str(width)    + str(' ') + \
                      str(height)   + '\n'
        output_label.write(new_line)

    # Close files
    output_label.close()