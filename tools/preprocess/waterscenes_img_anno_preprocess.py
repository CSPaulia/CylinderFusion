from mmengine.fileio import get
import numpy as np
import mmcv
import os
from tqdm import tqdm
import torch
import argparse

# the original label
original_label = np.array([0, 15, 19, 38, 53, 75, 90, 113, 128])
# revised label
revised_label = after = np.array([0, 4, 8, 1, 5, 2, 6, 3, 7])
# the corresponding class name of each label
class_name = ['free_space', 'ship', 'water', 'pier', 'boat', 'buoy', 'vessel', 'sailor', 'kayak']

def parse_args():
    parser = argparse.ArgumentParser(description='2D annotation converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        default='/data1/csj_data/dataset/WaterScenes/WaterScenes-Published-unzip-KITTI-format/',
        help='specify the root path of dataset')
    parser.add_argument(
        '--resource_dir',
        type=str,
        default='image_anno',
        help='specify the name of the resource directory')
    parser.add_argument(
        '--destination_dir',
        type=str,
        default='image_annotations',
        help='specify the name of the destination directory')
    args = parser.parse_args()
    return args

def convert_image_annotation(root_dir, 
                             resource_dir, 
                             destination_dir, 
                             mapping):
    for anno in tqdm(os.listdir(os.path.join(root_dir, resource_dir))):
        anno_filename = os.path.join(root_dir, resource_dir, anno)

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_anno_byte = get(anno_filename, backend_args=None)
        
        img_anno = mmcv.imfrombytes(
            img_anno_byte,
            flag='grayscale',
            backend='pillow',
            channel_order='rgb')

        img_anno = mapping[img_anno]
        
        mmcv.imwrite(img_anno, os.path.join(root_dir, destination_dir, anno))

def main():
    args = parse_args()

    mapping = np.zeros((max(original_label) + 1,))
    mapping[original_label] = revised_label
    # print(mapping)

    convert_image_annotation(args.data_root,
                             args.resource_dir,
                             args.destination_dir,
                             mapping=mapping)

if __name__ == '__main__':
    main()