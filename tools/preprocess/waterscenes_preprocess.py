import os
import csv
import argparse
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

def parse_args():
    parser = argparse.ArgumentParser(description='Data preprocess arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        default='/data1/csj_data/dataset/WaterScenes/WaterScenes-Published-unzip-copy',
        help='specify the root path of dataset')
    parser.add_argument(
        '--out-root',
        type=str,
        default='/data1/csj_data/dataset/WaterScenes/WaterScenes-Published-unzip-KITTI-format',
        help='specify the output path of dataset')
    parser.add_argument(
        '--radar_pc_dir',
        type=str,
        default='radar',
        help='specify the dir name of radar data')
    args = parser.parse_args()
    return args

def point_cloud_csv2bin(csv_path, saved_bin_path, save_proj_coord=False, save_semantic_label=False, saved_semantic_label_path=None):
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        points = []

        # 跳过csv文件的第一行
        next(csvreader)

        # -------------------------------------------------------------- #
        # 保存 [x, y, z, power, doppler, elevation, comp_velocity] 数据
        # -------------------------------------------------------------- #

        for row in csvreader:
            if not save_proj_coord:
                point_cloud = np.zeros((7,), dtype=np.float32)
            else:
                point_cloud = np.zeros((9,), dtype=np.float32)
            point_cloud[0] = float(row[6])
            point_cloud[1] = float(row[7])
            point_cloud[2] = float(row[8])
            point_cloud[3] = float(row[5])
            point_cloud[4] = float(row[2])
            point_cloud[5] = float(row[4])
            point_cloud[6] = float(row[10])
            if save_proj_coord:
                point_cloud[7] = float(row[11])
                point_cloud[8] = float(row[12])
            points.append(point_cloud)
        points = np.array(points, np.float32)
    
    with open(saved_bin_path, 'wb') as f:
        f.write(points.tobytes())
    
    if save_semantic_label:
        with open(csv_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            points = []

            # 跳过csv文件的第一行
            next(csvreader)

            # -------------------------------------------------------------- #
            # 保存 [x, y, z, power, doppler, elevation, comp_velocity] 数据
            # -------------------------------------------------------------- #

            for row in csvreader:
                point_cloud = np.array([int(row[13])], dtype=np.int8)
                points.append(point_cloud)
            points = np.array(points, np.float32)
        
        with open(saved_semantic_label_path, 'wb') as f:
            f.write(points.tobytes())

def convert_bbox_to_line(xml_path, txt_path):
    with open(txt_path, 'w') as file:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = bbox.find('xmin').text
            ymin = bbox.find('ymin').text
            xmax = bbox.find('xmax').text
            ymax = bbox.find('ymax').text
            class_name = obj.find('name').text

            file.write(f'{xmin} {ymin} {xmax} {ymax} {class_name}\n')

def get_ids(split_set_txt, output_path):
    ids = []

    with open(split_set_txt, 'r') as file:
        for line in file:
            ids.append(line.strip().split('/')[-1].split('.')[0])

    with open(output_path, 'w') as file:
        for id in ids:
            file.write(id + '\n')

def main():
    args = parse_args()

    # -------------------------------------------------------------- #
    # 将csv中的雷达数据保存为bin文件
    # -------------------------------------------------------------- #
    radar_pc_path = os.path.join(args.data_root, args.radar_pc_dir)
    radar_pc_out_path = os.path.join(args.out_root, 'points')

    if not os.path.exists(radar_pc_out_path):
        os.mkdir(radar_pc_out_path)

    if not os.path.exists(radar_pc_path):
        print(f'{radar_pc_path} do not exist.')
    else:
        for csv_file in tqdm(os.listdir(radar_pc_path)):
            bin_file = csv_file.split('.')[0] + '.bin'
            csv_path = os.path.join(radar_pc_path, csv_file)
            bin_path = os.path.join(radar_pc_out_path, bin_file)
            if args.radar_pc_dir != 'radar':
                point_cloud_csv2bin(csv_path, bin_path)
            else:
                label_dir_path = os.path.join(args.out_root, 'semantic_mask')
                if not os.path.exists(label_dir_path):
                    os.mkdir(label_dir_path)
                label_path = os.path.join(label_dir_path, bin_file)
                point_cloud_csv2bin(csv_path, bin_path, True, True, label_path)

    # -------------------------------------------------------------- #
    # 读取xml中的bbox并保存至txt文件中
    # -------------------------------------------------------------- #
    xml_dir_path = os.path.join(args.data_root, 'detection', 'xml')
    txt_dir_path = os.path.join(args.out_root, 'labels')

    if not os.path.exists(txt_dir_path):
        os.mkdir(txt_dir_path)

    if not os.path.exists(xml_dir_path):
        print(f'{xml_dir_path} do not exist.')
    else:
        for xml_file in tqdm(os.listdir(xml_dir_path)):
            txt_file = xml_file.split('.')[0] + '.txt'
            xml_path = os.path.join(xml_dir_path, xml_file)
            txt_path = os.path.join(txt_dir_path, txt_file)
            convert_bbox_to_line(xml_path, txt_path)

    # -------------------------------------------------------------- #
    # 获取训练集、测试集、验证集图像的id并保存在txt文件中
    # -------------------------------------------------------------- #
    txt_files = ['train.txt', 'test.txt', 'val.txt']
    for txt_file in txt_files:
        txt_path = os.path.join(args.data_root, txt_file)
        out_dir = os.path.join(args.out_root, 'ImageSets')

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        out_path = os.path.join(out_dir, txt_file)
        get_ids(txt_path, out_path)

if __name__ == '__main__':
    main()