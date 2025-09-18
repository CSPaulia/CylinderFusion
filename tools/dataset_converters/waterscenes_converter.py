import numpy as np
from os import path as osp
from pathlib import Path

import mmengine

split_list = ['train', 'valid', 'test']

def get_calibs(file_path):
    T = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split(' ')[1:]
            for i in range(len(line)):
                line[i] = float(line[i])
            line = np.array(line)
            line = line.reshape(-1, 4)
            T.append(line)

    return T[0], T[1]


def get_waterscenes_info(root_path, split):
    """Create info file in the form of
        data_infos={
            'metainfo': {'DATASET': 'WaterScenes'},
            'data_list': {
                05827: {
                    'lidar_points':{
                        'lidat_path':'points/05827.bin',
                        'num_pts_feats': 9
                    },
                    'pts_semantic_mask_path':
                        'semantic_mask/05827.labbel',
                    'sample_id': '05827',
                    'cams': {
                        'CAM_FRONT': {
                            'data_path': 'images/04161.jpg', 
                            'type': 'CAM_FRONT', 
                            'radar2cam_rotation': array(
                                [[ 0.9994332 ,  0.02416657,  0.02343607],
                                [-0.0242013 ,  0.99970639,  0.00119925],
                                [-0.0234002 , -0.00176575,  0.99972462]]), 
                            'radar2cam_translation': array(
                                [-0.0064237 ,  0.11013717,  0.03204261]), 
                            'cam_intrinsic': array(
                                [[ 1.06575370e+03, -2.61380460e+00,  8.90125800e+02],
                                [-0.00000000e+00,  1.06536119e+03,  4.92751658e+02],
                                [-0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
                        }
                    }
                },
                ...
            }
        }
    """
    data_infos = dict()
    data_infos['metainfo'] = dict(DATASET='WaterScenes')
    data_list = []

    split_path = osp.join(root_path, 'ImageSets', split + '.txt')
    with open(split_path, 'r') as file:
        for line in file:
            line = line.strip()

            radar2cam, cam_intrinsic = get_calibs(osp.join(root_path, 'calibs', line + '.txt'))
            radar2cam = radar2cam @ np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

            data_list.append({
                'lidar_points': {
                    'lidar_path':
                    osp.join('points',
                             line + '.bin'),
                    'num_pts_feats':
                    9
                },
                'pts_semantic_mask_path':
                osp.join('semantic_mask',
                         line + '.bin'),
                'sample_id':
                line,
                'images': {
                    'CAM_FRONT': {
                        'img_path': 
                        osp.join('images',
                                 line + '.jpg'),
                        'type':
                        'CAM_FRONT',
                        'lidar2cam':
                        radar2cam,
                        'cam2img':
                        cam_intrinsic[0:3, 0:3],
                        'img_seg_path':
                        osp.join('image_annotations',
                                 line + '.png')
                    }
                }
            })

    file.close()
    data_infos.update(dict(data_list=data_list))
    return data_infos


def create_waterscenes_info_file(root_path, pkl_prefix, save_path):
    """Create info file of WaterScenes dataset.

    Directly generate info file without raw data.

    Args:
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
    """
    print('Generate info.')
    save_path = Path(save_path)

    waterscenes_infos_train = get_waterscenes_info(root_path, split='train')
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'WaterScenes info train file is saved to {filename}')
    mmengine.dump(waterscenes_infos_train, filename)
    waterscenes_infos_val = get_waterscenes_info(root_path, split='val')
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'WaterScenes info val file is saved to {filename}')
    mmengine.dump(waterscenes_infos_val, filename)
    waterscenes_infos_test = get_waterscenes_info(root_path, split='test')
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'WaterScenes info test file is saved to {filename}')
    mmengine.dump(waterscenes_infos_test, filename)


if __name__ == '__main__':
    root_path = '/data1/csj_data/dataset/WaterScenes/WaterScenes-Published-unzip-KITTI-format/'
    infos = get_waterscenes_info(root_path, split='test')
    print(infos['data_list'][100])