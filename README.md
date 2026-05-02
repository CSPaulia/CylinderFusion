<div align="center">   

# CylinderFusion: Self-Adaptive Cylindrical 3+1D Radar-Camera Fusion for Waterway Point Cloud Segmentation

</div>

<div align="center">   
  
[![License: Apache](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Xplore-blue)](https://ieeexplore.ieee.org/abstract/document/11461037)

</div>

## Abstract

Point cloud segmentation is crucial for unmanned vehicle perception on water, and radar–camera fusion further improves its performance. Most existing BEV methods fuse features within cubic voxel space, ignoring the non-uniform distribution of outdoor point clouds and image frustum points. Their performance also degrades severely under adverse weather or sensor malfunctions. To address these, we propose CylinderFusion, a robust radar–camera fusion network. It introduces a novel paradigm for multimodal fusion within cylindrical voxel space and incorporates a specially designed scatter module. To improve robustness, we introduce a Dynamic Feature Selection (DFS) mechanism that adaptively weights features during fusion. Our method achieves state-of-the-art results on the large-scale WaterScenes dataset and demonstrates strong performance on the VoD dataset. Extensive ablations validate its effectiveness.

## Qualitative results

### Visualization results on [WaterScenes](https://github.com/WaterScenes/WaterScenes) test set

https://github.com/user-attachments/assets/564c7a96-b5f5-4852-85df-48fef4fc5580

![waterscenes](resources/robustness.png)

Visualization results of different models on the WaterScenes dataset and the image enhancement dataset. Red points represent incorrect predictions, while blue points indicate correct predictions.

### Visualization results on [VoD](https://github.com/tudelft-iv/view-of-delft-dataset) validation set

https://github.com/user-attachments/assets/e49046f6-9d53-4b3e-96ca-bb4dd7e8ef9c

![vod](resources/vod.png)

Visualization results on the VoD dataset. Red bounding boxes indicate cyclists, green indicate pedestrians, and blue indicate cars.

## Environment

1. Create a conda environment and activate it.

    ```bash
    conda create --name cylinderfusion python=3.8 -y
    conda activate cylinderfusion
    ```

2. Install Pytorch.

    ```bash
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    ```

3. Install MMEngine, MMCV and MMDetection using MIM.

    ```bash
    pip install -U openmim
    mim install mmengine
    mim install 'mmcv==2.1.0'
    mim install 'mmdet==3.3.0'
    ```

4. Install mmdet3D from this repository.
    
    ```bash
    pip install -v -e .
    ```

5. Install other dependencies.

    ```bash
    pip install einops timm spconv-cu117
    # Install DCNv3
    cd projects/CylinderFusion/cylinderfusion/ops_dcnv3
    sh ./make.sh
    # Install BEVPooling
    python projects/CylinderFusion/setup.py develop
    ```

## Dataset

Generate KITTI format Dataset from [original waterscenes dataset](https://github.com/waterscenes/waterscenes):

```bash
# preprocessing radar data, bbox and dataset partition
python tools/preprocess/waterscenes_preprocess.py --data-root /path/to/your/original/dataset --out-root /path/to/your/destination --radar_pc_dir radar
# copy calibs, images, image segmentation annotations directory
cp -r /path/to/your/original/dataset/calib /path/to/your/destination/calibs
cp -r /path/to/your/original/dataset/images /path/to/your/destination/images
cp -r /path/to/your/original/dataset/semantic/SegmentationClass /path/to/your/destination/image_anno
# convert image segmentation annotations
python tools/preprocess/waterscenes_img_anno_preprocess.py --data-root /path/to/your/destination --resource_dir image_anno --destination_dir image_annotations
```

Generate pkl files:
  
```bash
python tools/create_data.py waterscenes --root-path /path/to/your/destination --out-dir /path/to/your/destination --extra-tag waterscenes
```

## Training and Evaluating

Train CylinderFusion with multiple GPUs:

```bash
./tools/dist_train.sh projects/CylinderFusion/configs/cylinderfusionseg_radar-cam_voxel720-720-32_dcalpe-fuser_second_secfpn_6xb4-cyclic-20e_xyz2xzy_waterscenes.py $num_gpu
```

Evaluate CylinderFusion with multiple GPUs:

```bash
./tools/dist_test.sh projects/CylinderFusion/configs/cylinderfusionseg_radar-cam_voxel720-720-32_dcalpe-fuser_second_secfpn_6xb4-cyclic-20e_xyz2xzy_waterscenes.py $ckpt $num_gpu
```

## Model Checkpoint

| Model | Dataset | Link |
|-------|---------|------|
| CylinderFusion | WaterScenes | [Google Drive](https://drive.google.com/file/d/1YQQm3w6uQPEYqgtycdjeNcZWG0cNv9tS/view?usp=drive_link) |

## Citation

```bibtex
@inproceedings{chen2026cylinderfusion,
  title={Cylinderfusion: Self-Adaptive Cylindrical 3+ 1D Radar-Camera Fusion for Waterway Point Cloud Segmentation},
  author={Chen, Shuaijia and Wei, Ping and Zhang, Linyu and Liao, Zhimin and Wang, Bole},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={12617--12621},
  year={2026},
  organization={IEEE}
}
```

## Acknowledgements

This project is developed based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), licensed under the Apache License 2.0.

Many thanks to the open-source repositories:

- [WaterScenes](https://github.com/WaterScenes/WaterScenes)
- [view-of-delft-dataset](https://github.com/tudelft-iv/view-of-delft-dataset)