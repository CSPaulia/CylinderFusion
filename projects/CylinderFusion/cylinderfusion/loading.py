# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional

import os
import mmcv
import torch
import numpy as np
from mmengine.fileio import get

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # TODO: consider split the multi-sweep part out of this pipeline
        # Derive the mask and transform for loading of multi-sweep data
        if self.num_ref_frames > 0:
            # init choice with the current frame
            init_choice = np.array([0], dtype=np.int64)
            num_frames = len(results['img_filename']) // self.num_views - 1
            if num_frames == 0:  # no previous frame, then copy cur frames
                choices = np.random.choice(
                    1, self.num_ref_frames, replace=True)
            elif num_frames >= self.num_ref_frames:
                # NOTE: suppose the info is saved following the order
                # from latest to earlier frames
                if self.test_mode:
                    choices = np.arange(num_frames - self.num_ref_frames,
                                        num_frames) + 1
                # NOTE: +1 is for selecting previous frames
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=False) + 1
            elif num_frames > 0 and num_frames < self.num_ref_frames:
                if self.test_mode:
                    base_choices = np.arange(num_frames) + 1
                    random_choices = np.random.choice(
                        num_frames,
                        self.num_ref_frames - num_frames,
                        replace=True) + 1
                    choices = np.concatenate([base_choices, random_choices])
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=True) + 1
            else:
                raise NotImplementedError
            choices = np.concatenate([init_choice, choices])
            select_filename = []
            for choice in choices:
                select_filename += results['img_filename'][choice *
                                                           self.num_views:
                                                           (choice + 1) *
                                                           self.num_views]
            results['img_filename'] = select_filename
            for key in ['cam2img', 'lidar2cam']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += results[key][choice *
                                                       self.num_views:(choice +
                                                                       1) *
                                                       self.num_views]
                    results[key] = select_results
            for key in ['ego2global']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += [results[key][choice]]
                    results[key] = select_results
            # Transform lidar2cam to
            # [cur_lidar]2[prev_img] and [cur_lidar]2[prev_cam]
            for key in ['lidar2cam']:
                if key in results:
                    # only change matrices of previous frames
                    for choice_idx in range(1, len(choices)):
                        pad_prev_ego2global = np.eye(4)
                        prev_ego2global = results['ego2global'][choice_idx]
                        pad_prev_ego2global[:prev_ego2global.
                                            shape[0], :prev_ego2global.
                                            shape[1]] = prev_ego2global
                        pad_cur_ego2global = np.eye(4)
                        cur_ego2global = results['ego2global'][0]
                        pad_cur_ego2global[:cur_ego2global.
                                           shape[0], :cur_ego2global.
                                           shape[1]] = cur_ego2global
                        cur2prev = np.linalg.inv(pad_prev_ego2global).dot(
                            pad_cur_ego2global)
                        for result_idx in range(choice_idx * self.num_views,
                                                (choice_idx + 1) *
                                                self.num_views):
                            results[key][result_idx] = \
                                results[key][result_idx].dot(cur2prev)
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        filename, cam2img, lidar2cam, cam2lidar, lidar2img = [], [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            lidar2cam_array = np.array(cam_item['lidar2cam']).astype(
                np.float32)
            lidar2cam_rot = lidar2cam_array[:3, :3]
            lidar2cam_trans = lidar2cam_array[:3, 3:4]
            camera2lidar = np.eye(4)
            camera2lidar[:3, :3] = lidar2cam_rot.T
            camera2lidar[:3, 3:4] = -1 * np.matmul(
                lidar2cam_rot.T, lidar2cam_trans.reshape(3, 1))
            cam2lidar.append(camera2lidar)

            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img']).astype(
                np.float32)
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['cam2lidar'] = np.stack(cam2lidar, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(
                img_byte,
                flag=self.color_type,
                backend='pillow',
                channel_order='rgb') for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results
    

@TRANSFORMS.register_module()
class BEVLoadMultiViewEnhancedImageFromFiles(object):

    def __init__(
            self, 
            enhance_image_path=None, 
            to_float32=False,
            color_type: str = 'unchanged',
            backend_args: Optional[dict] = None):
        self.enhance_image_path = enhance_image_path
        self.to_float32 = to_float32
        self.color_type = color_type
        self.backend_args = backend_args

    def __call__(self, results):
        filename = []
        for file_path in results['img_path']:
            name = os.path.basename(file_path)
            filename.append(os.path.join(self.enhance_image_path, name))
        results['img_path'] = filename

        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(
                img_byte,
                flag=self.color_type,
                backend='pillow',
                channel_order='rgb') for img_byte in img_bytes
        ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        return results


@TRANSFORMS.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, dbound, ignore_index=None):
        self.dbound = dbound
        self.ignore_index = ignore_index

    def __call__(self, results):
        points_lidar = results['points']
        if self.ignore_index is not None:
            pts_semantic_masks = results['pts_semantic_mask']

        depth = torch.zeros((len(results['img']), 
                             results['img'][0].shape[0],
                             results['img'][0].shape[1]))
        # depth_coords_set = []

        cur_coords = copy.deepcopy(points_lidar[:, :3].tensor)
        if self.ignore_index is not None:
            cur_coords_semantic_masks = copy.deepcopy(pts_semantic_masks)
            cur_coords_semantic_masks = torch.tensor(cur_coords_semantic_masks)
        cur_img_aug_matrix = np.array(results['img_aug_matrix'])
        cur_img_aug_matrix = torch.tensor(cur_img_aug_matrix).to(cur_coords.device)
        cur_lidar2image = np.array(results['lidar2img'])
        cur_lidar2image = torch.tensor(cur_lidar2image).to(cur_coords.device)

        if self.ignore_index is not None:
            cur_coords = cur_coords[cur_coords_semantic_masks != self.ignore_index]

        # lidar2image
        cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords.transpose(1, 0))
        cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
        # get 2d coords
        dist = cur_coords[:, 2, :]
        cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
        cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

        # imgaug
        cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
        cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
        cur_coords = cur_coords[:, :2, :].transpose(1, 2)

        # normalize coords for grid sample
        cur_coords = cur_coords[..., [1, 0]]

        on_img = ((cur_coords[..., 0] < results['img'][0].shape[0])
                  & (cur_coords[..., 0] >= 0)
                  & (cur_coords[..., 1] < results['img'][0].shape[1])
                  & (cur_coords[..., 1] >= 0)
                  & (dist >= self.dbound[0])
                  & (dist < self.dbound[1]))
        for c in range(on_img.shape[0]):
            masked_coords = cur_coords[c, on_img[c]].long()
            masked_dist = dist[c, on_img[c]]
            depth[c, masked_coords[:, 0],
                  masked_coords[:, 1]] = masked_dist
            # depth_coords_set.append(masked_coords)

        results['gt_depth'] = depth

        # Visualize Depth
        # from mmdet3d.visualization import Det3DLocalVisualizer
        # img = mmcv.imconvert(results['img'][0], 'bgr', 'rgb')
        # depth_coords_set = np.concatenate(depth_coords_set, axis=0)
        # visualizer = Det3DLocalVisualizer()
        # visualizer.set_image(img)
        # visualizer.ax_save.scatter(
        #         depth_coords_set[:, 1],
        #         depth_coords_set[:, 0],
        #         c='y',
        #         s=10,
        #         edgecolors='none')
        # visualizer.show(wait_time=1000)

        # import time
        # time.sleep(1000)

        return results
    

@TRANSFORMS.register_module()
class BEVLoadMultiViewImageAnnotationFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        anno_filename = []
        for _, cam_item in results['images'].items():
            anno_filename.append(cam_item['img_seg_path'])

        results['img_anno_path'] = anno_filename

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_anno_bytes = [
            get(name, backend_args=self.backend_args) for name in anno_filename
        ]
        img_annos = [
            mmcv.imfrombytes(
                img_anno_byte,
                flag=self.color_type,
                backend='pillow') for img_anno_byte in img_anno_bytes
        ]
        # handle the image with different shape
        img_anno_shapes = np.stack([img_anno.shape for img_anno in img_annos], axis=0)
        img_anno_shape_max = np.max(img_anno_shapes, axis=0)
        img_anno_shape_min = np.min(img_anno_shapes, axis=0)
        assert img_anno_shape_min[-1] == img_anno_shape_max[-1]
        if not np.all(img_anno_shape_max == img_anno_shape_min):
            pad_shape = img_anno_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            img_annos = [
                mmcv.impad(img_anno, shape=pad_shape, pad_val=0) for img_anno in img_annos
            ]
        img_anno = np.stack(img_annos, axis=-1)

        results['anno_filename'] = anno_filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['gt_semantic_mask_2d'] = [img_anno[..., i] for i in range(img_anno.shape[-1])]
        results['img_anno_shape'] = img_anno.shape[:2]
        results['anno_ori_shape'] = img_anno.shape[:2]
        # Set initial values for default meta_keys
        results['anno_pad_shape'] = img_anno.shape[:2]
        if self.set_default_scale:
            results['anno_scale_factor'] = 1.0
        return results
    

@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFilesKITTI(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        img_path_prefix = results['lidar_path'].replace(
            'velodyne_reduced', 'image_2').split('/')[0:-1]
        img_path_prefix = '/'.join(img_path_prefix)

        filename, cam2img, lidar2cam, cam2lidar, lidar2img = [], [], [], [], []
        filename.append(img_path_prefix + '/' + results['img_path'])
        lidar2cam.append(results['lidar2cam'])

        lidar2cam_array = np.array(results['lidar2cam']).astype(
            np.float32)
        lidar2cam_rot = lidar2cam_array[:3, :3]
        lidar2cam_trans = lidar2cam_array[:3, 3:4]
        camera2lidar = np.eye(4)
        camera2lidar[:3, :3] = lidar2cam_rot.T
        camera2lidar[:3, 3:4] = -1 * np.matmul(
            lidar2cam_rot.T, lidar2cam_trans.reshape(3, 1))
        cam2lidar.append(camera2lidar)

        cam2img_array = np.eye(4).astype(np.float32)
        cam2img_array[:3, :3] = np.array(results['cam2img'][:3, :3]).astype(
            np.float32)
        cam2img.append(cam2img_array)
        lidar2img.append(cam2img_array @ lidar2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['cam2lidar'] = np.stack(cam2lidar, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(
                img_byte,
                flag=self.color_type,
                backend='pillow',
                channel_order='rgb') for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results