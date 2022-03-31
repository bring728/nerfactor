# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import basename, dirname, join, exists
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

from third_party.xiuminglib import xiuminglib as xm
from nerfactor.util import logging as logutil, io as ioutil, tensor as tutil, img as imgutil
from nerfactor.datasets.base import Dataset as BaseDataset

logger = logutil.Logger(loggee="datasets/nerf")


class Dataset(BaseDataset):
    def __init__(self, config, mode, debug=False, always_all_rays=False, spp=1):
        self.meta2img = {}
        self.dict_mask = {}
        # To allow supersampling a pixel
        sps = np.sqrt(spp)  # samples per side
        assert sps == int(sps), "Samples per pixel must be a square number so that samples per side are integers"
        self.sps = int(sps)
        # Parent init.
        super().__init__(config, mode, debug=debug)
        # Trigger init. in a main thread before starting multi-threaded work.
        # See http://yaqs/eng/q/6292200559345664 for details
        Image.init()
        # To allow getting all rays for training images
        self.always_all_rays = always_all_rays

    def get_n_views(self):
        if hasattr(self, 'files'):
            return len(self.files)
        raise RuntimeError("Call `_glob()` before `get_n_views()`")

    def _get_batch_size(self):
        if self.mode == 'train':
            bs = self.config.getint('DEFAULT', 'n_rays_per_step')
        else:
            # Total number of pixels is batch size, and will need to load
            # a datapoint to figure that out
            any_path = self.files[0]
            ret = self._load_data(any_path)
            # self._process_example_postcache(*ret)
            map_data = ret[-3]  # OK as long as shape is (H, W[, ?])
            bs = int(map_data.shape[0])
        return bs

    def _glob(self):
        root = self.config.get('DEFAULT', 'data_root')
        if self.mode in ('train', 'test'):
            mode_str = self.mode
        else:
            mode_str = 'val'
        metadata_dir = join(root, '%s_???' % mode_str)
        # Shortcircuit if testing
        if self.mode == 'test':
            metadata_paths = xm.os.sortglob(metadata_dir, 'metadata.json')
            logger.info("Number of '%s' views: %d", self.mode, len(metadata_paths))
            return metadata_paths
        # Training or validation
        # Include only cameras with paired RGB images
        metadata_paths = []
        for metadata_path in xm.os.sortglob(metadata_dir, 'metadata.json'):
            img_path = join(dirname(metadata_path), 'rgba.png')
            if exists(img_path):
                metadata_paths.append(metadata_path)
                self.meta2img[metadata_path] = img_path
            else:
                logger.warning(("Skipping camera\n\t%s\nbecause its paried RGB image\n\t%s\ndoesn't exist"),
                               metadata_path, img_path)
        logger.info("Number of '%s' views: %d", self.mode, len(metadata_paths))
        return metadata_paths

    @staticmethod
    def _parse_id(metadata_path):  # pylint: disable=arguments-differ
        return basename(dirname(metadata_path))

    def _process_example_precache(self, path):
        """Loads data from paths.
        """
        id_, hw, rayo, rayd, rgb, radii, tmin, tmax, local_mat, local_tr = tf.py_function(self._load_data, [path],
                                                                                          (tf.string, tf.int32,
                                                                                           tf.float32, tf.float32,
                                                                                           tf.float32, tf.float32,
                                                                                           tf.float32, tf.float32,
                                                                                           tf.float32, tf.float32))
        return id_, hw, rayo, rayd, rgb, radii, tmin, tmax, local_mat, local_tr

    # pylint: disable=arguments-differ
    def _process_example_postcache(self, id_, hw, rayo, rayd, rgb, radii, tmin, tmax, local_mat, local_tr):
        """Records image dimensions and samples rays.
        """
        rayo, rayd, rgb, radii, tmin, tmax = self._sample_rays(rayo, rayd, rgb, radii, tmin, tmax)
        # NOTE: some memory waste below to make distributed strategy happy
        id_ = tf.tile(tf.expand_dims(id_, axis=0), (tf.shape(rgb)[0],))
        hw = tf.tile(tf.expand_dims(hw, axis=0), (tf.shape(rgb)[0], 1))
        local_mat = tf.tile(tf.expand_dims(local_mat, axis=0), (tf.shape(rgb)[0], 1, 1))
        local_tr = tf.tile(tf.expand_dims(local_tr, axis=0), (tf.shape(rgb)[0], 1))
        return id_, hw, rayo, rayd, rgb, radii, tmin, tmax, local_mat, local_tr

    def _sample_rays(self, rayo, rayd, rgb, radii, tmin, tmax):
        # Shortcircuit if need all rays
        if self.mode in ('vali', 'test') or self.always_all_rays:
            return rayo, rayd, rgb, radii, tmin, tmax
        # Training: sample rays
        # coords = tf.stack(tf.meshgrid(tf.range(tf.shape(rgb)[0]), tf.range(tf.shape(rgb)[1]), indexing='ij'), axis=-1)
        # coords = tf.reshape(coords, (-1, 2))
        # Use tf.random instead of np.random here so that the randomness is
        # correct even if we compile this to static graph using tf.function
        select_ind = tf.random.uniform((self.bs,), minval=0, maxval=tf.shape(rayo)[0], dtype=tf.int32)
        rayo = tf.gather(rayo, select_ind)
        rayd = tf.gather(rayd, select_ind)
        rgb = tf.gather(rgb, select_ind)
        tmin = tf.gather(tmin, select_ind)
        tmax = tf.gather(tmax, select_ind)
        radii = tf.gather(radii, select_ind)
        return rayo, rayd, rgb, radii, tmin, tmax

    def ray_intersect_aabb(self, rayo, rayd, aabb):
        direction = rayd
        dir_fraction = np.ones_like(rayd)
        dir_fraction[direction == 0.0] = np.inf
        dir_fraction[direction != 0.0] = np.divide(1.0, direction[direction != 0.0])

        t1 = (aabb[0] - rayo[:, 0]) * dir_fraction[:, 0]
        t2 = (aabb[3] - rayo[:, 0]) * dir_fraction[:, 0]
        t3 = (aabb[1] - rayo[:, 1]) * dir_fraction[:, 1]
        t4 = (aabb[4] - rayo[:, 1]) * dir_fraction[:, 1]
        t5 = (aabb[2] - rayo[:, 2]) * dir_fraction[:, 2]
        t6 = (aabb[5] - rayo[:, 2]) * dir_fraction[:, 2]

        tmin = np.maximum(np.maximum(np.minimum(t1, t2), np.minimum(t3, t4)), np.minimum(t5, t6))
        tmax = np.minimum(np.minimum(np.maximum(t1, t2), np.maximum(t3, t4)), np.maximum(t5, t6))

        mask = np.ones_like(tmax, dtype=np.bool)
        mask[tmax < 0] = False
        # ray_mask[tmin < 1] = False
        mask[tmin > tmax] = False

        tmin = tmin[mask]
        tmax = tmax[mask]

        return tmin, tmax, mask

    def _load_data(self, metadata_path):  # pylint: disable=arguments-differ
        imh = self.config.getint('DEFAULT', 'imh')
        white_bg = self.config.getboolean('DEFAULT', 'white_bg')
        metadata_path = tutil.eager_tensor_to_str(metadata_path)
        id_ = self._parse_id(metadata_path)
        # Generate rays
        metadata = ioutil.read_json(metadata_path)
        imw = int(imh / metadata['imh'] * metadata['imw'])
        cam_to_world = np.array([float(x) for x in metadata['cam_transform_mat'].split(',')]).reshape(4, 4)
        cam_angle_x = metadata['cam_angle_x']
        local_trfm = metadata['Object local trfm']
        local_mat = np.array([[local_trfm[0], 0, 0], [0, local_trfm[1], 0], [0, 0, local_trfm[2]]])
        local_tr = np.array([local_trfm[3], local_trfm[4], local_trfm[5]])

        obj_3d_bb = np.array(metadata['Object 3D BB']).astype(np.float32).reshape([6])
        # obj_2d_pos = np.array(metadata['Object 2D pos']).astype(np.float32)
        # mask = np.zeros((imh, imw)).astype(np.float32)
        # cv2.drawContours(mask, [cv2.convexHull(obj_2d_pos.T.astype(np.int32))], 0, (1), 5)  ##일부로 약간 크게 잡음.
        # cv2.drawContours(mask, [cv2.convexHull(obj_2d_pos.T.astype(np.int32))], 0, (1), -1)
        rayo, rayd, radii = self._gen_rays(cam_to_world, cam_angle_x, imh, imw)
        rayo, rayd, radii = rayo.astype(np.float32), rayd.astype(np.float32), radii.astype(np.float32)

        rayo = np.reshape(rayo, (-1, 3))
        rayd = np.reshape(rayd, (-1, 3))
        radii = np.reshape(radii, (-1, 1))
        tmin, tmax, mask = self.ray_intersect_aabb(rayo, rayd, obj_3d_bb)
        rayo = rayo[mask]
        rayd = rayd[mask]
        radii = radii[mask]
        self.dict_mask[id_] = mask
        # np.matmul(rayo + rayd * tmax[:, None], local_mat) + local_tr

        hw = (imh, imw)
        if self.mode == 'test':
            rgb = np.zeros_like(rayd, dtype=np.float32)  # placeholder
            return id_, hw, rayo, rayd, rgb, radii, tmin, tmax, local_mat, local_tr

        # Training or validation, where each camera has a paired image
        img_path = self.meta2img[metadata_path]
        rgba = xm.io.img.load(img_path)
        assert rgba.ndim == 3 and rgba.shape[2] == 4, "Input image is not RGBA"
        rgba = xm.img.normalize_uint(rgba)
        # Resize RGB
        if imh != rgba.shape[0]:
            rgba = xm.img.resize(rgba, new_h=imh)
        rgb, alpha = rgba[:, :, :3], rgba[:, :, 3]
        # Composite RGBA image onto white or black background
        bg = np.ones_like(rgb) if white_bg else np.zeros_like(rgb)
        rgb = imgutil.alpha_blend(rgb, alpha, tensor2=bg)
        rgb = rgb.astype(np.float32)

        rgb = np.reshape(rgb, (-1, 3))
        rgb = rgb[mask]
        return id_, hw, rayo, rayd, rgb, radii, tmin, tmax, local_mat, local_tr

    # pylint: disable=arguments-differ
    def _gen_rays(self, to_world, angle_x, imh, imw):
        # Ray origin
        cam_loc = to_world[:3, 3]
        rayo = np.tile(cam_loc[None, None, :], (imh * self.sps, imw * self.sps, 1))  # (H * SPS, W * SPS, 3)
        # Ray directions
        xs = np.linspace(0, imw, imw * self.sps, endpoint=False) + 0.5
        ys = np.linspace(0, imh, imh * self.sps, endpoint=False) + 0.5
        xs, ys = np.meshgrid(xs, ys)
        # (0, 0)
        # +--------> (w, 0)
        # |           x
        # |
        # v y (0, h)
        fl = .5 * imw / np.tan(.5 * angle_x)
        rayd = np.stack(((xs - .5 * imw) / fl, (ys - .5 * imh) / fl, np.ones_like(xs)), axis=-1)  # local
        rayd = np.sum(rayd[:, :, np.newaxis, :] * to_world[:3, :3], axis=-1)  # world

        dx = np.sqrt(np.sum((rayd[:-1, :, :] - rayd[1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[-2:-1, :]], 0)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / np.sqrt(12)
        if not np.isclose(radii[0, 0], (1 / fl / np.sqrt(3))):
            print('radii error??')
            print(radii[0, 0], (1 / fl / np.sqrt(3)))

        return rayo, rayd, radii
