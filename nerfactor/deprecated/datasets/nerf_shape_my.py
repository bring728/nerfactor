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

# pylint: disable=invalid-unary-operand-type

from os.path import dirname, join
import numpy as np
import tensorflow as tf

from third_party.xiuminglib import xiuminglib as xm
from nerfactor.util import logging as logutil, io as ioutil, tensor as tutil
from nerfactor.datasets.nerf import Dataset as BaseDataset

logger = logutil.Logger(loggee="datasets/nerf_shape_my")


class Dataset(BaseDataset):
    def __init__(self, config, mode, debug=False, always_all_rays=False):
        self.meta2buf = {}
        self.dict_mask = {}
        super().__init__(config, mode, debug=debug, always_all_rays=always_all_rays)

    def _glob(self):
        root = self.config.get('DEFAULT', 'data_root')
        nerf_root = self.config.get('DEFAULT', 'data_nerf_root')
        # Glob metadata paths
        mode_str = 'val' if self.mode == 'vali' else self.mode
        if self.debug:
            logger.warn("Globbing a single data file for faster debugging")
            metadata_dir = join(root, '%s_002' % mode_str)
        else:
            metadata_dir = join(root, '%s_???' % mode_str)
        # Include only cameras with all required buffers (depending on mode)
        metadata_paths, incomplete_paths = [], []
        for metadata_path in xm.os.sortglob(metadata_dir, 'metadata.json'):
            id_ = self._parse_id(metadata_path)
            lvis_path = join(nerf_root, id_, 'lvis.npy')
            normal_path = join(nerf_root, id_, 'normal.npy')
            xyz_path = join(nerf_root, id_, 'xyz.npy')
            alpha_path = join(nerf_root, id_, 'alpha.png')
            if self.mode == 'test':
                paths = {'xyz': xyz_path, 'normal': normal_path, 'alpha': alpha_path}
            else:
                paths = {'xyz': xyz_path, 'normal': normal_path, 'lvis': lvis_path, 'alpha': alpha_path}

            if self.mode != 'test':
                rgba_path = join(dirname(metadata_path), 'rgba.png')
                paths['rgba'] = rgba_path
            if ioutil.all_exist(paths):
                metadata_paths.append(metadata_path)
                self.meta2buf[metadata_path] = paths
            else:
                incomplete_paths.append(metadata_path)
        if incomplete_paths:
            logger.warn(("Skipping\n\t%s\nbecause at least one of their paired buffers doesn't exist"),
                        incomplete_paths)
        logger.info("Number of '%s' views: %d", self.mode, len(metadata_paths))
        return metadata_paths

    def _process_example_precache(self, path):
        """Loads data from paths.
        """
        id_, hw, rayo, rayd, rgb, alpha, xyz, normal, lvis = tf.py_function(self._load_data, [path],
                                                                            (
                                                                                tf.string, tf.int32, tf.float32,
                                                                                tf.float32, tf.float32, tf.float32,
                                                                                tf.float32, tf.float32, tf.float32))
        return id_, hw, rayo, rayd, rgb, alpha, xyz, normal, lvis

    # pylint: disable=arguments-differ
    def _process_example_postcache(self, id_, hw, rayo, rayd, rgb, alpha, xyz, normal, lvis):
        """Records image dimensions and samples rays.
        """
        rayo, rayd, rgb, alpha, xyz, normal, lvis = self._sample_rays(rayo, rayd, rgb, alpha, xyz, normal, lvis)

        # NOTE: some memory waste below to make distributed strategy happy
        id_ = tf.tile(tf.expand_dims(id_, axis=0), (tf.shape(rgb)[0],))
        hw = tf.tile(tf.expand_dims(hw, axis=0), (tf.shape(rgb)[0], 1))
        return id_, hw, rayo, rayd, rgb, alpha, xyz, normal, lvis

    def _sample_rays(self, rayo, rayd, rgb, alpha, xyz, normal, lvis):
        # Shortcircuit if need all rays
        if self.mode in ('vali', 'test') or self.always_all_rays:
            rayo = tf.reshape(rayo, (-1, 3))
            rayd = tf.reshape(rayd, (-1, 3))
            rgb = tf.reshape(rgb, (-1, 3))
            alpha = tf.reshape(alpha, (-1, 1))
            xyz = tf.reshape(xyz, (-1, 3))
            normal = tf.reshape(normal, (-1, 3))
            lvis = tf.reshape(lvis, (-1, tf.shape(lvis)[1]))
            return rayo, rayd, rgb, alpha, xyz, normal, lvis
        # Training: sample rays
        select_ind = tf.random.uniform((self.bs,), minval=0, maxval=tf.shape(alpha)[0], dtype=tf.int32)

        rayo = tf.gather(rayo, select_ind)
        rayd = tf.gather(rayd, select_ind)
        rgb = tf.gather(rgb, select_ind)
        alpha = tf.gather(alpha, select_ind)
        alpha = tf.reshape(alpha, (-1, 1))
        xyz = tf.gather(xyz, select_ind)
        normal = tf.gather(normal, select_ind)
        lvis = tf.gather(lvis, select_ind)
        return rayo, rayd, rgb, alpha, xyz, normal, lvis

    # pylint: disable=arguments-differ
    def _load_data(self, metadata_path, alpha_thres=0.9):
        imh = self.config.getint('DEFAULT', 'imh')
        use_nerf_alpha = self.config.getboolean('DEFAULT', 'use_nerf_alpha')
        metadata_path = tutil.eager_tensor_to_str(metadata_path)
        id_ = self._parse_id(metadata_path)
        # Rays
        metadata = ioutil.read_json(metadata_path)
        imw = int(imh / metadata['imh'] * metadata['imw'])
        cam_to_world = np.array([float(x) for x in metadata['cam_transform_mat'].split(',')]).reshape(4, 4)
        cam_angle_x = metadata['cam_angle_x']
        rayo, rayd = self._gen_rays(cam_to_world, cam_angle_x, imh, imw)
        rayo, rayd = rayo.astype(np.float32), rayd.astype(np.float32)
        # Load precomputed shape properties from vanilla NeRF
        paths = self.meta2buf[metadata_path]
        xyz = ioutil.load_np(paths['xyz'])
        normal = ioutil.load_np(paths['normal'])

        if self.debug:
            logger.warn("Faking light visibility for faster debugging")
            lvis = 0.5 * np.ones(normal.shape[:2] + (512,), dtype=np.float32)
        else:
            if self.mode == 'test':
                lvis = 0.5 * np.ones(normal.shape[:2] + (512,), dtype=np.float32)
            else:
                lvis = ioutil.load_np(paths['lvis'])
        # RGB and alpha, depending on the mode
        if self.mode == 'test':
            # No RGBA, so NeRF-traced alpha and placeholder RGB
            alpha = xm.io.img.load(paths['alpha'])
            alpha = xm.img.normalize_uint(alpha)
            rgb = np.zeros_like(xyz)
        else:
            # Training or validation, where each camera has a paired image
            rgba = xm.io.img.load(paths['rgba'])
            assert rgba.ndim == 3 and rgba.shape[2] == 4, \
                "Input image is not RGBA"
            rgba = xm.img.normalize_uint(rgba)
            rgb = rgba[:, :, :3]
            if use_nerf_alpha:  # useful for real scenes
                alpha = xm.io.img.load(paths['alpha'])
                alpha = xm.img.normalize_uint(alpha)
            else:
                alpha = rgba[:, :, 3]  # ground-truth alpha
        # Resize
        if imh != xyz.shape[0]:
            xyz = xm.img.resize(xyz, new_h=imh)
            normal = xm.img.resize(normal, new_h=imh)
            lvis = xm.img.resize(lvis, new_h=imh)
            alpha = xm.img.resize(alpha, new_h=imh)
            rgb = xm.img.resize(rgb, new_h=imh)
        # Make sure there's no XYZ coinciding with camera (caused by occupancy
        # accumulating to 0)
        assert not np.isclose(xyz, rayo).all(axis=2).any(), "Found XYZs coinciding with the camera"
        # Re-normalize normals and clip light visibility before returning
        normal = xm.linalg.normalize(normal, axis=2)
        assert np.isclose(np.linalg.norm(normal, axis=2), 1).all(), \
            "Found normals with a norm far away from 1"
        lvis = np.clip(lvis, 0, 1)

        hw = (imh, imw)

        alpha_flat = alpha.reshape((-1))
        mask = np.zeros_like(alpha_flat, dtype=np.bool)
        mask[alpha_flat > alpha_thres] = True
        self.dict_mask[id_] = mask

        rayo = tf.boolean_mask(rayo, alpha > alpha_thres)
        rayd = tf.boolean_mask(rayd, alpha > alpha_thres)
        rgb = tf.boolean_mask(rgb, alpha > alpha_thres)
        rgb = tf.cast(rgb, tf.float32)
        xyz = tf.boolean_mask(xyz, alpha > alpha_thres)
        normal = tf.boolean_mask(normal, alpha > alpha_thres)
        lvis = tf.boolean_mask(lvis, alpha > alpha_thres)
        alpha = tf.boolean_mask(alpha, alpha > alpha_thres)
        alpha = tf.cast(alpha, tf.float32)

        return id_, hw, rayo, rayd, rgb, alpha, xyz, normal, lvis
