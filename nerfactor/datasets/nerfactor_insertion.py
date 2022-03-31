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

logger = logutil.Logger(loggee="datasets/nerf_shape")


class Dataset(BaseDataset):
    def __init__(self, config, mode, debug=False, always_all_rays=False):
        self.meta2buf = {}
        self.fixed_view = config.getboolean('DEFAULT', 'fixed_view')
        if self.fixed_view:
            self._glob = self._glob_fixed_view
        else:
            self._glob = self._glob_basic

        super().__init__(config, mode, debug=debug, always_all_rays=always_all_rays)

    def _glob_basic(self):
        bg_test_root = self.config.get('DEFAULT', 'data_root')
        shape_root = self.config.get('DEFAULT', 'shape_root')
        # Glob metadata paths
        mode_str = 'val' if self.mode == 'vali' else self.mode
        if self.debug:
            logger.warn("Globbing a single data file for faster debugging")
            metadata_dir = join(bg_test_root, '%s_002' % mode_str)
        else:
            metadata_dir = join(bg_test_root, '%s_???' % mode_str)
        # Include only cameras with all required buffers (depending on mode)
        metadata_paths, incomplete_paths = [], []
        for metadata_path in xm.os.sortglob(metadata_dir, 'metadata.json'):
            id_ = self._parse_id(metadata_path)
            xyz_path = join(shape_root, id_, 'xyz.npy')
            alpha_path = join(shape_root, id_, 'alpha.png')
            depth_path = join(shape_root, id_, 'depth.npy')
            obj_loc_path = join(shape_root, id_, 'obj_loc.npy')
            paths = {'xyz': xyz_path, 'alpha': alpha_path, 'depth': depth_path, 'obj_loc': obj_loc_path}
            if self.mode != 'test':
                rgba_path = join(dirname(metadata_path), 'rgba.png')
                paths['rgba'] = rgba_path
            if ioutil.all_exist(paths):
                metadata_paths.append(metadata_path)
                self.meta2buf[metadata_path] = paths
            else:
                incomplete_paths.append(metadata_path)
        if incomplete_paths:
            logger.warn((
                "Skipping\n\t%s\nbecause at least one of their paired "                "buffers doesn't exist"),
                incomplete_paths)
        logger.info("Number of '%s' views: %d", self.mode, len(metadata_paths))
        return metadata_paths

    def _glob_fixed_view(self):
        shape_root = self.config.get('DEFAULT', 'shape_root')
        # Glob metadata paths
        if self.debug:
            logger.warn("Globbing a single data file for faster debugging")
            metadata_dir = join(shape_root, 'loc_002')
        else:
            metadata_dir = join(shape_root, 'loc_???')
        # Include only cameras with all required buffers (depending on mode)
        metadata_paths, incomplete_paths = [], []
        for metadata_path in xm.os.sortglob(metadata_dir, 'metadata.json'):
            id_ = self._parse_id(metadata_path)
            xyz_path = join(shape_root, id_, 'xyz.npy')
            alpha_path = join(shape_root, id_, 'alpha.png')
            depth_path = join(shape_root, id_, 'depth.npy')
            obj_loc_path = join(shape_root, id_, 'obj_loc.npy')
            paths = {'xyz': xyz_path, 'alpha': alpha_path, 'depth': depth_path, 'obj_loc': obj_loc_path}
            if self.mode != 'test':
                rgba_path = join(dirname(metadata_path), 'rgba.png')
                paths['rgba'] = rgba_path
            if ioutil.all_exist(paths):
                metadata_paths.append(metadata_path)
                self.meta2buf[metadata_path] = paths
            else:
                incomplete_paths.append(metadata_path)
        if incomplete_paths:
            logger.warn((
                "Skipping\n\t%s\nbecause at least one of their paired buffers doesn't exist"), incomplete_paths)
        logger.info("Number of '%s' views: %d", self.mode, len(metadata_paths))
        return metadata_paths

    def _process_example_precache(self, path):
        """Loads data from paths.
        """
        id_, rayo, rayd, radii, viewdir, alpha, xyz, depth, obj_loc = tf.py_function(self._load_data, [path],
                                                                                     (tf.string, tf.float32, tf.float32,
                                                                                      tf.float32, tf.float32,
                                                                                      tf.float32, tf.float32,
                                                                                      tf.float32, tf.float32))
        return id_, rayo, rayd, radii, viewdir, alpha, xyz, depth, obj_loc

    # pylint: disable=arguments-differ
    def _process_example_postcache(self, id_, rayo, rayd, radii, viewdir, alpha, xyz, depth, obj_loc):
        """Records image dimensions and samples rays.
        """
        hw = tf.shape(viewdir)[:2]
        rayo, rayd, radii, viewdir, alpha, xyz, depth = self._sample_rays(rayo, rayd, radii, viewdir, alpha, xyz, depth)

        # NOTE: some memory waste below to make distributed strategy happy
        id_ = tf.tile(tf.expand_dims(id_, axis=0), (tf.shape(viewdir)[0],))
        hw = tf.tile(tf.expand_dims(hw, axis=0), (tf.shape(viewdir)[0], 1))
        return id_, hw, rayo, rayd, radii, viewdir, alpha, xyz, depth, obj_loc

    def _sample_rays(self, rayo, rayd, radii, viewdir, alpha, xyz, depth):
        rayo = tf.reshape(rayo, (-1, 3))
        rayd = tf.reshape(rayd, (-1, 3))
        radii = tf.reshape(radii, (-1, 1))
        viewdir = tf.reshape(viewdir, (-1, 3))
        alpha = tf.reshape(alpha, (-1, 1))
        xyz = tf.reshape(xyz, (-1, 3))
        depth = tf.reshape(depth, (-1, 1))
        return rayo, rayd, radii, viewdir, alpha, xyz, depth

    def _load_data(self, metadata_path):
        imh = self.config.getint('DEFAULT', 'imh')
        metadata_path = tutil.eager_tensor_to_str(metadata_path)
        id_ = self._parse_id(metadata_path)
        # Rays
        metadata = ioutil.read_json(metadata_path)
        imw = int(imh / metadata['imh'] * metadata['imw'])
        cam_to_world = np.array([float(x) for x in metadata['cam_transform_mat'].split(',')]).reshape(4, 4)
        cam_angle_x = metadata['cam_angle_x']
        rayo, rayd, radii, view_dir = self._gen_rays(cam_to_world, cam_angle_x, imh, imw)
        rayo, rayd, radii, view_dir = rayo.astype(np.float32), rayd.astype(np.float32), radii.astype(
            np.float32), view_dir.astype(np.float32)

        # Load precomputed shape properties from vanilla NeRF
        paths = self.meta2buf[metadata_path]
        xyz = ioutil.load_np(paths['xyz'])
        depth = ioutil.load_np(paths['depth'])
        obj_loc = ioutil.load_np(paths['obj_loc'])
        # RGB and alpha, depending on the mode
        alpha = xm.io.img.load(paths['alpha'])
        alpha = xm.img.normalize_uint(alpha)
        # Resize
        if imh != xyz.shape[0]:
            xyz = xm.img.resize(xyz, new_h=imh)
            alpha = xm.img.resize(alpha, new_h=imh)
        # Make sure there's no XYZ coinciding with camera (caused by occupancy
        # accumulating to 0)
        assert not np.isclose(xyz, rayo).all(axis=2).any(), "Found XYZs coinciding with the camera"
        # Re-normalize normals and clip light visibility before returning
        return id_, rayo, rayd, radii, view_dir, alpha, xyz, depth, obj_loc

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

        view_dir = rayd
        view_dir = view_dir / np.linalg.norm(view_dir, axis=-1, keepdims=True)

        dx = np.sqrt(np.sum((rayd[:-1, :, :] - rayd[1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[-2:-1, :]], 0)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / np.sqrt(12)
        if not np.isclose(radii[0, 0], (1 / fl / np.sqrt(3))):
            print('radii error??')
            print(radii[0, 0], (1 / fl / np.sqrt(3)))

        return rayo, rayd, radii, view_dir
