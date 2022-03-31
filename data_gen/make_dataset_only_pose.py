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

from os.path import join
import numpy as np
from tqdm import tqdm
from absl import app

from third_party.xiuminglib import xiuminglib as xm
from data_gen.util import recenter_poses, spherify_poses, manual_poses
from data_gen.util_colmap import render_path_spiral_world, normalize, poses_avg

root_dir = '/data/happily/source/SVL-nerf_data/'


def main(_):
    scene = '0917'
    FF = False
    manual = False
    scene_dir = root_dir + 'raw_data/' + scene

    poses_path = join(scene_dir, 'poses_bounds.npy')
    poses_arr = xm.io.np.read_or_write(poses_path)

    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    poses = np.moveaxis(poses, -1, 0).astype(np.float32)  # Nx3x5
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)  # Nx2

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    c2w = np.concatenate([poses[0, :, :-1], bottom], 0)
    w2c = np.linalg.inv(c2w)

    if manual:
        poses, test_poses, scale = manual_poses(poses)
    elif FF:
        poses = recenter_poses(poses)
        bound_factor = 0.75
        scale = 1. / (bds.min() * bound_factor)
        poses[:, :3, 3] *= scale  # scale translation
        bds *= scale
        down = normalize(poses[:, :3, 1].sum(0))
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz
        # Get radii for spiral path
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w = poses_avg(poses)
        c2w_path = c2w
        N_views = 60
        N_rots = 1
        render_poses = render_path_spiral_world(c2w_path, down, rads, focal, zrate=0.5, rots=N_rots, N=N_views)
    else:
        poses = recenter_poses(poses)
        poses, test_poses, scale = spherify_poses(poses)
        bds *= scale

    points_world = np.array([[0.0866027, 0.783265, 2.27473, 1],
                             [0.864274, 1.58951, -0.248336, 1],
                             [-0.748021, 1.24543, 0.754443, 1],
                             [1.40528, 1.046, 1.42684, 1],
                             [0.419717, 0.633677, 0.866237, 1],
                             [0.419717, 0.633677, 0.866237, 1]])
    points_world[:, -1] = 1.0
    points_world = np.transpose(points_world, (1, 0))
    points_cam = w2c @ points_world
    c2w = np.concatenate([poses[0, :, :-1], bottom], 0)

    points_cam = points_cam * scale
    points_cam[-1, :] = 1.0
    points_new_world = c2w @ points_cam
    print(points_new_world)


if __name__ == '__main__':
    app.run(main)
