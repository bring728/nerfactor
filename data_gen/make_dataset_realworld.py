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

from third_party.xiuminglib import xiuminglib as xm
from data_gen.util import recenter_poses, spherify_poses, manual_poses
from data_gen.util_colmap import render_path_spiral_world, normalize, poses_avg

root_dir = '/data/happily/source/SVL-nerf_data/'


def main():
    debug = False
    scene = '0924-1'
    FF = False
    if scene == '0924-1':
        manual = True
    else:
        manual = False
    scene_dir = root_dir + 'raw_data/' + scene
    view_folder = '{mode}_{i:03d}'

    resize_h = 1008
    factor = 4
    # resize_h = 756
    # resize_h = 1008
    auto_rotate = True
    outroot = root_dir + 'real_scene/' + scene

    # ------ Training and validation
    # Load poses
    poses_path = join(scene_dir, 'poses_bounds.npy')
    poses_arr = xm.io.np.read_or_write(poses_path)
    # poses_arr = np.concatenate([poses_arr[:41], poses_arr[42:]], axis=0)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])

    bds = poses_arr[:, -2:].transpose([1, 0])

    # Load and resize images
    img_dir = join(scene_dir, 'images')
    img_paths = xm.os.sortglob(img_dir, filename='*', ext='jpg', ext_ignore_case=True)
    assert img_paths, "No image globbed"
    if debug:
        img_paths = img_paths[:4]
        poses = poses[..., :4]
        bds = bds[..., :4]
    imgs = []
    # factor = None
    for img_file in tqdm(img_paths, desc="Loading images"):
        img = xm.io.img.read(img_file, auto_rotate)
        img = xm.img.normalize_uint(img)
        if factor is None:
            factor = float(img.shape[0]) / resize_h
        else:
            if float(img.shape[0]) / resize_h != factor:
                print("Images are of varying sizes")
                continue
            # assert float(img.shape[0]) / resize_h == factor, "Images are of varying sizes"
        img = xm.img.resize(img, new_h=resize_h, method='tf')
        if img.shape[2] == 3:
            # NOTE: add an all-one alpha
            img = np.dstack((img, np.ones_like(img)[:, :, :1]))
        imgs.append(img)
    imgs = np.stack(imgs, axis=-1)

    # Sanity check
    n_poses = poses.shape[-1]
    n_imgs = imgs.shape[-1]
    assert n_poses == n_imgs, ("Mismatch between numbers of images ({n_imgs}) and "
                               "poses ({n_poses})").format(n_imgs=n_imgs, n_poses=n_poses)

    # Update poses according to downsampling
    poses[:2, 4, :] = np.array(imgs.shape[:2]).reshape([2, 1])  # override image size
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor  # scale focal length

    poses = np.moveaxis(poses, -1, 0).astype(np.float32)  # Nx3x5
    imgs = np.moveaxis(imgs, -1, 0)  # NxHxWx4
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)  # Nx2

    if manual:
        poses, test_poses, scale = manual_poses(poses)
    elif FF:
        poses = recenter_poses(poses)
        bound_factor = 0.75
        scale = 1. / (bds.min() * bound_factor)
        poses[:, :3, 3] *= scale  # scale translation
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
        test_poses = render_path_spiral_world(c2w_path, down, rads, focal, zrate=0.5, rots=N_rots, N=N_views)
    else:
        poses = recenter_poses(poses)
        poses, test_poses, scale = spherify_poses(poses)

    bds *= scale
    print(bds)
    print(bds[:, 0].min())
    print(bds[:, 1].max())

    train_json = join(outroot, 'transforms_train.json')
    vali_json = join(outroot, 'transforms_val.json')
    test_json = join(outroot, 'transforms_test.json')

    # Training-validation split
    # ind_vali = np.arange(n_imgs)[:-1:(n_imgs // n_vali)]
    ind_vali = np.array([50, 70])
    ind_train = np.array([x for x in np.arange(n_imgs) if x not in ind_vali])

    # Training frames
    # train_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    train_meta = {'frames': []}
    for vi, i in enumerate(ind_train):
        view_folder_ = view_folder.format(mode='train', i=vi)
        # Write image
        img = imgs[i, :, :, :]
        xm.io.img.write_float(img, join(outroot, view_folder_, 'rgba.png'), clip=True)
        # Record metadata
        pose = poses[i, :, :]
        fl = pose[-1, -1]
        cam_angle_x = np.arctan2(imgs.shape[2] / 2, fl) * 2
        train_meta['camera_angle_x'] = cam_angle_x

        c2w = np.vstack((pose[:3, :4], np.array([0, 0, 0, 1]).reshape(1, 4)))
        frame_meta = {'file_path': './%s/rgba' % view_folder_, 'rotation': 0,
                      'transform_matrix': c2w.tolist()}
        train_meta['frames'].append(frame_meta)
        # Write this frame's metadata to the view folder
        frame_meta = {
            'cam_angle_x': cam_angle_x,
            'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
            'envmap': '', 'envmap_inten': 0, 'imh': img.shape[0],
            'imw': img.shape[1], 'scene': '', 'spp': 0,
            'original_path': img_paths[i]}
        xm.io.json.write(frame_meta, join(outroot, view_folder_, 'metadata.json'))

    # Validation views
    vali_meta = {'frames': []}
    for vi, i in enumerate(ind_vali):
        view_folder_ = view_folder.format(mode='val', i=vi)
        # Write image
        img = imgs[i, :, :, :]
        xm.io.img.write_float(img, join(outroot, view_folder_, 'rgba.png'), clip=True)
        # Record metadata
        pose = poses[i, :, :]
        fl = pose[-1, -1]
        cam_angle_x = np.arctan2(imgs.shape[2] / 2, fl) * 2
        vali_meta['camera_angle_x'] = cam_angle_x

        c2w = np.vstack((pose[:3, :4], np.array([0, 0, 0, 1]).reshape(1, 4)))
        frame_meta = {'file_path': './%s/rgba' % view_folder_, 'rotation': 0,
                      'transform_matrix': c2w.tolist()}
        vali_meta['frames'].append(frame_meta)
        # Write this frame's metadata to the view folder
        frame_meta = {
            'cam_angle_x': cam_angle_x,
            'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
            'envmap': '', 'envmap_inten': 0, 'imh': img.shape[0],
            'imw': img.shape[1], 'scene': '', 'spp': 0,
            'original_path': img_paths[i]}
        xm.io.json.write(frame_meta, join(outroot, view_folder_, 'metadata.json'))

    # Write training and validation JSONs
    xm.io.json.write(train_meta, train_json)
    xm.io.json.write(vali_meta, vali_json)

    test_meta = {'frames': []}
    for i in range(test_poses.shape[0]):
        view_folder_ = view_folder.format(mode='test', i=i)
        # Record metadata
        pose = test_poses[i, :, :]
        fl = pose[-1, -1]
        cam_angle_x = np.arctan2(imgs.shape[2] / 2, fl) * 2
        vali_meta['camera_angle_x'] = cam_angle_x

        c2w = np.vstack((pose[:3, :4], np.array([0, 0, 0, 1]).reshape(1, 4)))
        frame_meta = {'file_path': '', 'rotation': 0, 'transform_matrix': c2w.tolist()}
        test_meta['frames'].append(frame_meta)
        # Write the nearest input to this test view folder
        dist = np.linalg.norm(pose[:, 3] - poses[:, :, 3], axis=1)
        nn_i = np.argmin(dist)
        nn_img = imgs[nn_i, :, :, :]
        xm.io.img.write_float(nn_img, join(outroot, view_folder_, 'nn.png'), clip=True)
        # Write this frame's metadata to the view folder
        frame_meta = {
            'cam_angle_x': cam_angle_x,
            'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
            'envmap': '', 'envmap_inten': 0, 'imh': img.shape[0],
            'imw': img.shape[1], 'scene': '', 'spp': 0, 'original_path': ''}
        xm.io.json.write(frame_meta, join(outroot, view_folder_, 'metadata.json'))

    # Write JSON
    xm.io.json.write(test_meta, test_json)
    print('making dataset is done.')


if __name__ == '__main__':
    main()
