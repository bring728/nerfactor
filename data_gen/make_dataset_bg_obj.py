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
from absl import app, flags
import json

from third_party.xiuminglib import xiuminglib as xm
from data_gen.util_colmap import render_path_spiral_world, normalize, poses_avg

scene_dir = '/home/vig-titan2/Data/SVL-nerf_data/data created by myself/simple_preprocess'
bg_out_root = '/home/vig-titan2/Data/SVL-nerf_data/synthetic_scene/bg_simple'
obj_out_root = '/home/vig-titan2/Data/SVL-nerf_data/synthetic_scene/obj_simple'
scene_out_root = '/home/vig-titan2/Data/SVL-nerf_data/synthetic_scene/scene_simple'
flags.DEFINE_string('scene_dir', scene_dir, "scene directory")
flags.DEFINE_integer('n_vali', 2, "number of held-out validation views")
flags.DEFINE_float('bound_factor', .75, "bound factor")
flags.DEFINE_string('bg_outroot', bg_out_root, "output root")
flags.DEFINE_string('obj_outroot', obj_out_root, "output root")
flags.DEFINE_string('scene_outroot', scene_out_root, "output root")
flags.DEFINE_boolean('debug', False, "debug toggle")
FLAGS = flags.FLAGS


def main(_):
    view_folder = '{mode}_{i:03d}'

    # Only the original NeRF and JaxNeRF implementations need these
    bg_train_json = join(FLAGS.bg_outroot, 'transforms_train.json')
    obj_train_json = join(FLAGS.obj_outroot, 'transforms_train.json')
    scene_train_json = join(FLAGS.scene_outroot, 'transforms_train.json')
    bg_vali_json = join(FLAGS.bg_outroot, 'transforms_val.json')
    obj_vali_json = join(FLAGS.obj_outroot, 'transforms_val.json')
    scene_vali_json = join(FLAGS.scene_outroot, 'transforms_val.json')
    bg_test_json = join(FLAGS.bg_outroot, 'transforms_test.json')
    obj_test_json = join(FLAGS.obj_outroot, 'transforms_test.json')
    scene_test_json = join(FLAGS.scene_outroot, 'transforms_test.json')

    # ------ Training and validation

    # Load poses
    poses_path = join(FLAGS.scene_dir, 'poses_bounds.npy')
    poses_arr = xm.io.np.read_or_write(poses_path)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    # Load and resize images
    # img_dir = join(FLAGS.scene_dir, 'images')
    bg_img_paths = xm.os.sortglob(scene_dir, filename='bg_*', ext='png', ext_ignore_case=True)
    bg_imgs = []
    for img_file in tqdm(bg_img_paths, desc="Loading images"):
        bg_img = xm.io.img.read(img_file)
        bg_img = xm.img.normalize_uint(bg_img)
        if bg_img.shape[2] == 3:
            # NOTE: add an all-one alpha
            bg_img = np.dstack((bg_img, np.ones_like(bg_img)[:, :, :1]))
        bg_imgs.append(bg_img)
    bg_imgs = np.stack(bg_imgs, axis=-1)

    obj_img_paths = xm.os.sortglob(scene_dir, filename='obj_*', ext='png', ext_ignore_case=True)
    obj_imgs = []
    for img_file in tqdm(obj_img_paths, desc="Loading images"):
        obj_img = xm.io.img.read(img_file)
        obj_img = xm.img.normalize_uint(obj_img)
        if obj_img.shape[2] == 3:
            # NOTE: add an all-one alpha
            obj_img = np.dstack((obj_img, np.ones_like(obj_img)[:, :, :1]))
        obj_imgs.append(obj_img)
    obj_imgs = np.stack(obj_imgs, axis=-1)

    # Sanity check
    n_poses = poses.shape[-1]
    n_imgs = bg_imgs.shape[-1]
    n_imgs2 = obj_imgs.shape[-1]
    assert n_poses == n_imgs and n_imgs == n_imgs2, ("Mismatch between numbers of images ({n_imgs}) and "
                                                     "poses ({n_poses})").format(n_imgs=n_imgs, n_poses=n_poses)

    # Correct rotation matrix ordering and move variable dim to axis 0
    # poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)  # Nx3x5
    bg_imgs = np.moveaxis(bg_imgs, -1, 0)  # NxHxWx4
    obj_imgs = np.moveaxis(obj_imgs, -1, 0)  # NxHxWx4
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)  # Nx2

    # Rescale according to a default bd factor
    scale = 1. / (bds.min() * FLAGS.bound_factor)
    poses[:, :3, 3] *= scale  # scale translation
    bds *= scale

    # Recenter poses
    # poses = recenter_poses(poses)

    # Generate a spiral/spherical path for rendering videos
    # poses, test_poses = spherify_poses(poses)

    # Training-validation split
    # ind_vali = np.arange(n_imgs)[:-1:(n_imgs // FLAGS.n_vali)]
    # ind_train = np.array([x for x in np.arange(n_imgs) if x not in ind_vali])
    ind_train = np.array([0, 2, 4, 6, 14, 16, 18, 20, 28, 30, 32, 34, 42, 44, 46, 48])
    ind_vali = np.delete(np.arange(49), ind_train)

    # Figure out camera angle
    fl = poses[0, -1, -1]
    cam_angle_x = np.arctan2(bg_imgs.shape[2] / 2, fl) * 2

    scene_json = xm.io.json.load(scene_dir + '/scene_data.json')
    obj_3d_BB = np.array(scene_json['scene body'][0]['Object 3D BB']) * scale
    obj_3d_rot = np.array(scene_json['scene body'][0]['Object Rotation'])

    # tr mat for world to obj local coordinate
    pt_1 = np.array([obj_3d_BB[0, 0], obj_3d_BB[0, 1], obj_3d_BB[0, 2]], dtype=np.float32)
    pt_2 = np.array([obj_3d_BB[1, 0], obj_3d_BB[1, 1], obj_3d_BB[1, 2]], dtype=np.float32)
    tmp_1 = np.array(
        [[pt_1[0], 0, 0, 1, 0, 0], [0, pt_1[1], 0, 0, 1, 0], [0, 0, pt_1[2], 0, 0, 1], [pt_2[0], 0, 0, 1, 0, 0],
         [0, pt_2[1], 0, 0, 1, 0], [0, 0, pt_2[2], 0, 0, 1]], dtype=np.float32)
    tmp_2 = np.array([-1, -1, -1, 1, 1, 1])
    obj_transform = np.linalg.inv(tmp_1) @ tmp_2

    # Training frames
    bg_train_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    scene_train_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    for vi, i in enumerate(ind_train):
        bg_view_folder = view_folder.format(mode='train', i=vi)
        # Write image
        bg_img = bg_imgs[i, :, :, :]
        obj_img = obj_imgs[i, :, :, :]
        xm.io.img.write_float(bg_img, join(FLAGS.bg_outroot, bg_view_folder, 'rgba.png'), clip=True)
        xm.io.img.write_float(obj_img, join(FLAGS.scene_outroot, bg_view_folder, 'rgba.png'), clip=True)

        # Record metadata
        pose = poses[i, :, :]
        c2w = np.vstack((pose[:3, :4], np.array([0, 0, 0, 1]).reshape(1, 4)))
        frame_meta = {'file_path': './%s/rgba' % bg_view_folder, 'rotation': 0,
                      'transform_matrix': c2w.tolist()}
        bg_train_meta['frames'].append(frame_meta)
        scene_train_meta['frames'].append(frame_meta)
        # Write this frame's metadata to the view folder
        frame_meta = {
            'cam_angle_x': cam_angle_x,
            'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
            'envmap': '', 'envmap_inten': 0, 'imh': bg_img.shape[0],
            'imw': bg_img.shape[1], 'scene': '', 'spp': 0,
            'original_path': bg_img_paths[i]}
        xm.io.json.write(frame_meta, join(FLAGS.bg_outroot, bg_view_folder, 'metadata.json'))
        xm.io.json.write(frame_meta, join(FLAGS.scene_outroot, bg_view_folder, 'metadata.json'))

    obj_body = scene_json['scene body'][0]['obj body']
    obj_train_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    flag = True
    for vi, i in enumerate(ind_train):
        for obj in obj_body:
            id = int(obj['image name'][-7:-4])
            if id == i:
                flag = False
                obj_pos_2d = np.array(obj['Object 2D boundary'])
                obj_proj = np.array(obj['c2cc proj mat'])
                break
        if flag:
            print('not matching image and index~!')
            return

        obj_view_folder = view_folder.format(mode='train', i=vi)
        # Write image
        obj_img = obj_imgs[i, :, :, :]
        xm.io.img.write_float(obj_img, join(FLAGS.obj_outroot, obj_view_folder, 'rgba.png'), clip=True)
        # Record metadata
        pose = poses[i, :, :]
        # print(pose[:3, 3] - (obj_proj * scale)[:3, 3])
        c2w = np.vstack((pose[:3, :4], np.array([0, 0, 0, 1]).reshape(1, 4)))
        frame_meta = {'file_path': './%s/rgba' % obj_view_folder, 'rotation': 0,
                      'transform_matrix': c2w.tolist()}
        obj_train_meta['frames'].append(frame_meta)
        # Write this frame's metadata to the view folder
        frame_meta = {
            'cam_angle_x': cam_angle_x,
            'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
            'envmap': '', 'envmap_inten': 0, 'imh': obj_img.shape[0],
            'imw': obj_img.shape[1], 'scene': '', 'spp': 0,
            'original_path': obj_img_paths[i],
            'Object 3D BB': obj_3d_BB.tolist(), 'Object 2D pos': obj_pos_2d.tolist(),
            'Object local trfm': obj_transform.tolist(),
        }
        xm.io.json.write(frame_meta, join(FLAGS.obj_outroot, obj_view_folder, 'metadata.json'))

    # Validation views
    bg_vali_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    scene_vali_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    for vi, i in enumerate(ind_vali):
        bg_view_folder = view_folder.format(mode='val', i=vi)
        # Write image
        bg_img = bg_imgs[i, :, :, :]
        obj_img = obj_imgs[i, :, :, :]
        xm.io.img.write_float(bg_img, join(FLAGS.bg_outroot, bg_view_folder, 'rgba.png'), clip=True)
        xm.io.img.write_float(obj_img, join(FLAGS.scene_outroot, bg_view_folder, 'rgba.png'), clip=True)
        # Record metadata
        pose = poses[i, :, :]
        c2w = np.vstack((pose[:3, :4], np.array([0, 0, 0, 1]).reshape(1, 4)))
        frame_meta = {'file_path': './%s/rgba' % bg_view_folder, 'rotation': 0,
                      'transform_matrix': c2w.tolist()}
        bg_vali_meta['frames'].append(frame_meta)
        scene_vali_meta['frames'].append(frame_meta)
        # Write this frame's metadata to the view folder
        frame_meta = {'cam_angle_x': cam_angle_x,
                      'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
                      'envmap': '', 'envmap_inten': 0, 'imh': bg_img.shape[0],
                      'imw': bg_img.shape[1], 'scene': '', 'spp': 0,
                      'original_path': bg_img_paths[i]}
        xm.io.json.write(frame_meta, join(FLAGS.bg_outroot, bg_view_folder, 'metadata.json'))
        xm.io.json.write(frame_meta, join(FLAGS.scene_outroot, bg_view_folder, 'metadata.json'))

    obj_vali_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    flag = True
    for vi, i in enumerate(ind_vali):
        for obj in obj_body:
            id = int(obj['image name'][-7:-4])
            if id == i:
                flag = False
                obj_pos_2d = np.array(obj['Object 2D boundary'])
                obj_proj = np.array(obj['c2cc proj mat'])
                break
        if flag:
            print('not matching image and index~!')
            return

        obj_view_folder = view_folder.format(mode='val', i=vi)
        # Write image
        obj_img = obj_imgs[i, :, :, :]
        xm.io.img.write_float(obj_img, join(FLAGS.obj_outroot, obj_view_folder, 'rgba.png'), clip=True)
        # Record metadata
        pose = poses[i, :, :]
        # print(pose[:3, 3] - (obj_proj * scale)[:3, 3])

        c2w = np.vstack((pose[:3, :4], np.array([0, 0, 0, 1]).reshape(1, 4)))
        frame_meta = {'file_path': './%s/rgba' % obj_view_folder, 'rotation': 0,
                      'transform_matrix': c2w.tolist()}
        obj_train_meta['frames'].append(frame_meta)
        # Write this frame's metadata to the view folder
        frame_meta = {
            'cam_angle_x': cam_angle_x,
            'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
            'envmap': '', 'envmap_inten': 0, 'imh': obj_img.shape[0],
            'imw': obj_img.shape[1], 'scene': '', 'spp': 0,
            'original_path': obj_img_paths[i],
            'Object 3D BB': obj_3d_BB.tolist(), 'Object 2D pos': obj_pos_2d.tolist(),
            'Object local trfm': obj_transform.tolist(),
        }
        xm.io.json.write(frame_meta, join(FLAGS.obj_outroot, obj_view_folder, 'metadata.json'))

    # Write training and validation JSONs
    xm.io.json.write(bg_train_meta, bg_train_json)
    xm.io.json.write(obj_train_meta, obj_train_json)
    xm.io.json.write(scene_train_meta, scene_train_json)
    xm.io.json.write(bg_vali_meta, bg_vali_json)
    xm.io.json.write(obj_vali_meta, obj_vali_json)
    xm.io.json.write(scene_vali_meta, scene_vali_json)

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
    render_poses = np.array(render_poses).astype(np.float32)

    test_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    for i in range(render_poses.shape[0]):
        view_folder_ = view_folder.format(mode='test', i=i)
        # Record metadata
        pose = render_poses[i, :, :]
        c2w = np.vstack((pose[:3, :4], np.array([0, 0, 0, 1]).reshape(1, 4)))
        frame_meta = {'file_path': '', 'rotation': 0, 'transform_matrix': c2w.tolist()}
        test_meta['frames'].append(frame_meta)
        # Write the nearest input to this test view folder
        dist = np.linalg.norm(pose[:, 3] - poses[:, :, 3], axis=1)
        nn_i = np.argmin(dist)
        bg_nn_img = bg_imgs[nn_i, :, :, :]
        obj_nn_img = obj_imgs[nn_i, :, :, :]
        xm.io.img.write_float(bg_nn_img, join(FLAGS.bg_outroot, view_folder_, 'nn.png'), clip=True)
        xm.io.img.write_float(obj_nn_img, join(FLAGS.obj_outroot, view_folder_, 'nn.png'), clip=True)
        xm.io.img.write_float(obj_nn_img, join(FLAGS.scene_outroot, view_folder_, 'nn.png'), clip=True)
        # Write this frame's metadata to the view folder
        frame_meta = {
            'cam_angle_x': cam_angle_x,
            'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
            'envmap': '', 'envmap_inten': 0, 'imh': bg_nn_img.shape[0],
            'imw': bg_nn_img.shape[1], 'scene': '', 'spp': 0, 'original_path': '', 'Object 3D BB': obj_3d_BB.tolist(),
            'Object local trfm': obj_transform.tolist(), }
        xm.io.json.write(frame_meta, join(FLAGS.bg_outroot, view_folder_, 'metadata.json'))
        xm.io.json.write(frame_meta, join(FLAGS.obj_outroot, view_folder_, 'metadata.json'))
        xm.io.json.write(frame_meta, join(FLAGS.scene_outroot, view_folder_, 'metadata.json'))

    # Write JSON
    xm.io.json.write(test_meta, bg_test_json)
    xm.io.json.write(test_meta, obj_test_json)
    xm.io.json.write(test_meta, scene_test_json)


if __name__ == '__main__':
    app.run(main)
