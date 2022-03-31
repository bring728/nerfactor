from os import makedirs
from os.path import join, basename, exists
import numpy as np
from absl import app
from tqdm import tqdm
import tensorflow as tf

from third_party.xiuminglib import xiuminglib as xm
from brdf.renderer import gen_light_xyz
from nerfactor import datasets
from nerfactor import models
from nerfactor.util import io as ioutil, logging as logutil, config as configutil, img as imgutil
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

debug = False
logger = logutil.Logger(loggee="geometry_from_nerf")
root_dir = '/data/happily/SVL-nerf_data/'

scene = 'pinecone'
scene_bbox = '-0.2,0.2,-0.2,0.2,-0.12,0.3'
data_root = root_dir + 'real_scene/' + scene
imh = '1008'
lr = '5e-4'
trained_nerf = root_dir + 'output/train/' + scene + '/' + scene + "_nerf/lr" + lr

occu_thres = 0.5
mlp_chunk = 900000  # bump this up until GPU gets OOM for faster computation
spp = 1
fps = 12

bg_scene = '0810-4'
bg_datadir = root_dir + 'real_scene/' + bg_scene
out_root = root_dir + 'output/surf/' + scene + '_' + bg_scene
bg_near = 1.3
bg_far = 6

obj_3d_bb = np.array([-0.5, 0, 1.75, 0, 0.5, 2.25])


def get_rot_mat(radian, axis):
    if axis is 'x':
        return np.array([[1, 0, 0], [0, np.cos(radian), -np.sin(radian)], [0, np.sin(radian), np.cos(radian)]])
    elif axis is 'y':
        return np.array([[np.cos(radian), 0, np.sin(radian)], [0, 1, 0], [-np.sin(radian), 0, np.cos(radian)]])
    elif axis is 'z':
        return np.array([[np.cos(radian), -np.sin(radian), 0], [np.sin(radian), np.cos(radian), 0], [0, 0, 1]])


def main(_):
    # Get the latest checkpoint
    ckpts = xm.os.sortglob(join(trained_nerf, 'checkpoints'), 'ckpt-*.index')
    ckpt_ind = [int(basename(x)[len('ckpt-'):-len('.index')]) for x in ckpts]
    latest_ckpt = ckpts[np.argmax(ckpt_ind)]
    latest_ckpt = latest_ckpt[:-len('.index')]

    # Load its config.
    config_ini = configutil.get_config_ini(latest_ckpt)
    config = ioutil.read_config(config_ini)
    if imh is not None:  # if using a new image resolution
        config.set('DEFAULT', 'imh', str(imh))

    # Restore model
    model = restore_model(config, latest_ckpt)
    config.set('DEFAULT', 'data_root', bg_datadir)
    config.set('DEFAULT', 'dataset', 'nerf_boundingbox')

    obj_loc = np.array([-0.25, 0.1, 2.0]).reshape([3, 1])
    rot_x = np.radians(0)
    rot_y = np.radians(0)
    rot_z = np.radians(0)
    rot_x_axis = np.radians(90)
    rot_y_axis = np.radians(0)
    rot_z_axis = np.radians(0)
    scale_x, scale_y, scale_z = 0.8, 0.8, 0.8
    # scale_x, scale_y, scale_z = 1, 1, 1

    o2b_S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, scale_z]], dtype=np.float32)
    o2b_R = get_rot_mat(rot_z, axis='z') @ get_rot_mat(rot_x, axis='x') @ get_rot_mat(rot_y, axis='y') @ get_rot_mat(
        rot_x_axis, axis='x').astype(np.float32)
    o2b_R = o2b_S @ o2b_R

    o2b_proj = np.concatenate([o2b_R, obj_loc], axis=1)
    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
    o2b_proj = np.concatenate([o2b_proj, bottom], 0)
    obj_loc = o2b_proj[:3, 3:]
    print(o2b_proj)
    b2o_proj = np.linalg.inv(o2b_proj)
    w2l_mat = tf.convert_to_tensor(b2o_proj[:3, :3], dtype=tf.float32)
    w2l_tr = tf.convert_to_tensor(b2o_proj[:3, 3], dtype=tf.float32)
    # obj_3d_bb = np.array([-0.5, 0, 1.75, 0, 0.5, 2.25])
    # x_min, x_max, y_min, y_max, z_min, z_max = scene_bbox.split(',')
    #
    # tmp_1 = np.array([[obj_3d_bb[0], 0, 0, 1, 0, 0], [0, obj_3d_bb[1], 0, 0, 1, 0], [0, 0, obj_3d_bb[2], 0, 0, 1],
    #                   [obj_3d_bb[3], 0, 0, 1, 0, 0], [0, obj_3d_bb[4], 0, 0, 1, 0], [0, 0, obj_3d_bb[5], 0, 0, 1]],
    #                  dtype=np.float32)
    # tmp_2 = np.array([float(x_min), float(y_min), float(z_min), float(x_max), float(y_max), float(z_max)])
    # w2l_trfm = np.linalg.inv(tmp_1) @ tmp_2
    #
    # w2l_mat = tf.constant([[w2l_trfm[0], 0, 0], [0, w2l_trfm[1], 0], [0, 0, w2l_trfm[2]]], dtype=tf.float32)
    # w2l_tr = tf.constant([[w2l_trfm[3], w2l_trfm[4], w2l_trfm[5]]], dtype=tf.float32)
    #
    # axis_swap = tf.constant([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=tf.float32)
    # w2l_mat = tf.matmul(w2l_mat, axis_swap)
    # w2l_tr = tf.squeeze(tf.matmul(w2l_tr, axis_swap), 0)
    # l2w_proj = np.linalg.inv(np.array([[w2l_trfm[0], 0, 0, w2l_trfm[3]], [0, w2l_trfm[1], 0, w2l_trfm[4]],
    #                                    [0, 0, w2l_trfm[2], w2l_trfm[5]], [0, 0, 0, 1]], dtype=np.float32))
    # print(l2w_proj)

    mode = 'test'
    dataset_name = config.get('DEFAULT', 'dataset')
    no_batch = config.getboolean('DEFAULT', 'no_batch')
    Dataset = datasets.get_dataset_class(dataset_name)
    dataset = Dataset(config, mode, always_all_rays=True, spp=spp)
    n_views = dataset.get_n_views()
    datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)

    # Process all views of this mode
    for batch in tqdm(datapipe, desc=f"Views ({mode})", total=n_views):
        id = batch[0][0].numpy().decode()
        process_view(config, model, batch, w2l_mat, w2l_tr, obj_loc)


def process_view(config, model, batch, w2l_mat, w2l_tr, obj_loc):
    sps = int(np.sqrt(spp))  # no need to check if square

    id_, hw, rayo, rayd, radii, tmin, tmax = batch
    id_ = id_[0].numpy().decode()
    hw = hw[0, :]

    # not normalize in bg nerf
    # rayd = tf.linalg.l2_normalize(rayd, axis=1)

    out_dir = join(out_root, id_)
    if not exists(out_dir):
        makedirs(out_dir)

        # Is this job done already?
    expected = [join(out_dir, 'alpha.png'), join(out_dir, 'xyz.npy'), join(out_dir, 'xyz.png'),
                join(out_dir, 'depth.npy'), join(out_dir, 'depth.png')]

    all_exist = all(exists(x) for x in expected)
    if all_exist:
        logger.info(f"Skipping {id_} since it's done already")
        return

    # ------ Tracing from Camera to Object
    _, _, occu, exp_depth = compute_depth(model, rayo, rayd, tmin, tmax, config, w2l_mat, w2l_tr)

    # Clip smaller-than-threshold alpha to 0
    transp_ind = tf.where(occu < occu_thres)
    occu = tf.tensor_scatter_nd_update(occu, transp_ind, tf.zeros((tf.shape(transp_ind)[0],)))

    # Write alpha map
    alpha_map = tf.reshape(occu, hw * sps)
    alpha_map = average_supersamples(alpha_map, sps)
    alpha_map = tf.clip_by_value(alpha_map, 0., 1.)
    write_alpha(alpha_map, out_dir)

    # Write XYZ map, whose background filling value is (0, 0, 0)
    surf = rayo + rayd * exp_depth[:, None]  # Surface XYZs
    surf = tf.linalg.matmul(surf, w2l_mat, transpose_b=True) + w2l_tr[None, :]

    xyz_map = tf.reshape(surf, (hw[0] * sps, hw[1] * sps, 3))
    xyz_map = average_supersamples(xyz_map, sps)
    xyz_map = imgutil.alpha_blend(xyz_map, alpha_map)
    write_xyz_my(xyz_map, out_dir, alpha_map)
    # write_xyz(xyz_map, out_dir)

    exp_depth = tf.reshape(exp_depth, hw * sps)
    exp_depth = average_supersamples(exp_depth, sps)
    exp_depth = imgutil.alpha_blend(exp_depth, alpha_map)
    write_depth(exp_depth, out_dir)

    # Dump raw
    raw_out = join(out_dir, 'obj_loc.npy')
    with open(raw_out, 'wb') as h:
        np.save(h, obj_loc)


def compute_depth(model, rayo, rayd, tmin, tmax, config, w2l_mat, w2l_tr):
    n_samples_coarse = 64 + config.getint('DEFAULT', 'n_samples_coarse')
    n_samples_fine = 64 + config.getint('DEFAULT', 'n_samples_fine')

    lin_in_disp = config.getboolean('DEFAULT', 'lin_in_disp')
    perturb = False  # NOTE: do not randomize at test time

    # Points in space to evaluate the coarse model at
    z = model.gen_z(bg_near, bg_far, n_samples_coarse, rayo.shape[0], lin_in_disp=lin_in_disp, perturb=perturb)
    # z = model.gen_z_box(tmin, tmax, n_samples_coarse, lin_in_disp=lin_in_disp, perturb=perturb)
    pts = rayo[:, None, :] + rayd[:, None, :] * z[:, :, None]
    # shape is
    pts = tf.linalg.matmul(pts, w2l_mat, transpose_b=True) + w2l_tr[None, None, :]
    pts_flat = tf.reshape(pts, (-1, 3))

    # Evaluate coarse model for importance sampling
    sigma_flat = eval_sigma_mlp(model, pts_flat, use_fine=False)
    sigma = tf.reshape(sigma_flat, pts.shape[:2])
    weights_coarse = model.accumulate_sigma(sigma, z, rayd)
    occu_coarse = tf.reduce_sum(weights_coarse, -1)  # (n_rays,)
    exp_depth_coarse = tf.reduce_sum(weights_coarse * z, axis=-1)  # (n_rays,)

    # Obtain additional samples using importance sampling
    z = model.gen_z_fine(z, weights_coarse, n_samples_fine, perturb=perturb)
    pts = rayo[:, None, :] + rayd[:, None, :] * z[:, :, None]
    pts = tf.linalg.matmul(pts, w2l_mat, transpose_b=True) + w2l_tr[None, None, :]
    pts_flat = tf.reshape(pts, (-1, 3))

    sigma_flat = eval_sigma_mlp(model, pts_flat, use_fine=True)
    sigma = tf.reshape(sigma_flat, pts.shape[:2])
    weights = model.accumulate_sigma(sigma, z, rayd)  # (n_rays, n_samples)
    occu = tf.reduce_sum(weights, -1)  # (n_rays,)
    exp_depth = tf.reduce_sum(weights * z, axis=-1)  # (n_rays,)
    return occu_coarse, exp_depth_coarse, occu, exp_depth


def eval_sigma_mlp(model, pts, use_fine=False):
    embedder = model.embedder['xyz']
    if use_fine:
        pref = 'fine_'
    else:
        pref = 'coarse_'
    enc = model.net[pref + 'enc']
    sigma_out = model.net.get(pref + 'a_out', model.net[pref + 'sigma_out'])

    # For real scenes: override out-of-bound sigma to be 0
    in_bounds = check_bounds(pts)
    pts_in = tf.boolean_mask(pts, in_bounds)

    # Chunk by chunk to avoid OOM
    sigma_chunks = []
    for i in range(0, pts_in.shape[0], mlp_chunk):
        end_i = min(pts_in.shape[0], i + mlp_chunk)
        pts_chunk = pts_in[i:end_i, :]
        sigma_chunk = tf.nn.relu(sigma_out(enc(embedder(pts_chunk))))
        sigma_chunks.append(sigma_chunk)
    assert sigma_chunks, "No sigma chunk to concat."
    sigma_in = tf.concat(sigma_chunks, axis=0)

    # Assign these predicted sigma to a full zero tensor
    full_shape = (tf.shape(pts)[0], 1)
    in_ind = tf.where(in_bounds)
    sigma = tf.scatter_nd(in_ind, sigma_in, full_shape)
    return sigma


def average_supersamples(map_supersampled, sps):
    maps = []
    for i in range(sps):
        for j in range(sps):
            sample = map_supersampled[i::sps, j::sps, ...]
            sample = sample[None, ...]
            maps.append(sample)
    assert maps, "No map to concat."
    maps = tf.concat(maps, axis=0)
    return tf.reduce_mean(maps, axis=0)


def check_bounds(pts):
    if scene_bbox is None or scene_bbox == '':
        return tf.ones((tf.shape(pts)[0],), dtype=bool)
    # Parse bounds
    x_min, x_max, y_min, y_max, z_min, z_max = scene_bbox.split(',')
    # Assume cube bottom center at world origin on XY plane
    x_min, x_max = float(x_min), float(x_max)
    y_min, y_max = float(y_min), float(y_max)
    z_min, z_max = float(z_min), float(z_max)
    in_x = tf.logical_and(pts[:, 0] >= x_min, pts[:, 0] <= x_max)
    in_y = tf.logical_and(pts[:, 1] >= y_min, pts[:, 1] <= y_max)
    in_z = tf.logical_and(pts[:, 2] >= z_min, pts[:, 2] <= z_max)
    in_bounds = tf.logical_and(in_x, tf.logical_and(in_y, in_z))
    return in_bounds


def write_xyz(xyz_arr, out_dir):
    arr = xyz_arr.numpy()
    # Dump raw
    raw_out = join(out_dir, 'xyz.npy')
    with open(raw_out, 'wb') as h:
        np.save(h, arr)
    # Visualization
    vis_out = join(out_dir, 'xyz.png')
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
    xm.io.img.write_arr(arr_norm, vis_out)


def write_xyz_my(xyz_arr, out_dir, alpha_map):
    arr = xyz_arr.numpy()
    # Dump raw
    raw_out = join(out_dir, 'xyz.npy')
    with open(raw_out, 'wb') as h:
        np.save(h, arr)
    # Visualization
    vis_out = join(out_dir, 'xyz.png')
    arr_norm = np.stack([(arr[..., 0] - arr[..., 0].min()) / (arr[..., 0].max() - arr[..., 0].min()),
                         (arr[..., 1] - arr[..., 1].min()) / (arr[..., 1].max() - arr[..., 1].min()),
                         (arr[..., 2] - arr[..., 2].min()) / (arr[..., 2].max() - arr[..., 2].min())], axis=-1)
    arr_norm = imgutil.alpha_blend(arr_norm, alpha_map)
    xm.io.img.write_arr(arr_norm, vis_out)


def write_depth(depth_arr, out_dir):
    arr = depth_arr.numpy()
    # Dump raw
    raw_out = join(out_dir, 'depth.npy')
    with open(raw_out, 'wb') as h:
        np.save(h, arr)
    # Visualization
    vis_out = join(out_dir, 'depth.png')
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
    xm.io.img.write_arr(arr_norm, vis_out)


def write_alpha(arr, out_dir):
    arr = arr.numpy()
    vis_out = join(out_dir, 'alpha.png')
    xm.io.img.write_arr(arr, vis_out)


def restore_model(config, ckpt_path):
    model_name = config.get('DEFAULT', 'model')
    Model = models.get_model_class(model_name)
    model = Model(config)
    ioutil.restore_model(model, ckpt_path)
    return model


if __name__ == '__main__':
    app.run(main)
