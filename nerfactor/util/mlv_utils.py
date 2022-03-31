import tensorflow as tf
import numpy as np


def tfmm(A, B):
    """Redefined tensorflow matrix multiply."""
    return tf.reduce_sum(A[..., :, :, tf.newaxis] * B[..., tf.newaxis, :, :], axis=-2)


def trilerp_gather(vol, inds, bad_inds=None):
    """Trilinear interpolation dense gather from volume at query inds."""

    inds_b = inds[Ellipsis, 0]
    inds_x = inds[Ellipsis, 1]
    inds_y = inds[Ellipsis, 2]
    inds_z = inds[Ellipsis, 3]

    inds_x_0 = tf.floor(inds_x)
    inds_x_1 = inds_x_0 + 1
    inds_y_0 = tf.floor(inds_y)
    inds_y_1 = inds_y_0 + 1
    inds_z_0 = tf.floor(inds_z)
    inds_z_1 = inds_z_0 + 1

    # store invalid indices to implement correct out-of-bounds conditions
    invalid_x = tf.logical_or(tf.less(inds_x_0, 0.0), tf.greater(inds_x_1, tf.cast(tf.shape(vol)[1] - 1, tf.float32)))
    invalid_y = tf.logical_or(tf.less(inds_y_0, 0.0), tf.greater(inds_y_1, tf.cast(tf.shape(vol)[2] - 1, tf.float32)))
    invalid_z = tf.logical_or(tf.less(inds_z_0, 0.0), tf.greater(inds_z_1, tf.cast(tf.shape(vol)[3] - 1, tf.float32)))
    if bad_inds is not None:
        invalid_inds = tf.logical_or(tf.logical_or(tf.logical_or(invalid_x, invalid_y), invalid_z), bad_inds)
    else:
        invalid_inds = tf.logical_or(tf.logical_or(invalid_x, invalid_y), invalid_z)

    inds_x_0 = tf.clip_by_value(inds_x_0, 0.0, tf.cast(tf.shape(vol)[1] - 2, tf.float32))
    inds_x_1 = tf.clip_by_value(inds_x_1, 0.0, tf.cast(tf.shape(vol)[1] - 1, tf.float32))
    inds_y_0 = tf.clip_by_value(inds_y_0, 0.0, tf.cast(tf.shape(vol)[2] - 2, tf.float32))
    inds_y_1 = tf.clip_by_value(inds_y_1, 0.0, tf.cast(tf.shape(vol)[2] - 1, tf.float32))
    inds_z_0 = tf.clip_by_value(inds_z_0, 0.0, tf.cast(tf.shape(vol)[3] - 2, tf.float32))
    inds_z_1 = tf.clip_by_value(inds_z_1, 0.0, tf.cast(tf.shape(vol)[3] - 1, tf.float32))

    # compute interp weights
    w_x_0 = 1.0 - (inds_x - inds_x_0)
    w_x_1 = 1.0 - w_x_0
    w_y_0 = 1.0 - (inds_y - inds_y_0)
    w_y_1 = 1.0 - w_y_0
    w_z_0 = 1.0 - (inds_z - inds_z_0)
    w_z_1 = 1.0 - w_z_0

    # w_0_0_0 = w_y_0 * w_x_0 * w_z_0
    # w_1_0_0 = w_y_1 * w_x_0 * w_z_0
    # w_0_1_0 = w_y_0 * w_x_1 * w_z_0
    # w_0_0_1 = w_y_0 * w_x_0 * w_z_1
    # w_1_1_0 = w_y_1 * w_x_1 * w_z_0
    # w_0_1_1 = w_y_0 * w_x_1 * w_z_1
    # w_1_0_1 = w_y_1 * w_x_0 * w_z_1
    # w_1_1_1 = w_y_1 * w_x_1 * w_z_1

    w_0_0_0 = w_x_0 * w_y_0 * w_z_0
    w_1_0_0 = w_x_1 * w_y_0 * w_z_0
    w_0_1_0 = w_x_0 * w_y_1 * w_z_0
    w_0_0_1 = w_x_0 * w_y_0 * w_z_1
    w_1_1_0 = w_x_1 * w_y_1 * w_z_0
    w_0_1_1 = w_x_0 * w_y_1 * w_z_1
    w_1_0_1 = w_x_1 * w_y_0 * w_z_1
    w_1_1_1 = w_x_1 * w_y_1 * w_z_1

    # gather for interp
    inds_0_0_0 = tf.cast(tf.stack([inds_b, inds_x_0, inds_y_0, inds_z_0], axis=-1), tf.int32)
    inds_1_0_0 = tf.cast(tf.stack([inds_b, inds_x_1, inds_y_0, inds_z_0], axis=-1), tf.int32)
    inds_0_1_0 = tf.cast(tf.stack([inds_b, inds_x_0, inds_y_1, inds_z_0], axis=-1), tf.int32)
    inds_0_0_1 = tf.cast(tf.stack([inds_b, inds_x_0, inds_y_0, inds_z_1], axis=-1), tf.int32)
    inds_1_1_0 = tf.cast(tf.stack([inds_b, inds_x_1, inds_y_1, inds_z_0], axis=-1), tf.int32)
    inds_0_1_1 = tf.cast(tf.stack([inds_b, inds_x_0, inds_y_1, inds_z_1], axis=-1), tf.int32)
    inds_1_0_1 = tf.cast(tf.stack([inds_b, inds_x_1, inds_y_0, inds_z_1], axis=-1), tf.int32)
    inds_1_1_1 = tf.cast(tf.stack([inds_b, inds_x_1, inds_y_1, inds_z_1], axis=-1), tf.int32)

    # inds_0_0_0 = tf.cast(tf.stack([inds_b, inds_y_0, inds_x_0, inds_z_0], axis=-1), tf.int32)
    # inds_1_0_0 = tf.cast(tf.stack([inds_b, inds_y_1, inds_x_0, inds_z_0], axis=-1), tf.int32)
    # inds_0_1_0 = tf.cast(tf.stack([inds_b, inds_y_0, inds_x_1, inds_z_0], axis=-1), tf.int32)
    # inds_0_0_1 = tf.cast(tf.stack([inds_b, inds_y_0, inds_x_0, inds_z_1], axis=-1), tf.int32)
    # inds_1_1_0 = tf.cast(tf.stack([inds_b, inds_y_1, inds_x_1, inds_z_0], axis=-1), tf.int32)
    # inds_0_1_1 = tf.cast(tf.stack([inds_b, inds_y_0, inds_x_1, inds_z_1], axis=-1), tf.int32)
    # inds_1_0_1 = tf.cast(tf.stack([inds_b, inds_y_1, inds_x_0, inds_z_1], axis=-1), tf.int32)
    # inds_1_1_1 = tf.cast(tf.stack([inds_b, inds_y_1, inds_x_1, inds_z_1], axis=-1), tf.int32)

    vol_0_0_0 = tf.gather_nd(vol, inds_0_0_0) * w_0_0_0[Ellipsis, tf.newaxis]
    vol_1_0_0 = tf.gather_nd(vol, inds_1_0_0) * w_1_0_0[Ellipsis, tf.newaxis]
    vol_0_1_0 = tf.gather_nd(vol, inds_0_1_0) * w_0_1_0[Ellipsis, tf.newaxis]
    vol_0_0_1 = tf.gather_nd(vol, inds_0_0_1) * w_0_0_1[Ellipsis, tf.newaxis]
    vol_1_1_0 = tf.gather_nd(vol, inds_1_1_0) * w_1_1_0[Ellipsis, tf.newaxis]
    vol_0_1_1 = tf.gather_nd(vol, inds_0_1_1) * w_0_1_1[Ellipsis, tf.newaxis]
    vol_1_0_1 = tf.gather_nd(vol, inds_1_0_1) * w_1_0_1[Ellipsis, tf.newaxis]
    vol_1_1_1 = tf.gather_nd(vol, inds_1_1_1) * w_1_1_1[Ellipsis, tf.newaxis]

    out_vol = vol_0_0_0 + vol_1_0_0 + vol_0_1_0 + vol_0_0_1 + vol_1_1_0 + vol_0_1_1 + vol_1_0_1 + vol_1_1_1

    # boundary conditions for invalid indices
    invalid_inds = tf.tile(invalid_inds[:, :, :, :, tf.newaxis], [1, 1, 1, 1, tf.shape(vol)[4]])
    out_vol = tf.where(invalid_inds, tf.zeros_like(out_vol), out_vol)

    return out_vol


def spherical_cubevol_resample(vol, env2ref, cube_center, side_length, n_phi, n_theta, n_r):
    """Resample cube volume onto spherical coordinates centered at target point.

    Args:
      vol: [B,H,W,D,C], input volume
      env2ref: [B,4,4], relative pose transformation (transform env to ref)
      cube_center: [B,3], [x,y,z] coordinates for center of cube volume
      side_length: side length of cube
      n_phi: number of samples along vertical spherical coordinate dim
      n_theta: number of samples along horizontal spherical coordinate dim
      n_r: number of samples along radius spherical coordinate dim

    Returns:
      resampled: [B, n_phi, n_theta, n_r, C]
    """
    env2ref = env2ref[tf.newaxis, ...]
    cube_center = cube_center[tf.newaxis, ...]
    batch_size = tf.shape(vol)[0]
    height = tf.shape(vol)[1]

    cube_res = tf.cast(height, tf.float32)

    # create spherical coordinates
    b_vals = tf.cast(tf.range(batch_size), tf.float32)
    phi_vals = tf.linspace(0.0, np.pi, n_phi)
    theta_vals = tf.linspace(1.5 * np.pi, -0.5 * np.pi, n_theta)

    # compute radii to use
    x_vals = tf.linspace(-side_length / 2.0, side_length / 2.0, tf.cast(cube_res, tf.int32))
    y_vals = tf.linspace(side_length / 2.0, -side_length / 2.0, tf.cast(cube_res, tf.int32))
    z_vals = tf.linspace(side_length / 2.0, -side_length / 2.0, tf.cast(cube_res, tf.int32))
    x_c, y_c, z_c = tf.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    x_c = x_c + cube_center[:, 0, tf.newaxis, tf.newaxis, tf.newaxis]
    y_c = y_c + cube_center[:, 1, tf.newaxis, tf.newaxis, tf.newaxis]
    z_c = z_c + cube_center[:, 2, tf.newaxis, tf.newaxis, tf.newaxis]

    # x_vals = tf.linspace(-side_length / 2.0, side_length / 2.0, tf.cast(cube_res, tf.int32))
    # y_vals = tf.linspace(-side_length / 2.0, side_length / 2.0, tf.cast(cube_res, tf.int32))
    # z_vals = tf.linspace(side_length / 2.0, -side_length / 2.0, tf.cast(cube_res, tf.int32))
    # y_c, x_c, z_c = tf.meshgrid(y_vals, x_vals, z_vals, indexing='ij')
    # x_c = x_c + cube_center[:, 0, tf.newaxis, tf.newaxis, tf.newaxis]
    # y_c = y_c + cube_center[:, 1, tf.newaxis, tf.newaxis, tf.newaxis]
    # z_c = z_c + cube_center[:, 2, tf.newaxis, tf.newaxis, tf.newaxis]

    cube_coords = tf.stack([x_c, y_c, z_c], axis=4)
    min_r = tf.reduce_min(tf.norm(cube_coords - env2ref[:, :3, 3][:, tf.newaxis, tf.newaxis, tf.newaxis, :], axis=4),
                          axis=[0, 1, 2, 3])  # side_length / cube_res
    max_r = tf.reduce_max(tf.norm(cube_coords - env2ref[:, :3, 3][:, tf.newaxis, tf.newaxis, tf.newaxis, :], axis=4),
                          axis=[0, 1, 2, 3])

    r_vals = tf.linspace(max_r, min_r, n_r)
    b, phi, theta, r = tf.meshgrid(b_vals, phi_vals, theta_vals, r_vals, indexing='ij')  # currently in env frame

    # transform spherical coordinates into cartesian
    # (currently in env frame, z points forwards)
    x = r * tf.cos(theta) * tf.sin(phi)
    z = r * tf.sin(theta) * tf.sin(phi)
    y = r * tf.cos(phi)

    # transform coordinates into ref frame
    sphere_coords = tf.stack([x, y, z, tf.ones_like(x)], axis=-1)[Ellipsis, tf.newaxis]
    sphere_coords_ref = tfmm(env2ref, sphere_coords)
    x = sphere_coords_ref[Ellipsis, 0, 0]
    y = sphere_coords_ref[Ellipsis, 1, 0]
    z = sphere_coords_ref[Ellipsis, 2, 0]

    # transform coordinates into vol indices
    x_inds = (x - cube_center[:, 0, tf.newaxis, tf.newaxis, tf.newaxis] + side_length / 2.0) * (
            (cube_res - 1) / side_length)
    y_inds = -(y - cube_center[:, 1, tf.newaxis, tf.newaxis, tf.newaxis] - side_length / 2.0) * (
            (cube_res - 1) / side_length)
    z_inds = -(z - cube_center[:, 2, tf.newaxis, tf.newaxis, tf.newaxis] - side_length / 2.0) * (
            (cube_res - 1) / side_length)
    sphere_coords_inds = tf.stack([b, x_inds, y_inds, z_inds], axis=-1)

    # trilinear interpolation gather from volume
    # interpolate pre-multiplied RGBAs, then un-pre-multiply
    vol_alpha = tf.clip_by_value(vol[Ellipsis, -1:], 0.0, 1.0)
    vol_channels_p = vol[Ellipsis, :-1] * vol_alpha
    vol_p = tf.concat([vol_channels_p, vol_alpha], axis=-1)

    resampled_p = trilerp_gather(vol_p, sphere_coords_inds)

    resampled_alpha = resampled_p[Ellipsis, -1:]
    resampled_channels = resampled_p[Ellipsis, :-1] / (resampled_alpha + 1e-8)
    resampled = tf.concat([resampled_channels, resampled_alpha], axis=-1)

    return resampled, r_vals


def interleave_shells(shells, radii):
    """Interleave spherical shell tensors out-to-in by radii."""

    radius_order = tf.argsort(radii, direction='DESCENDING')
    shells_interleaved = tf.gather(shells, radius_order, axis=3)
    return shells_interleaved


def over_composite(rgbas):
    """Combines a list of rgba images using the over operation.

    Combines RGBA images from back to front (where back is index 0 in list)
    with the over operation.

    Args:
      rgbas: A list of rgba images, these are combined from *back to front*.

    Returns:
      Returns an RGB image.
    """

    alphas = rgbas[:, :, :, :, -1:]
    colors = rgbas[:, :, :, :, :-1]
    transmittance = tf.math.cumprod(1.0 - alphas + 1.0e-8, axis=3, exclusive=True, reverse=True) * alphas
    output = tf.reduce_sum(transmittance * colors, axis=3)
    accum_alpha = tf.reduce_sum(transmittance, axis=3)

    return tf.concat([output, accum_alpha], axis=3)


def render_envmap(cubes, cube_centers, cube_side_lengths, cube_rel_shapes, cube_nest_inds, ref_pose, env_pose,
                  phi_res, theta_res, r_res):
    """Render environment map from volumetric lights.

    Args:
      cubes: input list of cubes in multiscale volume
      cube_centers: position of cube centers
      cube_side_lengths: side lengths of cubes
      cube_rel_shapes: size of "footprint" of each cube within next coarser cube
      cube_nest_inds: indices for cube "footprints"
      ref_pose: c2w pose of ref camera
      env_pose: c2w pose of environment map camera
      theta_res: resolution of theta (width) for environment map
      phi_res: resolution of phi (height) for environment map
      r_res: number of spherical shells to sample for environment map rendering

    Returns:
      An environment map at the input pose
    """
    num_scales = len(cubes)

    env_c2w = env_pose
    env2ref = tf.matmul(tf.linalg.inv(ref_pose), env_c2w)

    # cube-->sphere resampling
    all_shells_list = []
    all_rad_list = []
    for i in range(num_scales):
        if i == num_scales - 1:
            # "finest" resolution cube, don't zero out
            cube_removed = cubes[i]
        else:
            # zero out areas covered by finer resolution cubes
            cube_shape = cubes[i].get_shape().as_list()[1]

            zm_x, zm_y, zm_z = tf.meshgrid(
                tf.range(cube_nest_inds[i][0], cube_nest_inds[i][0] + cube_rel_shapes[i]),
                tf.range(cube_nest_inds[i][1], cube_nest_inds[i][1] + cube_rel_shapes[i]),
                tf.range(cube_nest_inds[i][2], cube_nest_inds[i][2] + cube_rel_shapes[i]), indexing='ij')

            inds = tf.stack([zm_x, zm_y, zm_z], axis=-1)
            updates = tf.cast(tf.ones_like(zm_x), tf.float32)
            zero_mask = 1.0 - tf.scatter_nd(inds, updates, shape=[cube_shape, cube_shape, cube_shape])
            cube_removed = zero_mask[tf.newaxis, :, :, :, tf.newaxis] * cubes[i]

        spheres_i, rad_i = spherical_cubevol_resample(cube_removed, env2ref, cube_centers[i],
                                                      cube_side_lengths[i], phi_res, theta_res, r_res)
        all_shells_list.append(spheres_i)
        all_rad_list.append(rad_i)

    all_shells = tf.concat(all_shells_list, axis=3)
    all_rad = tf.concat(all_rad_list, axis=0)
    all_shells = interleave_shells(all_shells, all_rad)
    all_shells_envmap = over_composite(all_shells)

    return all_shells_envmap, all_shells_list
