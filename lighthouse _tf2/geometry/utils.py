import tensorflow as tf
import projector as pj


def format_network_input(ref_image, psv_src_images, ref_pose, psv_src_poses, planes, intrinsics):
    """Format the network input.

    Args:
      ref_image: reference source image [batch, height, width, 3]
      psv_src_images: stack of source images (excluding the ref image) [batch,
        height, width, 3*(#source)]
      ref_pose: reference camera-to-world pose (where PSV is constructed)
        [batch, 4, 4]
      psv_src_poses: input poses (camera to world) [batch, 4, 4, #source]
      planes: list of scalar depth values for each plane
      intrinsics: camera intrinsics [batch, 3, 3]

    Returns:
      net_input: [batch, height, width, #planes, (#source+1)*3]
    """

    batch_size = tf.shape(psv_src_images)[0]
    height = tf.shape(psv_src_images)[1]
    width = tf.shape(psv_src_images)[2]
    _, _, _, num_psv_source = list(psv_src_poses.shape)
    num_planes = tf.shape(planes)[0]

    filler = tf.concat([tf.zeros([batch_size, 1, 3]), tf.ones([batch_size, 1, 1])], axis=2)
    intrinsics_filler = tf.stack(
        [tf.cast(height, tf.float32), tf.cast(width, tf.float32), tf.cast(intrinsics[0, 0, 0], tf.float32)],
        axis=0)[:, tf.newaxis]

    ref_pose_c2w = ref_pose
    ref_pose_c2w = tf.concat([
        tf.concat([ref_pose_c2w[:, :3, 0:1], ref_pose_c2w[:, :3, 1:2], -1.0 * ref_pose_c2w[:, :3, 2:3],
                   ref_pose_c2w[:, :3, 3:]], axis=2), filler], axis=1)
    ref_pose_c2w = tf.concat([ref_pose_c2w[0, :3, :], intrinsics_filler], axis=1)

    net_input = []
    for i in range(num_psv_source):
        curr_pose_c2w = psv_src_poses[:, :, :, i]
        curr_pose_c2w = tf.concat([
            tf.concat([
                curr_pose_c2w[:, :3, 0:1], curr_pose_c2w[:, :3, 1:2],
                -1.0 * curr_pose_c2w[:, :3, 2:3], curr_pose_c2w[:, :3, 3:]
            ], 2), filler
        ], 1)
        curr_pose_c2w = tf.concat([curr_pose_c2w[0, :3, :], intrinsics_filler],
                                  axis=1)
        curr_image = psv_src_images[:, :, :, i * 3:(i + 1) * 3]
        curr_psv = pj.make_psv_homogs(curr_image, curr_pose_c2w, ref_pose_c2w, 1.0 / planes, num_planes)
        net_input.append(curr_psv[tf.newaxis, Ellipsis])

    net_input = tf.concat(net_input, axis=4)
    ref_img_stack = tf.tile(tf.expand_dims(ref_image, 3), [1, 1, 1, num_planes, 1])
    net_input = tf.concat([ref_img_stack, net_input], axis=4)
    net_input.set_shape([1, None, None, None, 3 * (num_psv_source + 1)])

    return net_input


def predict_lighting_vol(mpi, planes, intrinsics, cube_res, scale_factors, depth_clip=20.0):
    """Predict lighting volumes from MPI.

    Args:
      mpi: input mpi
      planes: input mpi plane depths
      intrinsics: ref camera intrinsics
      cube_res: resolution of cube volume for lighting prediction
      scale_factors: scales for multiresolution cube sampling
      depth_clip: farthest depth (sets limits of coarsest cube)

    Returns:
      list of completed lighting volumes
    """

    batchsize = tf.shape(mpi)[0]
    max_depth = tf.minimum(planes[0], depth_clip)

    cube_side_lengths = [2.0 * max_depth]
    for i in range(len(scale_factors)):
        cube_side_lengths.append(2.0 * max_depth / tf.cast(scale_factors[i], tf.float32))

    # shape of each cube's footprint within the next coarser volume
    cube_rel_shapes = []
    for i in range(len(scale_factors)):
        if i == 0:
            i_rel_shape = cube_res // scale_factors[0]
        else:
            i_rel_shape = (cube_res * scale_factors[i - 1]) // scale_factors[i]
        cube_rel_shapes.append(i_rel_shape)

    cube_centers = [tf.zeros([batchsize, 3])]
    for i in range(len(scale_factors)):
        i_center_depth = (cube_side_lengths[i] / tf.cast((cube_res - 1), tf.float32)) * tf.cast(
            (cube_rel_shapes[i] // 2), tf.float32)
        cube_centers.append(tf.concat([tf.zeros([batchsize, 2]), i_center_depth * tf.ones([batchsize, 1])], axis=1))

    cube_nest_inds = []
    for i in range(len(scale_factors)):
        if i == 0:
            i_nest_inds = [(cube_res - cube_rel_shapes[i]) // 2,
                           (cube_res - cube_rel_shapes[i]) // 2,
                           cube_res // 2 - cube_rel_shapes[i]]
        else:
            i_nest_inds = [(cube_res - cube_rel_shapes[i]) // 2,
                           (cube_res - cube_rel_shapes[i]) // 2,
                           cube_res - cube_rel_shapes[i]]
        cube_nest_inds.append(i_nest_inds)

    cube_list = []
    for i in range(len(cube_centers)):
        i_cube, _ = pj.mpi_resample_cube(mpi, cube_centers[i], intrinsics, planes, cube_side_lengths[i], cube_res)

        cube_list.append(i_cube)
    return cube_list, cube_centers, cube_side_lengths, cube_rel_shapes, cube_nest_inds


def render_envmap(cubes, cube_centers, cube_side_lengths, cube_rel_shapes, cube_nest_inds, ref_pose, env_pose,
                  theta_res, phi_res, r_res):
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

            zm_y, zm_x, zm_z = tf.meshgrid(
                tf.range(cube_nest_inds[i][0], cube_nest_inds[i][0] + cube_rel_shapes[i]),
                tf.range(cube_nest_inds[i][1], cube_nest_inds[i][1] + cube_rel_shapes[i]),
                tf.range(cube_nest_inds[i][2], cube_nest_inds[i][2] + cube_rel_shapes[i]), indexing='ij')

            inds = tf.stack([zm_y, zm_x, zm_z], axis=-1)
            updates = tf.cast(tf.ones_like(zm_x), tf.float32)
            zero_mask = 1.0 - tf.scatter_nd(inds, updates, shape=[cube_shape, cube_shape, cube_shape])
            cube_removed = zero_mask[tf.newaxis, :, :, :, tf.newaxis] * cubes[i]

        spheres_i, rad_i = pj.spherical_cubevol_resample(cube_removed, env2ref, cube_centers[i],
                                                         cube_side_lengths[i], phi_res, theta_res, r_res)
        all_shells_list.append(spheres_i)
        all_rad_list.append(rad_i)

    all_shells = tf.concat(all_shells_list, axis=3)
    all_rad = tf.concat(all_rad_list, axis=0)
    all_shells = pj.interleave_shells(all_shells, all_rad)
    all_shells_envmap = pj.over_composite(all_shells)

    return all_shells_envmap, all_shells_list
