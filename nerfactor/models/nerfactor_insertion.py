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

# pylint: disable=arguments-differ

from os.path import basename, dirname, join, exists
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from third_party.xiuminglib import xiuminglib as xm
from brdf.renderer import gen_light_xyz
from nerfactor.models.shape import Model as ShapeModel
from nerfactor.models.brdf import Model as BRDFModel
from nerfactor.networks import mlp
from nerfactor.networks.embedder import Embedder
from nerfactor.util import vis as visutil, config as configutil, \
    io as ioutil, tensor as tutil, light as lightutil, img as imgutil, \
    math as mathutil, geom as geomutil
from nerfactor.models.MLV_nerf import MLV
from nerfactor import models
from nerfactor.util.mlv_utils import render_envmap


class Model(ShapeModel):
    def __init__(self, config, debug=False):
        # BRDF
        brdf_ckpt = config.get('DEFAULT', 'brdf_model_ckpt')
        brdf_config_path = configutil.get_config_ini(brdf_ckpt)
        self.config_brdf = ioutil.read_config(brdf_config_path)
        self.pred_brdf = config.getboolean('DEFAULT', 'pred_brdf')
        self.z_dim = self.config_brdf.getint('DEFAULT', 'z_dim')
        self.normalize_brdf_z = self.config_brdf.getboolean('DEFAULT', 'normalize_z')
        # Shape
        self.shape_mode = config.get('DEFAULT', 'shape_mode')
        self.shape_model_ckpt = config.get('DEFAULT', 'shape_model_ckpt')
        shape_config_path = configutil.get_config_ini(self.shape_model_ckpt)
        if self.shape_mode in ('nerf', 'scratch'):
            self.config_shape = None
        else:
            self.config_shape = ioutil.read_config(shape_config_path)
        # By now we have all attributes required by parent init.
        super().__init__(config, debug=debug)
        # BRDF
        self.albedo_smooth_weight = config.getfloat('DEFAULT', 'albedo_smooth_weight')
        self.brdf_smooth_weight = config.getfloat('DEFAULT', 'brdf_smooth_weight')
        self.brdf_model = BRDFModel(self.config_brdf)
        ioutil.restore_model(self.brdf_model, brdf_ckpt)
        self.brdf_model.trainable = False
        # Lighting
        self._light = None  # see the light property
        light_h = self.config.getint('DEFAULT', 'light_h')
        self.light_res = (light_h, 2 * light_h)
        lxyz, lareas = gen_light_xyz(*self.light_res)
        self.lxyz = tf.convert_to_tensor(lxyz, dtype=tf.float32)
        self.lareas = tf.convert_to_tensor(lareas, dtype=tf.float32)
        # PSNR calculator
        self.psnr = xm.metric.PSNR('uint8')

        bg_config_ini = configutil.get_config_ini(config.get('DEFAULT', 'bg_ckpt'))
        bg_config = ioutil.read_config(bg_config_ini)
        # Output directory
        model_name = bg_config.get('DEFAULT', 'model')
        Model = models.get_model_class(model_name)
        self.bg_net = Model(bg_config, False)
        ioutil.restore_model(self.bg_net, config.get('DEFAULT', 'bg_ckpt'))
        self.bg_net.trainable = False

        model_mlv = MLV(bg_ckpt=config.get('DEFAULT', 'bg_ckpt'), theta_res=light_h * 2, phi_res=light_h, r_res=128,
                        bg_net=self.bg_net)
        ioutil.restore_model_v2(model_mlv, config.get('DEFAULT', 'mlv_ckpt'))
        model_mlv.trainable = False
        model_mlv.cube_to_mlv(config.getboolean('DEFAULT', 'hallucinate'))
        self.mlv = model_mlv
        self.is_set_mlv = True
        # obj_pose = config.get('DEFAULT', 'obj_pose').split(',')
        # self.obj_pose = np.array(obj_pose).astype(np.float).reshape(4, 4)
        # l2w_proj = np.array(config.get('DEFAULT', 'l2w_proj').split(',')).astype(np.float).reshape(4, 4)
        # self.l2w_mat = l2w_proj[:3,:3]
        # self.l2w_tr = l2w_proj[:3,-1]
        self.alpha_thres = config.getfloat('DEFAULT', 'alpha_thres')
        self.bg_z = None
        self.bg_rgbs = None

    def _init_embedder(self):
        embedder = super()._init_embedder()
        pos_enc = self.config.getboolean('DEFAULT', 'pos_enc')
        # We need to use the level number used in training the BRDF MLP
        n_freqs_rusink = self.config_brdf.getint('DEFAULT', 'n_freqs')
        # Shortcircuit if not using embedders
        if not pos_enc:
            embedder['rusink'] = tf.identity
            return embedder
        # Rusink. embedder
        kwargs = {
            'incl_input': True,
            'in_dims': 3,
            'log2_max_freq': n_freqs_rusink - 1,
            'n_freqs': n_freqs_rusink,
            'log_sampling': True,
            'periodic_func': [tf.math.sin, tf.math.cos]}
        embedder_rusink = Embedder(**kwargs)
        embedder['rusink'] = embedder_rusink
        return embedder

    def _init_net(self):
        mlp_width = self.config.getint('DEFAULT', 'mlp_width')
        mlp_depth = self.config.getint('DEFAULT', 'mlp_depth')
        mlp_skip_at = self.config.getint('DEFAULT', 'mlp_skip_at')
        net = {}
        # Albedo
        net['albedo_mlp'] = mlp.Network([mlp_width] * mlp_depth, act=['relu'] * mlp_depth, skip_at=[mlp_skip_at])
        net['albedo_out'] = mlp.Network([3], act=['sigmoid'])  # [0, 1]
        # BRDF Z
        if self.pred_brdf:
            net['brdf_z_mlp'] = mlp.Network([mlp_width] * mlp_depth, act=['relu'] * mlp_depth, skip_at=[mlp_skip_at])
            net['brdf_z_out'] = mlp.Network([self.z_dim], act=None)
        # Training from scratch, finetuning, or just using NeRF geometry?
        if self.shape_mode == 'scratch':
            net['normal_mlp'] = mlp.Network([mlp_width] * mlp_depth, act=['relu'] * mlp_depth, skip_at=[mlp_skip_at])
            net['normal_out'] = mlp.Network([3], act=None)  # normalized elsewhere
            net['lvis_mlp'] = mlp.Network([mlp_width] * mlp_depth, act=['relu'] * mlp_depth, skip_at=[mlp_skip_at])
            net['lvis_out'] = mlp.Network([1], act=['sigmoid'])  # [0, 1]
        elif self.shape_mode in ('frozen', 'finetune'):
            shape_model = ShapeModel(self.config_shape)
            ioutil.restore_model(shape_model, self.shape_model_ckpt)
            shape_model.trainable = self.shape_mode == 'finetune'
            net['normal_mlp'] = shape_model.net['normal_mlp']
            net['normal_out'] = shape_model.net['normal_out']
            net['lvis_mlp'] = shape_model.net['lvis_mlp']
            net['lvis_out'] = shape_model.net['lvis_out']
        elif self.shape_mode == 'nerf':
            pass
        else:
            raise ValueError(self.shape_mode)
        return net

    def call(self, batch, mode='train', fixed_view=False):
        self._validate_mode(mode)
        id_, hw, rayo, rayd, radii, viewdir, alpha, xyz, depth, obj_loc = batch
        # Mask out 100% background
        mask = alpha[:, 0] > self.alpha_thres
        rayo_masked = tf.boolean_mask(rayo, mask)
        depth = tf.boolean_mask(depth, mask)
        xyz = tf.boolean_mask(xyz, mask)
        # Directions
        surf2l = self._calc_ldir(xyz)
        surf2c = self._calc_vdir(rayo_masked, xyz)
        # Jitter XYZs
        normal_pred = self._pred_normal_at(xyz)
        normal_pred = mathutil.safe_l2_normalize(normal_pred, axis=1)
        lvis_pred = self._pred_lvis_at(xyz, surf2l)
        albedo = self._pred_albedo_at(xyz)
        # ------ BRDFs
        brdf_prop = self._pred_brdf_at(xyz)
        if self.normalize_brdf_z:
            brdf_prop = mathutil.safe_l2_normalize(brdf_prop, axis=1)
        brdf = self._eval_brdf_at(surf2l, surf2c, normal_pred, albedo, brdf_prop)  # NxLx3

        env_pose =tf.concat([tf.concat([tf.eye(3), obj_loc], axis=1), tf.constant([[0,0,0,1]], dtype=tf.float32)], axis=0)
        light = self.mlv.nerfactor_call(env_pose)[0, :, :, :3]
        self.novel_probes_uint = lightutil.vis_light(light, h=48, tonemap=False)

        obj_rgb = self._render(lvis_pred, brdf, surf2l, normal_pred, light)  # all Nx3

        ind = tf.where(mask)
        n = tf.shape(alpha)[0]  # total number of rays
        l = tf.shape(lvis_pred)[1]  # total number of light directions
        obj_rgb = tf.scatter_nd(ind, obj_rgb, (n, 3))

        inf = 1e10
        obj_z = tf.ones((n, 1), dtype=tf.float32) * inf
        obj_z = tf.tensor_scatter_nd_update(obj_z, ind, depth)

        if fixed_view:
            if self.bg_z is None:
                bg_rgbs, bg_z = self.bg_net.nerfactor_call(rayo, rayd, radii, viewdir)
                self.bg_z = bg_z
                self.bg_rgbs = bg_rgbs
            else:
                bg_rgbs, bg_z = self.bg_rgbs, self.bg_z
        else:
            bg_rgbs, bg_z = self.bg_net.nerfactor_call(rayo, rayd, radii, viewdir)

        rgb_pred, alpha_pred = self.bg_net.accumulate_with_obj(bg_rgbs, obj_rgb, bg_z, obj_z, rayd, inf=inf)

        normal_pred = tf.scatter_nd(ind, normal_pred, (n, 3))
        lvis_pred = tf.scatter_nd(ind, lvis_pred, (n, l))
        albedo = tf.scatter_nd(ind, albedo, (n, 3))
        brdf_prop = tf.scatter_nd(ind, brdf_prop, (n, self.z_dim))

        gt = {'alpha': alpha}
        pred = {'rgb': rgb_pred, 'normal': normal_pred, 'lvis': lvis_pred,
                'albedo': albedo, 'brdf': brdf_prop, 'alpha': alpha_pred}

        # ------ To visualize
        to_vis = {'id': id_, 'hw': hw}
        for k, v in pred.items():
            to_vis['pred_' + k] = v
        for k, v in gt.items():
            to_vis['gt_' + k] = v
        return pred, to_vis

    def _render(self, light_vis, brdf, l, n, light):
        linear2srgb = self.config.getboolean('DEFAULT', 'linear2srgb')
        cos = tf.einsum('ijk,ik->ij', l, n)  # NxL
        # Areas for intergration
        areas = tf.reshape(self.lareas, (1, -1, 1))  # 1xLx1
        # NOTE: unnecessary if light_vis already encodes it, but won't hurt
        front_lit = tf.cast(cos > 0, tf.float32)
        lvis = front_lit * light_vis  # NxL

        def integrate(light):
            light_flat = tf.reshape(light, (-1, 3))  # Lx3
            light = lvis[:, :, None] * light_flat[None, :, :]  # NxLx3
            light_pix_contrib = brdf * light * cos[:, :, None] * areas  # NxLx3
            rgb = tf.reduce_sum(light_pix_contrib, axis=1)  # Nx3
            # Tonemapping
            rgb = tf.clip_by_value(rgb, 0., 1.)  # NOTE
            # Colorspace transform
            if linear2srgb:
                rgb = imgutil.linear2srgb(rgb)
            return rgb

        rgb = integrate(light)
        return rgb

    def set_obj_envmap(self, obj_pose):
        self.obj_pose = obj_pose
        self.light = self.mlv(self.obj_pose)[0, :, :, :3]

    # @property
    # def light(self):
    #     if self._light is None:  # initialize just once
    #         maxv = self.config.getfloat('DEFAULT', 'light_init_max')
    #         light = tf.random.uniform(self.light_res + (3,), minval=0., maxval=maxv)
    #         self._light = tf.Variable(light, trainable=True)
    #     # No negative light
    #     return tf.clip_by_value(self._light, 0., np.inf)  # 3D

    def _pred_albedo_at(self, pts):
        # Given that albedo generally ranges from 0.1 to 0.8
        albedo_scale = self.config.getfloat('DEFAULT', 'albedo_slope', fallback=0.7)
        albedo_bias = self.config.getfloat('DEFAULT', 'albedo_bias', fallback=0.1)
        mlp_layers = self.net['albedo_mlp']
        out_layer = self.net['albedo_out']  # output in [0, 1]
        embedder = self.embedder['xyz']

        def chunk_func(surf):
            surf_embed = embedder(surf)
            albedo = out_layer(mlp_layers(surf_embed))
            return albedo

        albedo = self.chunk_apply(chunk_func, pts, 3, self.mlp_chunk)
        albedo = albedo_scale * albedo + albedo_bias  # [bias, scale + bias]
        albedo = tf.debugging.check_numerics(albedo, "Albedo")
        return albedo  # Nx3

    def _pred_brdf_at(self, pts):
        mlp_layers = self.net['brdf_z_mlp']
        out_layer = self.net['brdf_z_out']
        embedder = self.embedder['xyz']

        def chunk_func(surf):
            surf_embed = embedder(surf)
            brdf_z = out_layer(mlp_layers(surf_embed))
            return brdf_z

        brdf_z = self.chunk_apply(chunk_func, pts, self.z_dim, self.mlp_chunk)
        return brdf_z  # NxZ

    def _eval_brdf_at(self, pts2l, pts2c, normal, albedo, brdf_prop):
        brdf_scale = self.config.getfloat('DEFAULT', 'learned_brdf_scale')
        z = brdf_prop
        world2local = geomutil.gen_world2local(normal)
        # Transform directions into local frames
        vdir = tf.einsum('jkl,jl->jk', world2local, pts2c)
        ldir = tf.einsum('jkl,jnl->jnk', world2local, pts2l)

        # Directions to Rusink.
        ldir_flat = tf.reshape(ldir, (-1, 3))
        vdir_rep = tf.tile(vdir[:, None, :], (1, tf.shape(ldir)[1], 1))
        vdir_flat = tf.reshape(vdir_rep, (-1, 3))
        rusink = geomutil.dir2rusink(ldir_flat, vdir_flat)  # NLx3

        # Repeat BRDF Z
        z_rep = tf.tile(z[:, None, :], (1, tf.shape(ldir)[1], 1))
        z_flat = tf.reshape(z_rep, (-1, self.z_dim))
        # Mask out back-lit directions for speed
        local_normal = tf.convert_to_tensor((0, 0, 1), dtype=tf.float32)
        local_normal = tf.reshape(local_normal, (3, 1))
        cos = ldir_flat @ local_normal
        front_lit = tf.reshape(cos, (-1,)) > 0
        rusink_fl = rusink[front_lit]
        z_fl = z_flat[front_lit]
        # Predict BRDF values given identities and Rusink.
        mlp_layers = self.brdf_model.net['brdf_mlp']
        out_layer = self.brdf_model.net['brdf_out']
        embedder = self.embedder['rusink']

        def chunk_func(rusink_z):
            rusink, z = rusink_z[:, :3], rusink_z[:, 3:]
            rusink_embed = embedder(rusink)
            z_rusink = tf.concat((z, rusink_embed), axis=1)
            # Strange that shape can't be inferred from the restored
            # `self.brdf_model`, so we set it manually
            z_rusink = tf.ensure_shape(
                z_rusink, (None, self.z_dim + embedder.out_dims))
            brdf = out_layer(mlp_layers(z_rusink))
            return brdf

        rusink_z = tf.concat((rusink_fl, z_fl), 1)
        brdf_fl = self.chunk_apply(chunk_func, rusink_z, 1, self.mlp_chunk)
        # Put front-lit BRDF values back into an all-zero flat tensor, ...
        brdf_flat = tf.scatter_nd(tf.where(front_lit), brdf_fl, (tf.shape(front_lit)[0], 1))
        # and then reshape the resultant flat tensor
        spec = tf.reshape(brdf_flat, (tf.shape(ldir)[0], tf.shape(ldir)[1], 1))
        spec = tf.tile(spec, (1, 1, 3))  # becasue they are achromatic
        # Combine specular and Lambertian components
        brdf = albedo[:, None, :] / np.pi + spec * brdf_scale
        return brdf  # NxLx3

    def compute_loss(self, pred, gt, **kwargs):
        """Additional priors on light probes.
        """
        normal_loss_weight = self.config.getfloat(
            'DEFAULT', 'normal_loss_weight')
        lvis_loss_weight = self.config.getfloat('DEFAULT', 'lvis_loss_weight')
        smooth_use_l1 = self.config.getboolean('DEFAULT', 'smooth_use_l1')
        light_tv_weight = self.config.getfloat('DEFAULT', 'light_tv_weight')
        light_achro_weight = self.config.getfloat('DEFAULT', 'light_achro_weight')
        smooth_loss = tf.keras.losses.MAE if smooth_use_l1 else tf.keras.losses.MSE
        #
        mode = kwargs.pop('mode')
        normal_jitter = kwargs.pop('normal_jitter')
        lvis_jitter = kwargs.pop('lvis_jitter')
        albedo_jitter = kwargs.pop('albedo_jitter')
        brdf_prop_jitter = kwargs.pop('brdf_prop_jitter')
        #
        alpha, rgb_gt = gt['alpha'], gt['rgb']
        rgb_pred = pred['rgb']
        normal_pred, normal_gt = pred['normal'], gt['normal']
        lvis_pred, lvis_gt = pred['lvis'], gt['lvis']
        albedo_pred = pred['albedo']
        brdf_prop_pred = pred['brdf']
        # Composite prediction and ground truth onto backgrounds
        bg = tf.ones_like(rgb_gt) if self.white_bg else tf.zeros_like(rgb_gt)
        rgb_pred = imgutil.alpha_blend(rgb_pred, alpha, tensor2=bg)
        rgb_gt = imgutil.alpha_blend(rgb_gt, alpha, tensor2=bg)

        # RGB recon. loss is always here
        loss = tf.keras.losses.MSE(rgb_gt, rgb_pred)  # N
        # If validation, just MSE -- return immediately
        if mode == 'vali':
            return loss

        bg = tf.ones_like(normal_gt) if self.white_bg else tf.zeros_like(normal_gt)
        normal_pred = imgutil.alpha_blend(normal_pred, alpha, tensor2=bg)
        normal_gt = imgutil.alpha_blend(normal_gt, alpha, tensor2=bg)
        bg = tf.ones_like(lvis_gt) if self.white_bg else tf.zeros_like(lvis_gt)
        lvis_pred = imgutil.alpha_blend(lvis_pred, alpha, tensor2=bg)
        lvis_gt = imgutil.alpha_blend(lvis_gt, alpha, tensor2=bg)

        # If we modify the geometry
        if self.shape_mode in ('scratch', 'finetune'):
            # Predicted values should be close to NeRF values
            normal_loss = tf.keras.losses.MSE(normal_gt, normal_pred)  # N
            lvis_loss = tf.keras.losses.MSE(lvis_gt, lvis_pred)  # N
            loss += normal_loss_weight * normal_loss
            loss += lvis_loss_weight * lvis_loss
            # Predicted values should be smooth
            if normal_jitter is not None:
                normal_smooth_loss = smooth_loss(normal_pred, normal_jitter)  # N
                loss += self.normal_smooth_weight * normal_smooth_loss
            if lvis_jitter is not None:
                lvis_smooth_loss = smooth_loss(lvis_pred, lvis_jitter)  # N
                loss += self.lvis_smooth_weight * lvis_smooth_loss
        # Albedo should be smooth
        if albedo_jitter is not None:
            albedo_smooth_loss = smooth_loss(albedo_pred, albedo_jitter)  # N
            loss += self.albedo_smooth_weight * albedo_smooth_loss
        # BRDF property should be smooth
        if brdf_prop_jitter is not None:
            brdf_smooth_loss = smooth_loss(brdf_prop_pred, brdf_prop_jitter)  # N
            loss += self.brdf_smooth_weight * brdf_smooth_loss
        # Light should be smooth
        if mode == 'train':
            light = self.light
            # Spatial TV penalty
            if light_tv_weight > 0:
                dx = light - tf.roll(light, 1, 1)
                dy = light - tf.roll(light, 1, 0)
                tv = tf.reduce_sum(dx ** 2 + dy ** 2)
                loss += light_tv_weight * tv
            # Cross-channel TV penalty
            if light_achro_weight > 0:
                dc = light - tf.roll(light, 1, 2)
                tv = tf.reduce_sum(dc ** 2)
                loss += light_achro_weight * tv
        loss = tf.debugging.check_numerics(loss, "Loss")
        return loss

    def _brdf_prop_as_img(self, brdf_prop):
        """Z in learned BRDF.

        Input and output are both NumPy arrays, not tensors.
        """
        # Get min. and max. from seen BRDF Zs
        seen_z = self.brdf_model.latent_code.z
        seen_z = seen_z.numpy()
        seen_z_rgb = seen_z[:, :3]
        min_ = seen_z_rgb.min()
        max_ = seen_z_rgb.max()
        range_ = max_ - min_
        assert range_ > 0, "Range of seen BRDF Zs is 0"
        # Clip predicted values and scale them to [0, 1]
        z_rgb = brdf_prop[:, :, :3]
        z_rgb = np.clip(z_rgb, min_, max_)
        z_rgb = (z_rgb - min_) / range_
        return z_rgb

    def vis_batch(self, data_dict, outdir, mode='train', dump_raw_to=None,
                  light_vis_h=256, olat_vis=False, alpha_thres=0.8):
        # Visualize estimated lighting
        if mode == 'vali':
            # The same for all batches/views, so do it just once
            light_vis_path = join(dirname(outdir), 'pred_light.png')
            if not exists(light_vis_path):
                lightutil.vis_light(self.light, outpath=light_vis_path, h=light_vis_h)
        # Do what parent does
        self._validate_mode(mode)
        # Shortcircuit if training because rays are randomly sampled and
        # therefore very likely don't form a complete image
        if mode == 'train':
            return
        hw = data_dict.pop('hw')[0, :]
        hw = tuple(hw.numpy())
        id_ = data_dict.pop('id')[0]
        id_ = tutil.eager_tensor_to_str(id_)
        # To NumPy and reshape back to images
        for k, v in data_dict.items():
            if v is None:
                continue  # no-op
            v_ = v.numpy()
            if k in ('pred_rgb_olat', 'pred_rgb_probes'):
                v_ = v_.reshape(hw + (v_.shape[1], 3))
            elif k.endswith(('rgb', 'albedo', 'normal')):
                v_ = v_.reshape(hw + (3,))
            elif k.endswith(('occu', 'depth', 'disp', 'alpha')):
                v_ = v_.reshape(hw)
            elif k.endswith('brdf'):
                v_ = v_.reshape(hw + (-1,))
            elif k.endswith('lvis'):
                v_ = v_.reshape(hw + (v_.shape[1],))
            else:
                raise NotImplementedError(k)
            data_dict[k] = v_

        # Write images
        img_dict = {}
        alpha = data_dict['pred_alpha']
        # alpha = data_dict['gt_alpha']
        alpha[alpha < alpha_thres] = 0  # stricter compositing

        for k, v in data_dict.items():
            if k.endswith('rgb'):  # HxWx3
                bg = np.ones_like(v) if self.white_bg else np.zeros_like(v)
                img = imgutil.alpha_blend(v, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)

            # Normals
            elif k.endswith('normal'):
                v_ = (v + 1) / 2  # [-1, 1] to [0, 1]
                bg = np.ones_like(v_) if self.white_bg else np.zeros_like(v_)
                img = imgutil.alpha_blend(v_, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
            # Albedo
            elif k.endswith('albedo'):
                bg = np.ones_like(v) if self.white_bg else np.zeros_like(v)
                v_gamma = v ** (1 / 2.2)
                img = imgutil.alpha_blend(v_gamma, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
            # Light visibility
            elif k.endswith('lvis'):
                mean = np.mean(v, axis=2)  # NOTE: average across all lights
                bg = np.ones_like(mean) if self.white_bg \
                    else np.zeros_like(mean)
                img = imgutil.alpha_blend(mean, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
                # Optionally, visualize per-light vis.
                if olat_vis:
                    for i in tqdm(
                            range(4 if self.debug else v.shape[2] // 2),  # half
                            desc="Writing Per-Light Visibility (%s)" % k):
                        v_olat = v[:, :, i]
                        ij = np.unravel_index(i, self.light_res)
                        k_olat = k + '_olat_%04d-%04d' % ij
                        img = imgutil.alpha_blend(v_olat, alpha, bg)
                        img_dict[k_olat] = xm.io.img.write_arr(
                            img, join(outdir, k_olat + '.png'), clip=True)
            # BRDF property
            elif k.endswith('brdf'):
                v_ = self._brdf_prop_as_img(v)
                bg = np.ones_like(v_) if self.white_bg else np.zeros_like(v_)
                img = imgutil.alpha_blend(v_, alpha, bg)
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)
            # Everything else

            else:
                img = v
                img_dict[k] = xm.io.img.write_arr(
                    img, join(outdir, k + '.png'), clip=True)

        np.save(join(outdir,'light'), self.novel_probes_uint)

        # Shortcircuit if testing because there will be no ground truth for
        # us to make .apng comparisons
        if mode == 'test':
            # Write metadata that doesn't require ground truth (e.g., view name)
            metadata = {'id': id_}
            ioutil.write_json(metadata, join(outdir, 'metadata.json'))
            return
        # Make .apng
        put_text_kwargs = {
            'label_top_left_xy': (
                int(self.put_text_param['text_loc_ratio'] * hw[1]),
                int(self.put_text_param['text_loc_ratio'] * hw[0])),
            'font_size': int(self.put_text_param['text_size_ratio'] * hw[0]),
            'font_color': (0, 0, 0) if self.white_bg else (1, 1, 1),
            'font_ttf': self.put_text_param['font_path']}
        im1 = xm.vis.text.put_text(
            img_dict['gt_rgb'], "Ground Truth", **put_text_kwargs)
        im2 = xm.vis.text.put_text(
            img_dict['pred_rgb'], "Prediction", **put_text_kwargs)
        xm.vis.anim.make_anim(
            (im1, im2), outpath=join(outdir, 'pred-vs-gt_rgb.apng'))
        if self.shape_mode != 'nerf':
            im1 = xm.vis.text.put_text(
                img_dict['gt_normal'], "NeRF", **put_text_kwargs)
            im2 = xm.vis.text.put_text(
                img_dict['pred_normal'], "Prediction", **put_text_kwargs)
            xm.vis.anim.make_anim(
                (im1, im2), outpath=join(outdir, 'pred-vs-gt_normal.apng'))
            im1 = xm.vis.text.put_text(
                img_dict['gt_lvis'], "NeRF", **put_text_kwargs)
            im2 = xm.vis.text.put_text(
                img_dict['pred_lvis'], "Prediction", **put_text_kwargs)
            xm.vis.anim.make_anim(
                (im1, im2), outpath=join(outdir, 'pred-vs-gt_lvis.apng'))
        # Write metadata (e.g., view name, PSNR, etc.)
        psnr = self.psnr(img_dict['gt_rgb'], img_dict['pred_rgb'])
        metadata = {'id': id_, 'psnr': psnr}
        ioutil.write_json(metadata, join(outdir, 'metadata.json'))

    def compile_batch_vis(self, batch_vis_dirs, outpref, mode='train', fps=12):
        viewer_prefix = self.config.get('DEFAULT', 'viewer_prefix')
        self._validate_mode(mode)
        # Shortcircuit if training (same reason as above)
        if mode == 'train':
            return None
        # Validation or testing
        if mode == 'vali':
            outpath = outpref + '.html'
            self._compile_into_webpage(batch_vis_dirs, outpath)
        else:
            outpath = outpref + '.mp4'
            self._compile_into_video(batch_vis_dirs, outpath, fps=fps)
        view_at = viewer_prefix + outpath
        return view_at  # to be logged into TensorBoard

    def _compile_into_webpage(self, batch_dirs, out_html):
        rows, caps, types = [], [], []
        # For each batch (which has just one sample)
        for batch_dir in batch_dirs:
            metadata_path = join(batch_dir, 'metadata.json')
            metadata = ioutil.read_json(metadata_path)
            metadata = str(metadata)
            row = [metadata, join(batch_dir, 'pred-vs-gt_rgb.apng'),
                   join(batch_dir, 'pred_rgb.png'), join(batch_dir, 'pred_albedo.png'),
                   join(batch_dir, 'pred_brdf.png')]
            rowcaps = ["Metadata", "RGB", "RGB (pred.)", "Albedo (pred.)",
                       "BRDF (pred.)"]
            rowtypes = ['text', 'image', 'image', 'image', 'image']
            if self.shape_mode == 'nerf':
                row.append(join(batch_dir, 'gt_normal.png'))
                rowcaps.append("Normal (NeRF)")
                rowtypes.append('image')
            else:
                row.append(join(batch_dir, 'pred-vs-gt_normal.apng'))
                rowcaps.append("Normal")
                rowtypes.append('image')
                row.append(join(batch_dir, 'pred_normal.png'))
                rowcaps.append("Normal (pred.)")
                rowtypes.append('image')
            if self.shape_mode == 'nerf':
                row.append(join(batch_dir, 'gt_lvis.png'))
                rowcaps.append("Light Visibility (NeRF)")
                rowtypes.append('image')
            else:
                row.append(join(batch_dir, 'pred-vs-gt_lvis.apng'))
                rowcaps.append("Light Visibility")
                rowtypes.append('image')
                row.append(join(batch_dir, 'pred_lvis.png'))
                rowcaps.append("Light Visibility (pred.)")
                rowtypes.append('image')
            #
            rows.append(row)
            caps.append(rowcaps)
            types.append(rowtypes)
        n_rows = len(rows)
        assert n_rows > 0, "No row"
        # Write HTML
        bg_color = 'white' if self.white_bg else 'black'
        text_color = 'black' if self.white_bg else 'white'
        html = xm.vis.html.HTML(bgcolor=bg_color, text_color=text_color)
        img_table = html.add_table()
        for r, rcaps, rtypes in zip(rows, caps, types):
            img_table.add_row(r, rtypes, captions=rcaps)
        html_save = xm.decor.colossus_interface(html.save)
        html_save(out_html)

    def _compile_into_video(self, batch_dirs, out_mp4, fps=12, fixed_view=False):
        if fixed_view:
            data_root = self.config.get('DEFAULT', 'data_root')
        else:
            data_root = self.config.get('DEFAULT', 'shape_root')
        # Assume batch directory order is the right view order
        batch_dirs = sorted(batch_dirs)
        if self.debug:
            batch_dirs = batch_dirs[:10]
        # Tonemap and visualize all lighting conditions used
        frames = []
        # View synthesis
        for batch_dir in tqdm(batch_dirs, desc="View Synthesis"):
            frame = visutil.make_frame(batch_dir, ('nn', 'rgb'),
                                       data_root=data_root, put_text_param=self.put_text_param,
                                       rgb_embed_light=True)
            # To guard against missing buffer, which makes the frame None
            if frame is not None:
                frames.append(frame)
        xm.vis.video.make_video(frames, outpath=out_mp4, fps=fps)
