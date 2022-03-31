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

from os.path import join, exists
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from third_party.xiuminglib import xiuminglib as xm
from nerfactor.networks import mlp
from nerfactor.networks.embedder import Embedder, Embedder_ipe
from nerfactor.util import logging as logutil, math as mathutil, io as ioutil, img as imgutil, config as configutil
from nerfactor import models

from nerfactor.models.base import Model as BaseModel

logger = logutil.Logger(loggee="models/nerf")


class Model(BaseModel):
    def __init__(self, config, debug=False):

        super().__init__(config, debug=debug)
        self.use_views = self.config.getboolean('DEFAULT', 'use_views')
        self.white_bg = self.config.getboolean('DEFAULT', 'white_bg')
        self.ipe = self.config.getboolean('DEFAULT', 'ipe')
        if self.config.get('DEFAULT', 'density_act') == 'softplus':
            self.density_activation = tf.nn.softplus
            self.density_bias = self.config.getfloat('DEFAULT', 'density_bias')
        else:
            self.density_activation = tf.nn.relu
            self.density_bias = 0.0

        if self.config.getboolean('DEFAULT', 'widened'):
            self.widened = True
            self.rgb_padding = self.config.getfloat('DEFAULT', 'rgb_padding')
        else:
            self.widened = False
        # Embedders
        self.embedder = self._init_embedder()
        # Network components
        if self.ipe:
            self.n_samples_coarse = self.config.getint('DEFAULT', 'ipe_n_samples_coarse') + 1
            self.n_samples_fine = self.config.getint('DEFAULT', 'ipe_n_samples_fine') + 1
            self.ipe_lambda = self.config.getfloat('DEFAULT', 'ipe_coarse_weight')

            self.net = {}
            for k, v in self._init_net().items():
                self.net['integrated_' + k] = v
        else:
            self.n_samples_coarse = self.config.getint('DEFAULT', 'n_samples_coarse')
            self.n_samples_fine = self.config.getint('DEFAULT', 'n_samples_fine')
            self.net = {}
            for k, v in self._init_net().items():
                self.net['coarse_' + k] = v
            if self.n_samples_fine > 0:
                for k, v in self._init_net().items():
                    self.net['fine_' + k] = v

        self.psnr = xm.metric.PSNR('uint8')

        bg_config_ini = configutil.get_config_ini(self.config.get('DEFAULT', 'bg_ckpt'))
        bg_config = ioutil.read_config(bg_config_ini)
        self.bg_near = bg_config.getfloat('DEFAULT', 'near')
        self.bg_far = bg_config.getfloat('DEFAULT', 'far')
        model_name = bg_config.get('DEFAULT', 'model')
        Model = models.get_model_class(model_name)
        self.bg_net = Model(bg_config, False)
        ioutil.restore_model(self.bg_net, self.config.get('DEFAULT', 'bg_ckpt'))
        self.bg_net.trainable = False

    def _init_net(self):
        mlp_width = self.config.getint('DEFAULT', 'mlp_width')
        enc_depth = self.config.getint('DEFAULT', 'enc_depth')
        act = self.config.get('DEFAULT', 'act', fallback='relu')
        enc = mlp.Network([mlp_width] * enc_depth, act=[act] * enc_depth, skip_at=[enc_depth // 2])
        net = {'enc': enc}
        # Shortcircuit if not using viewing directions
        if not self.use_views:
            net['rgbs_out'] = mlp.Network([4], act=[None])  # needs different activations
            return net
        # Using viewing directions
        net['sigma_out'] = mlp.Network([1], act=[None])  # ReLU later
        net['bottleneck'] = mlp.Network([mlp_width], act=[None])  # no act.
        net['rgb_out'] = mlp.Network([mlp_width // 2, 3], act=[act, None])  # sigmoid later
        return net

    def _init_embedder(self):
        if self.ipe:
            L = self.config.getint('DEFAULT', 'L')
            n_freqs_view = self.config.getint('DEFAULT', 'n_freqs_view')
            # Position embedder
            kwargs = {
                'in_dims': 3,
                'log2_max_freq': L - 1,
                'n_freqs': L,
                'log_sampling': True,
                'periodic_func': [tf.math.sin, tf.math.cos]}
            embedder_xyz = Embedder_ipe(**kwargs)
            # View direction embedder
            if self.use_views:
                kwargs['log2_max_freq'] = n_freqs_view - 1
                kwargs['n_freqs'] = n_freqs_view
                embedder_view = Embedder(**kwargs)
            else:
                embedder_view = tf.identity
            embedder = {'xyz': embedder_xyz, 'view': embedder_view}

        else:
            pos_enc = self.config.getboolean('DEFAULT', 'pos_enc')
            n_freqs_xyz = self.config.getint('DEFAULT', 'n_freqs_xyz')
            n_freqs_view = self.config.getint('DEFAULT', 'n_freqs_view')
            # Shortcircuit if not using embedders
            if not pos_enc:
                embedder_xyz = tf.identity
                embedder_view = tf.identity
                embedder = {'xyz': embedder_xyz, 'view': embedder_view}
                return embedder
            # Position embedder
            kwargs = {
                'incl_input': True,
                'in_dims': 3,
                'log2_max_freq': n_freqs_xyz - 1,
                'n_freqs': n_freqs_xyz,
                'log_sampling': True,
                'periodic_func': [tf.math.sin, tf.math.cos]}
            embedder_xyz = Embedder(**kwargs)
            # View direction embedder
            if self.use_views:
                kwargs['log2_max_freq'] = n_freqs_view - 1
                kwargs['n_freqs'] = n_freqs_view
                embedder_view = Embedder(**kwargs)
            else:
                embedder_view = tf.identity
            embedder = {'xyz': embedder_xyz, 'view': embedder_view}
        return embedder

    def call(self, batch, mode='train'):
        self._validate_mode(mode)
        id_, hw, rayo, rayd, rgb, radii, tmin, tmax, local_mat, local_tr = batch  # all flattened
        pred_coarse, pred_fine = self._render_rays(rayo, rayd, radii, tmin, tmax, local_mat, local_tr, mode=mode)

        # Prepare values to return
        pred = {
            'coarse': pred_coarse['rgb'],
            'fine': pred_fine.get('rgb', None)}
        gt = rgb
        loss_kwargs = {}
        to_vis = {'id': id_, 'hw': hw, 'gt_rgb': gt}
        # NOTE: gt_* are placeholders for test points
        for k, v in pred_coarse.items():
            to_vis['coarse_' + k] = v
        for k, v in pred_fine.items():
            to_vis['fine_' + k] = v
        return pred, gt, loss_kwargs, to_vis

    @staticmethod
    def gen_z(near, far, n_samples, lin_in_disp=False, perturb=False):
        near = tf.expand_dims(near, axis=1)
        far = tf.expand_dims(far, axis=1)
        t = tf.linspace(0., 1., n_samples)
        t = tf.expand_dims(t, axis=0)
        if lin_in_disp:
            # Sample linearly in inverse depth (disparity)
            z = 1. / (1. / near * (1. - t) + 1. / far * t)
        else:
            z = near * (1. - t) + far * t
        # z = tf.broadcast_to(z, (n_rays, len(z)))
        # Perturb sampling along each ray
        if perturb:
            mid = .5 * (z[:, 1:] + z[:, :-1])
            upper = tf.concat([mid, z[:, -1:]], -1)
            lower = tf.concat([z[:, :1], mid], -1)
            t_rand = tf.random.uniform(z.shape)
            z = lower + (upper - lower) * t_rand
        return z

    @staticmethod
    def gen_z_fine(z_coarse, weights, n_samples_fine, perturb=False):
        mid = .5 * (z_coarse[:, 1:] + z_coarse[:, :-1])
        z_fine = mathutil.inv_transform_sample(mid, weights[..., 1:-1], n_samples_fine, det=not perturb)
        z_fine = tf.stop_gradient(z_fine)
        # Obtain all points to evaluate the model at
        z_all = tf.sort(tf.concat((z_coarse, z_fine), -1), -1)
        return z_all

    @staticmethod
    def gen_z_fine_ipe(z_coarse, weights, n_samples_fine, perturb=False):
        resample_padding = 0.01
        weights_pad = tf.concat([weights[..., :1], weights, weights[..., -1:], ], axis=-1)
        weights_max = tf.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])
        # Add in a constant (the sampling function will renormalize the PDF).
        weights = weights_blur + resample_padding

        eps = 1e-5
        weight_sum = tf.math.reduce_sum(weights, axis=-1, keepdims=True)
        padding = tf.maximum(0.0, eps - weight_sum)
        weights += padding / weights.shape[-1]
        weight_sum += padding

        pdf = weights / weight_sum
        cdf = tf.minimum(1.0, tf.cumsum(pdf[..., :-1], axis=-1))
        cdf = tf.concat([tf.zeros(list(cdf.shape[:-1]) + [1]), cdf, tf.ones(list(cdf.shape[:-1]) + [1])], axis=-1)

        if not perturb:
            u = tf.linspace(0., 1. - tf.keras.backend.epsilon(), n_samples_fine)
            u = tf.broadcast_to(u, cdf.shape[:-1] + (n_samples_fine,))
        else:
            s = 1 / n_samples_fine
            u = tf.cast(tf.range(n_samples_fine), tf.float32) * s
            u += tf.random.uniform(list(cdf.shape[:-1]) + [n_samples_fine], maxval=s - tf.keras.backend.epsilon())
            # `u` is in [0, 1) --- it can be zero, but it can never be 1.
            u = tf.minimum(u, 1. - tf.keras.backend.epsilon())

        ind = tf.searchsorted(cdf, u, side='right')  # (n_rays, n_samples)
        below = tf.maximum(0, ind - 1)
        above = tf.minimum(ind, cdf.shape[-1] - 1)
        ind_g = tf.stack((below, above), -1)  # (n_rays, n_samples, 2)
        cdf_g = tf.gather(cdf, ind_g, axis=-1, batch_dims=len(ind_g.shape) - 2)
        val_g = tf.gather(z_coarse, ind_g, axis=-1, batch_dims=len(ind_g.shape) - 2)
        denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]  # (n_rays, n_samples)
        denom = tf.where(denom < eps, tf.ones_like(denom), denom)
        t = (u - cdf_g[:, :, 0]) / denom
        samples = val_g[:, :, 0] + t * (val_g[:, :, 1] - val_g[:, :, 0])
        return samples

    def _render_rays(self, rayo, rayd, radii, tmin, tmax, local_mat, local_tr, mode='train'):
        lin_in_disp = self.config.getboolean('DEFAULT', 'lin_in_disp')
        if mode == 'train':
            perturb = self.config.getboolean('DEFAULT', 'perturb')
        # NOTE: at validation/test time, do not randomize
        else:
            perturb = False
        # Normalize ray directions
        view_dir = tf.linalg.l2_normalize(rayd, axis=1)
        rgb_bg, _, _, _ = self.bg_net.obj_call(rayo, rayd, radii)

        if self.ipe:
            z = self.gen_z(tmin, tmax, self.n_samples_coarse, lin_in_disp=lin_in_disp, perturb=perturb)
            mean, covs = self.ipe_cast(z, rayo, rayd, radii)
            mean = tf.linalg.matmul(mean, local_mat) + local_tr[:, None]
            views = tf.broadcast_to(view_dir[:, None, :], mean.shape)
            rgbs = self._eval_nerf_at_ipe(mean, covs, views)
        else:
            z = self.gen_z(tmin, tmax, self.n_samples_coarse, lin_in_disp=lin_in_disp, perturb=perturb)
            pts = rayo[:, None, :] + rayd[:, None, :] * z[:, :, None]  # (n_rays, n_samples, 3)
            pts = tf.linalg.matmul(pts, local_mat) + local_tr[:, None]
            views = tf.broadcast_to(view_dir[:, None, :], pts.shape)
            rgbs = self._eval_nerf_at(pts, views, use_fine=False)

        rgb, occu, depth, disp, weights = self._accumulate(rgbs, z, rayd, rgb_bg)
        pred_coarse = {'rgb': rgb, 'occu': occu, 'depth': depth, 'disp': disp}

        # Shortcircuit if not using a fine model
        if self.n_samples_fine <= 0:
            pred_fine = {}
            return pred_coarse, pred_fine

        if self.ipe:
            z = self.gen_z_fine_ipe(z, weights, self.n_samples_fine, perturb=perturb)
            mean, covs = self.ipe_cast(z, rayo, rayd, radii)
            mean = tf.linalg.matmul(mean, local_mat) + local_tr[:, None]
            views = tf.broadcast_to(view_dir[:, None, :], mean.shape)
            rgbs = self._eval_nerf_at_ipe(mean, covs, views)
        else:
            z = self.gen_z_fine(z, weights, self.n_samples_fine, perturb=perturb)
            pts = rayo[:, None, :] + rayd[:, None, :] * z[:, :, None]  # (n_rays, n_samples, 3)
            pts = tf.linalg.matmul(pts, local_mat) + local_tr[:, None]
            views = tf.broadcast_to(view_dir[:, None, :], pts.shape)
            rgbs = self._eval_nerf_at(pts, views, use_fine=False)

        rgb, occu, depth, disp, _ = self._accumulate(rgbs, z, rayd, rgb_bg)
        pred_fine = {'rgb': rgb, 'occu': occu, 'depth': depth, 'disp': disp}
        return pred_coarse, pred_fine

    def ipe_cast(self, z, rayo, rayd, radii):
        t0 = z[..., :-1]
        t1 = z[..., 1:]
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2)
        t_var = (hw ** 2) / 3 - (4 / 15) * (hw ** 4 * (12 * mu ** 2 - hw ** 2)) / (3 * mu ** 2 + hw ** 2) ** 2
        r_var = radii ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 * (hw ** 4) / (3 * mu ** 2 + hw ** 2))
        mean = rayd[..., None, :] * t_mean[..., None]
        d_mag_sq = tf.maximum(1e-10, tf.reduce_sum(rayd ** 2, axis=-1, keepdims=True))
        d_outer_diag = rayd ** 2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return rayo[..., None, :] + mean, cov_diag

    def accumulate_sigma(self, sigma, z, rayd, noise_std=0., inf=1e10, accu_chunk=65536, is_activated=False):
        # Compute "distance" (in time) between each integration time along a ray
        dist = z[:, 1:] - z[:, :-1]
        # The "distance" from the last integration time is infinity
        if dist.shape != sigma.shape:
            dist = tf.concat((dist, tf.broadcast_to([inf], dist[:, :1].shape)), axis=-1)
        # should be redundant because rayd is# already normalized
        dist = dist * tf.linalg.norm(rayd[:, None, :], axis=-1)
        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts)
        noise = tf.random.normal(sigma.shape) * noise_std
        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point
        if is_activated:
            density = 1.0 - tf.exp(-sigma * dist)
        else:
            density = 1.0 - tf.exp(-self.density_activation(sigma + self.density_bias + noise) * dist)
        # Chunk by chunk to avoid OOM
        weights_chunks = []
        for i in range(0, density.shape[0], accu_chunk):
            end_i = min(density.shape[0], i + accu_chunk)
            density_chunk = density[i:end_i, :]
            # cumprod() is used to express the idea of the ray not having
            # reflected up to this sample yet
            weights_chunk = density_chunk * mathutil.safe_cumprod(1. - density_chunk)
            weights_chunks.append(weights_chunk)
        weights = tf.concat(weights_chunks, axis=0)
        return weights

    def _accumulate(self, rgbs, z, rayd, bg, eps=1e-10):
        noise_std = self.config.getfloat('DEFAULT', 'noise_std')
        accu_chunk = self.config.getint('DEFAULT', 'accu_chunk')
        # Compute weights along each ray from sigma volume
        sigma = rgbs[:, :, 3]
        weights = self.accumulate_sigma(sigma, z, rayd, noise_std=noise_std, accu_chunk=accu_chunk)
        # Extract RGB of each sample position along each ray
        rgb = rgbs[:, :, :3]
        rgb = tf.math.sigmoid(rgb)
        if self.widened:
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        if rgbs.shape[:2] != z.shape:
            z = 0.5 * (z[..., :-1] + z[..., 1:])
        rgb_chunks, depth_chunks, disp_chunks, occu_chunks = [], [], [], []
        for i in range(0, weights.shape[0], accu_chunk):
            end_i = min(weights.shape[0], i + accu_chunk)
            weights_chunk = weights[i:end_i, :]
            rgb_chunk = rgb[i:end_i, :, :]
            z_chunk = z[i:end_i, :]
            # Sum of weights along each ray, in [0, 1] up to numerical errors
            occu_chunk = tf.reduce_sum(weights_chunk, axis=-1)
            # Computed weighted color of each sample along each ray
            # (n_rays_chunk, 3)
            rgb_chunk = tf.reduce_sum(weights_chunk[:, :, None] * rgb_chunk, axis=-2)
            # Estimated depth is expected distance
            # (n_rays_chunk,)
            depth_chunk = tf.reduce_sum(weights_chunk * z_chunk, axis=-1)
            # Disparity is inverse depth
            denom = tf.maximum(depth_chunk, eps)
            disp_chunk = 1. / denom
            occu_chunks.append(occu_chunk)
            rgb_chunks.append(rgb_chunk)
            depth_chunks.append(depth_chunk)
            disp_chunks.append(disp_chunk)
        occu = tf.concat(occu_chunks, axis=0)
        rgb = tf.concat(rgb_chunks, axis=0)
        depth = tf.concat(depth_chunks, axis=0)
        disp = tf.concat(disp_chunks, axis=0)
        rgb = imgutil.alpha_blend(rgb, occu[:, None], bg)
        return rgb, occu, depth, disp, weights

    def _eval_nerf_at(self, pts, views, use_fine=False, reshape=True):
        mlp_chunk = self.config.getint('DEFAULT', 'mlp_chunk')
        pref = 'fine_' if use_fine else 'coarse_'
        enc = self.net[pref + 'enc']
        if reshape:
            pts_flat = tf.reshape(pts, (-1, 3))
            views_flat = tf.reshape(views, (-1, 3))
        else:
            pts_flat = pts
            views_flat = views
        rgbs_chunks = []
        for i in range(0, pts_flat.shape[0], mlp_chunk):
            end_i = min(pts_flat.shape[0], i + mlp_chunk)
            pts_chunk = pts_flat[i:end_i, :]
            views_chunk = views_flat[i:end_i, :]
            # Positional encoding
            pts_embed = self.embedder['xyz'](pts_chunk)
            views_embed = self.embedder['view'](views_chunk)
            # Evaluate NeRF
            if self.use_views:
                sigma_out = self.net[pref + 'sigma_out']
                bottleneck = self.net[pref + 'bottleneck']
                rgb_out = self.net[pref + 'rgb_out']
                feat = enc(pts_embed)
                sigma_flat = sigma_out(feat)
                feat = bottleneck(feat)
                feat_views = tf.concat((feat, views_embed), -1)
                rgb_flat = rgb_out(feat_views)
                rgbs_flat = tf.concat([rgb_flat, sigma_flat], -1)
            else:
                # Not using viewing directions
                rgbs_out = self.net[pref + 'rgbs_out']
                rgbs_flat = rgbs_out(enc(pts_embed))
            rgbs_chunks.append(rgbs_flat)
        rgbs = tf.concat(rgbs_chunks, axis=0)  # (n_rays, n_samples, -1)
        if reshape:
            rgbs = tf.reshape(rgbs, pts.shape[:2] + (4,))
        return rgbs

    def _eval_nerf_at_ipe(self, pts, diag_covs, views, reshape=True):
        mlp_chunk = self.config.getint('DEFAULT', 'mlp_chunk')
        pref = 'integrated_'
        enc = self.net[pref + 'enc']
        if reshape:
            pts_flat = tf.reshape(pts, (-1, 3))
            views_flat = tf.reshape(views, (-1, 3))
            diag_covs_flat = tf.reshape(diag_covs, (-1, 3))
        else:
            pts_flat = pts
            views_flat = views
            diag_covs_flat = diag_covs

        rgbs_chunks = []
        for i in range(0, pts_flat.shape[0], mlp_chunk):
            end_i = min(pts_flat.shape[0], i + mlp_chunk)
            pts_chunk = pts_flat[i:end_i, :]
            diag_covs_chunk = diag_covs_flat[i:end_i, :]
            views_chunk = views_flat[i:end_i, :]
            # Positional encoding
            pts_embed = self.embedder['xyz'](pts_chunk, diag_covs_chunk)
            views_embed = self.embedder['view'](views_chunk)
            # Evaluate NeRF
            if self.use_views:
                sigma_out = self.net[pref + 'sigma_out']
                bottleneck = self.net[pref + 'bottleneck']
                rgb_out = self.net[pref + 'rgb_out']
                feat = enc(pts_embed)
                sigma_flat = sigma_out(feat)
                feat = bottleneck(feat)
                feat_views = tf.concat((feat, views_embed), -1)
                rgb_flat = rgb_out(feat_views)
                rgbs_flat = tf.concat([rgb_flat, sigma_flat], -1)
            else:
                # Not using viewing directions
                rgbs_out = self.net[pref + 'rgbs_out']
                rgbs_flat = rgbs_out(enc(pts_embed))
            rgbs_chunks.append(rgbs_flat)
        rgbs = tf.concat(rgbs_chunks, axis=0)  # (n_rays, n_samples, -1)
        if reshape:
            rgbs = tf.reshape(rgbs, pts.shape[:2] + (4,))
        return rgbs

    def compute_loss(self, pred, gt, **kwargs):
        coarse, fine = pred['coarse'], pred['fine']
        # Accumulate loss
        loss = 0
        for weight, loss_func in self.wloss:
            if self.ipe:
                loss += weight * loss_func(gt, coarse, **kwargs) * self.ipe_lambda
            else:
                loss += weight * loss_func(gt, coarse, **kwargs)
            if fine is not None:
                loss += weight * loss_func(gt, fine, **kwargs)
        return loss

    def vis_batch(self, data_dict, outdir, mode='train', dump_raw_to=None,
                  text_loc_ratio=0.05, text_size_ratio=0.05):
        """Always visualizes results on a black background, regardless of
        whether the data have white or black backgrounds.
        """
        self._validate_mode(mode)
        # Shortcircuit if training because rays are randomly sampled and
        # therefore very likely don't form a complete image
        if mode == 'train':
            return
        hw = tuple(data_dict.pop('hw').numpy()[0, :])
        mask = data_dict.pop('mask').reshape(hw)
        id_ = data_dict.pop('id').numpy()[0].decode()

        # To NumPy and reshape back to images
        for k, v in data_dict.items():
            if k.endswith('rgb'):
                img_tmp = np.zeros(hw + (3,))
                img_tmp[np.where(mask)] = v.numpy()
                data_dict[k] = img_tmp
            elif k.endswith(('occu', 'depth', 'disp')):
                img_tmp = np.zeros(hw)
                img_tmp[np.where(mask)] = v.numpy()
                data_dict[k] = img_tmp
            else:
                raise NotImplementedError(k)
        # Write images
        img_dict = {}
        for k, v in data_dict.items():
            if k.endswith('depth'):
                img = (v - np.min(v)) / (np.max(v) - np.min(v))  # normalize
                alpha = data_dict[k.replace('depth', 'occu')]
                bg = np.ones_like(img) if self.white_bg else np.zeros_like(img)
                img = imgutil.alpha_blend(img, alpha, bg)
            # elif k.endswith('disp'):
            #     min_disp = 1 / np.max(v)
            #     max_disp = 1 / np.min(v)
            #     img = (v - min_disp) / (max_disp - min_disp)  # normalize
            #     alpha = data_dict[k.replace('disp', 'occu')]
            #     bg = np.ones_like(img) if self.white_bg else np.zeros_like(img)
            #     img = imgutil.alpha_blend(img, alpha, bg)
            elif k in ('coarse_occu', 'fine_occu'):
                img = 1 - v if self.white_bg else v
            else:  # RGB already [0, 1] and composited onto backgrounds
                img = v
            img_dict[k] = xm.io.img.write_arr(
                img, join(outdir, k + '.png'), clip=True)
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
                int(text_loc_ratio * hw[1]), int(text_loc_ratio * hw[0])),
            'font_size': int(text_size_ratio * hw[0]),
            'font_color': (0, 0, 0) if self.white_bg else (1, 1, 1),
            'font_ttf': xm.const.Path.open_sans_regular}
        im1 = xm.vis.text.put_text(
            img_dict['gt_rgb'], "Ground Truth", **put_text_kwargs)
        im2 = xm.vis.text.put_text(
            img_dict['fine_rgb'], "Prediction (fine)", **put_text_kwargs)
        xm.vis.anim.make_anim(
            (im1, im2), outpath=join(outdir, 'fine-vs-gt_rgb.apng'))
        im1 = xm.vis.text.put_text(
            img_dict['fine_rgb'], "Prediction (fine)", **put_text_kwargs)
        im2 = xm.vis.text.put_text(
            img_dict['coarse_rgb'], "Prediction (coarse)", **put_text_kwargs)
        xm.vis.anim.make_anim(
            (im1, im2), outpath=join(outdir, 'fine-vs-coarse_rgb.apng'))
        im1 = xm.vis.text.put_text(
            img_dict['fine_depth'], "Prediction (fine)", **put_text_kwargs)
        im2 = xm.vis.text.put_text(
            img_dict['coarse_depth'], "Prediction (coarse)", **put_text_kwargs)
        xm.vis.anim.make_anim(
            (im1, im2), outpath=join(outdir, 'fine-vs-coarse_depth.apng'))
        im1 = xm.vis.text.put_text(
            img_dict['fine_disp'], "Prediction (fine)", **put_text_kwargs)
        im2 = xm.vis.text.put_text(
            img_dict['coarse_disp'], "Prediction (coarse)", **put_text_kwargs)
        xm.vis.anim.make_anim(
            (im1, im2), outpath=join(outdir, 'fine-vs-coarse_disp.apng'))
        im1 = xm.vis.text.put_text(
            img_dict['fine_occu'], "Prediction (fine)", **put_text_kwargs)
        im2 = xm.vis.text.put_text(
            img_dict['coarse_occu'], "Prediction (coarse)", **put_text_kwargs)
        xm.vis.anim.make_anim(
            (im1, im2), outpath=join(outdir, 'fine-vs-coarse_occu.apng'))
        # Write metadata (e.g., view name, PSNR, etc.)
        psnr = self.psnr(img_dict['gt_rgb'], img_dict['fine_rgb'])
        metadata = {'id': id_, 'psnr': psnr}
        ioutil.write_json(metadata, join(outdir, 'metadata.json'))
        # Optionally dump raw to disk
        if dump_raw_to is not None:
            # ioutil.dump_dict_tensors(data_dict, dump_raw_to)
            pass

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

    def _compile_into_video(self, batch_dirs, out_mp4, fps=12):
        data_root = self.config.get('DEFAULT', 'data_root')
        # Load frames
        frames = {}
        for batch_dir in tqdm(batch_dirs, desc="Compiling visualized batches"):
            json_path = join(batch_dir, 'metadata.json')
            pred_path = join(batch_dir, 'fine_rgb.png')
            if not exists(json_path) or not exists(pred_path):
                logger.warn(
                    "Skipping because of missing files:\n\t%s" % batch_dir)
                continue
            # Metadata
            metadata = ioutil.read_json(json_path)
            id_ = metadata['id']
            # Prediction
            pred = xm.io.img.load(pred_path)
            # (Optional) Nearest neighbor real image
            nn_dir = join(data_root, 'test_phys_nn')
            nn_paths = xm.os.sortglob(nn_dir, id_ + '_nn_*.png')
            n_nn = len(nn_paths)
            if n_nn == 0:
                frame = pred
            elif n_nn == 1:
                nn_path = nn_paths[0]
                nn = xm.io.img.load(nn_path)
                frame = imgutil.hconcat((pred, nn))
            else:
                raise RuntimeError((
                                       "There must be either zero or one nearest neighbor for "
                                       "each test camera, but found %d") % n_nn)
            frames[id_] = frame
        # Make video
        frames_sorted = [frames[k] for k in sorted(frames)]
        xm.vis.video.make_video(frames_sorted, fps=fps, outpath=out_mp4)

    def _compile_into_webpage(self, batch_dirs, out_html):
        rows, caps, types = [], [], []
        # For each batch (which has just one sample)
        for batch_dir in batch_dirs:
            metadata_path = join(batch_dir, 'metadata.json')
            metadata = ioutil.read_json(metadata_path)
            metadata = str(metadata)
            row = [
                metadata,
                join(batch_dir, 'fine-vs-gt_rgb.apng'),
                join(batch_dir, 'fine-vs-coarse_rgb.apng'),
                join(batch_dir, 'fine-vs-coarse_depth.apng'),
                join(batch_dir, 'fine-vs-coarse_disp.apng'),
                join(batch_dir, 'fine-vs-coarse_occu.apng')]
            rowcaps = [
                "Metadata", "RGB", "RGB", "Depth", "Disparity", "Occupancy"]
            rowtypes = [
                'text', 'image', 'image', 'image', 'image', 'image', 'image']
            rows.append(row)
            caps.append(rowcaps)
            types.append(rowtypes)
        n_rows = len(rows)
        assert n_rows > 0, "No row"
        # Write HTML
        bg_color = 'white' if self.white_bg else 'black'
        text_color = 'black' if self.white_bg else 'white'
        html = xm.vis.html.HTML(bgcolor=bg_color, text_color=text_color)
        html.add_header("NeRF")
        img_table = html.add_table()
        for r, rcaps, rtypes in zip(rows, caps, types):
            img_table.add_row(r, rtypes, captions=rcaps)
        html_save = xm.decor.colossus_interface(html.save)
        html_save(out_html)
