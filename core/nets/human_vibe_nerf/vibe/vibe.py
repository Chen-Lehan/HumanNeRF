# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import TemporalEncoder, Regressor, HMR, Bottleneck

EXCLUDE_LAYERS = [
    'regressor.smpl.betas',
    'regressor.smpl.global_orient',
    'regressor.smpl.body_pose',
    'regressor.smpl.faces_tensor',
    'regressor.smpl.v_template',
    'regressor.smpl.shapedirs',
    'regressor.smpl.J_regressor',
    'regressor.smpl.posedirs',
    'regressor.smpl.parents',
    'regressor.smpl.lbs_weights',
    'regressor.smpl.J_regressor_extra',
    'regressor.smpl.vertex_joint_selector.extra_joints_idxs',
]

class VIBE(nn.Module):
    def __init__(
            self,
            pretrained,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
    ):

        super(VIBE, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size 

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        self.hmr = HMR(Bottleneck, [3, 4, 6, 3])

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        hmr_pretrained_path, gru_pretrained_path = pretrained['hmr'], pretrained['gru']
        
        hmr_pretrained_dict = torch.load(hmr_pretrained_path, map_location='cpu')['model']
        self.hmr.load_state_dict(hmr_pretrained_dict, strict=False)
        print(f'=> loaded pretrained spin model from \'{hmr_pretrained_path}\'')

        gru_pretrained_dict = torch.load(gru_pretrained_path, map_location='cpu')['gen_state_dict']
        self.load_state_dict(gru_pretrained_dict, strict=False)
        print(f'=> loaded pretrained vibe model from \'{gru_pretrained_path}\'')


    def forward(self, input):
        # input size NTHWC or THWC
        if len(input.shape) == 5:
            batch_size, seqlen, H, W, C = input.shape[:]
            input = input.reshape(-1, H, W, C)
        else:
            batch_size = 1
            seqlen, H, W, C = input.shape[:]

        assert len(input.shape) == 4

        inf = self.hmr(input).reshape(batch_size, seqlen, -1)
        feature = self.encoder(inf)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature)
        for s in smpl_output:
            s['cam']   = s['cam'].reshape(batch_size, seqlen, -1)
            s['shape'] = s['shape'].reshape(batch_size, seqlen, -1)
            s['rmtx']  = s['rmtx'].reshape(batch_size, seqlen, -1, 3, 3)
            s['pose']  = s['pose'].reshape(batch_size, seqlen, -1, 3)

        return smpl_output
