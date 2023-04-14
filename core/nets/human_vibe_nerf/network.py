import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os.path as osp

from core.utils.network_util import MotionBasisComputer, RodriguesModule
from core.utils.body_util import SMPL_PARENT, BONE_STDS, HEAD_STDS, JOINT_STDS, SMPL_JOINT_IDX, TORSO_JOINTS
from core.nets.human_vibe_nerf.component_factory import \
    load_positional_embedder, \
    load_canonical_mlp, \
    load_mweight_vol_decoder, \
    load_pose_decoder, \
    load_non_rigid_motion_mlp, \
    load_vibe, \
    load_motion_discriminator

from smplx import SMPL, SMPLLayer

from configs import cfg


def skeleton_to_bbox(skeleton):
    min_xyz = torch.min(skeleton, dim=0)[0] - cfg.bbox_offset
    max_xyz = torch.max(skeleton, dim=0)[0] + cfg.bbox_offset

    return torch.stack([min_xyz, max_xyz])


def _construct_G(R_mtx, T):
    r""" Build 4x4 [R|T] matrix from rotation matrix, and translation vector
    
    Args:
        - R_mtx: Array (3, 3)
        - T: Array (3,)

    Returns:
        - Array (4, 4)
    """
    tmp = torch.zeros(1, 4)
    tmp[0, 3] = 1.
    G = torch.cat([torch.cat([R_mtx, T[None, :].T], dim=1), tmp])

    return G


def _std_to_scale_mtx(stds):
    r""" Build scale matrix from standard deviations
    
    Args:
        - stds: Array(3,)

    Returns:
        - Array (3, 3)
    """

    scale_mtx = torch.eye(3, dtype=torch.float32)
    scale_mtx[0][0] = 1.0/stds[0]
    scale_mtx[1][1] = 1.0/stds[1]
    scale_mtx[2][2] = 1.0/stds[2]

    return scale_mtx


def _to_skew_matrix(v):
    r""" Compute the skew matrix given a 3D vectors.

    Args:
        - v: Array (3, )

    Returns:
        - Array (3, 3)

    """
    vx, vy, vz = v.ravel()
    return torch.tensor([[0, -vz, vy],
                         [vz, 0, -vx],
                         [-vy, vx, 0]])


def _to_skew_matrices(batch_v):
    r""" Compute the skew matrix given 3D vectors. (batch version)

    Args:
        - batch_v: Array (N, 3)

    Returns:
        - Array (N, 3, 3)

    """
    batch_size = batch_v.shape[0]
    skew_matrices = torch.zeros(size=(batch_size, 3, 3), dtype=torch.float32)

    for i in range(batch_size):
        skew_matrices[i] = _to_skew_matrix(batch_v[i])

    return skew_matrices


def _get_rotation_mtx(v1, v2):
    r""" Compute the rotation matrices between two 3D vector. (batch version)
    
    Args:
        - v1: Array (N, 3)
        - v2: Array (N, 3)

    Returns:
        - Array (N, 3, 3)

    Reference:
        https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """

    batch_size = v1.shape[0]
    # v1 = v1 / torch.clip(torch.norm(v1, dim=-1, keepdim=True), 1e-5, None)
    # v2 = v2 / torch.clip(torch.norm(v2, dim=-1, keepdim=True), 1e-5, None)
    v1 = v1 / torch.norm(v1)
    v2 = v2 / torch.norm(v2)
    
    normal_vec = torch.cross(v1, v2, dim=-1)
    cos_v = torch.zeros(size=(batch_size, 1))
    for i in range(batch_size):
        cos_v[i] = v1[i].dot(v2[i])

    skew_mtxs = _to_skew_matrices(normal_vec)
    
    Rs = torch.zeros(size=(batch_size, 3, 3), dtype=torch.float32)
    for i in range(batch_size):
        Rs[i] = torch.eye(3) + skew_mtxs[i] + \
                    (skew_mtxs[i].matmul(skew_mtxs[i])) * (1./(1. + cos_v[i]))
    
    return Rs


def get_canonical_global_tfms(canonical_joints):
    r""" Convert canonical joints to 4x4 global transformation matrix.
    
    Args:
        - canonical_joints: Array (Total_Joints, 3)

    Returns:
        - Array (Total_Joints, 4, 4)
    """
    total_bones = canonical_joints.shape[0]
    ident = torch.eye(3).expand(size=(total_bones, 3, 3)) # (Total_Joints, 3, 3)
    gtfms = torch.cat([ident, canonical_joints[:, :, None]], dim=2) # (Total_Joints, 3, 4)
    tmp = torch.zeros(size=(24, 1, 4))
    tmp[:, :, 3] = 1
    gtfms = torch.cat([gtfms, tmp], dim=1) # (Total_Joints, 4, 4)

    return gtfms


def body_pose_to_body_Ts(tpose_joints):
    r""" Convert body pose to global rotation matrix R and translation T.
    
    Args:
        - jangles (joint angles): Array (Total_Joints x 3, )
        - tpose_joints:           Array (Total_Joints, 3)

    Returns:
        - Rs: Array (Total_Joints, 3, 3)
        - Ts: Array (Total_Joints, 3)
    """

    total_joints = tpose_joints.shape[0]

    Ts = torch.zeros(size=[total_joints, 3], dtype=torch.float32)
    Ts[0] = tpose_joints[0,:]

    for i in range(1, total_joints):
        Ts[i] = tpose_joints[i,:] - tpose_joints[SMPL_PARENT[i], :]
    
    return Ts


def get_rays_from_KRT(H, W, K, R, T):
    r""" Sample rays on an image based on camera matrices (K, R and T)

    Args:
        - H: Integer
        - W: Integer
        - K: Tensor (3, 3)
        - R: Tensor (3, 3)
        - T: Tensor (3, )
        
    Returns:
        - rays_o: Array (H, W, 3)
        - rays_d: Array (H, W, 3)
    """

    # calculate the camera origin
    rays_o = -torch.matmul(T[None, :], R).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    xy1 = torch.from_numpy(xy1)
    pixel_camera = torch.matmul(xy1, torch.inverse(K).T)
    pixel_world = torch.matmul(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = torch.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d


def get_camera_parameters(pred_cam, img_size, global_orient):
    FOCAL_LENGTH = 5000.

    cam_intrinsics = torch.eye(3)
    cam_intrinsics[0, 0] = FOCAL_LENGTH
    cam_intrinsics[1, 1] = FOCAL_LENGTH
    cam_intrinsics[0, 2] = img_size / 2. 
    cam_intrinsics[1, 2] = img_size / 2.

    cam_s, cam_tx, cam_ty = pred_cam
    trans = [cam_tx, cam_ty, 2*FOCAL_LENGTH/(img_size*cam_s + 1e-9)]

    cam_extrinsics = torch.eye(4)
    cam_extrinsics[:3, 3] = torch.stack(trans)

    global_tfms = torch.eye(4)
    global_tfms[:3, :3] = global_orient

    return cam_intrinsics, torch.matmul(cam_extrinsics, global_tfms.T)


def rays_intersect_3d_bbox(bounds, ray_o, ray_d):
    r"""calculate intersections with 3d bounding box
        Args:
            - bounds: (2, 3)
            - ray_o: (N_rays, 3)
            - ray_d, (N_rays, 3)
        Output:
            - near: (N_VALID_RAYS, )
            - far: (N_VALID_RAYS, )
            - mask_at_box: (N_RAYS, )
    """

    bounds = bounds + torch.tensor([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None] # (N_rays, 2, 3)
    # calculate the step of intersections at six planes of the 3d bounding box
    ray_d[torch.abs(ray_d) < 1e-5] = 1e-5
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6) # (N_rays, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None] # (N_rays, 6, 3)
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))  # (N_rays, 6)
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2  #(N_rays, )
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3) # (N_VALID_rays, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = torch.norm(ray_d, dim=1)
    d0 = torch.norm(p_intervals[:, 0] - ray_o, dim=1) / norm_ray
    d1 = torch.norm(p_intervals[:, 1] - ray_o, dim=1) / norm_ray
    near = torch.minimum(d0, d1)
    far = torch.maximum(d0, d1)

    return near, far, mask_at_box


def _deform_gaussian_volume(
        grid_size, 
        bbox_min_xyz,
        bbox_max_xyz,
        center, 
        scale_mtx, 
        rotation_mtx):
    r""" Deform a standard Gaussian volume.
    
    Args:
        - grid_size:    Integer
        - bbox_min_xyz: Array (3, )
        - bbox_max_xyz: Array (3, )
        - center:       Array (3, )   - center of Gaussain to be deformed
        - scale_mtx:    Array (3, 3)  - scale of Gaussain to be deformed
        - rotation_mtx: Array (3, 3)  - rotation matrix of Gaussain to be deformed

    Returns:
        - Array (grid_size, grid_size, grid_size)
    """

    R = rotation_mtx
    S = scale_mtx

    # covariance matrix after scaling and rotation
    SIGMA = R.matmul(S).matmul(S).matmul(R.T)

    min_x, min_y, min_z = bbox_min_xyz
    max_x, max_y, max_z = bbox_max_xyz
    
    x_linspace = torch.linspace(0, 1, grid_size) * (max_x - min_x) + min_x
    y_linspace = torch.linspace(0, 1, grid_size) * (max_y - min_y) + min_y
    z_linspace = torch.linspace(0, 1, grid_size) * (max_z - min_z) + min_z

    zgrid, ygrid, xgrid = torch.meshgrid(x_linspace, y_linspace, z_linspace)
    grid = torch.stack([xgrid - center[0], 
                        ygrid - center[1], 
                        zgrid - center[2]],
                    dim=-1)

    dist = torch.einsum('abci, abci->abc', torch.einsum('abci, ij->abcj', grid, SIGMA), grid)

    return torch.exp(-1 * dist)


def approx_gaussian_bone_volumes(
    tpose_joints, 
    bbox_min_xyz, bbox_max_xyz,
    grid_size=32):
    r""" Compute approximated Gaussian bone volume.
    
    Args:
        - tpose_joints:  Array (Total_Joints, 3)
        - bbox_min_xyz:  Array (3, )
        - bbox_max_xyz:  Array (3, )
        - grid_size:     Integer
        - has_bg_volume: boolean

    Returns:
        - Array (Total_Joints + 1, 3, 3, 3)
    """

    total_joints = tpose_joints.shape[0]

    grid_shape = [grid_size] * 3

    calibrated_bone = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)[None, :]
    g_volumes = []
    for joint_idx in range(0, total_joints):
        gaussian_volume = torch.zeros(size=grid_shape, dtype=torch.float32)

        is_parent_joint = False
        for bone_idx, parent_idx in SMPL_PARENT.items():
            if joint_idx != parent_idx:
                continue

            S = _std_to_scale_mtx(BONE_STDS * 2.)
            if joint_idx in TORSO_JOINTS:
                S[0][0] *= 1/1.5
                S[2][2] *= 1/1.5

            start_joint = tpose_joints[SMPL_PARENT[bone_idx]]
            end_joint = tpose_joints[bone_idx]
            target_bone = (end_joint - start_joint)[None, :]

            R = _get_rotation_mtx(calibrated_bone, target_bone)[0]

            center = (start_joint + end_joint) / 2.0

            bone_volume = _deform_gaussian_volume(
                            grid_size, 
                            bbox_min_xyz,
                            bbox_max_xyz,
                            center, S, R)
            gaussian_volume = gaussian_volume + bone_volume

            is_parent_joint = True

        if not is_parent_joint:
            # The joint is not other joints' parent, meaning it is an end joint
            joint_stds = HEAD_STDS if joint_idx == SMPL_JOINT_IDX['head'] else JOINT_STDS
            S = _std_to_scale_mtx(joint_stds * 2.)

            center = tpose_joints[joint_idx]
            gaussian_volume = _deform_gaussian_volume(
                                grid_size, 
                                bbox_min_xyz,
                                bbox_max_xyz,
                                center, 
                                S, 
                                torch.eye(3, dtype=torch.float32))
            
        g_volumes.append(gaussian_volume)
    g_volumes = torch.stack(g_volumes, dim=0)

    # concatenate background weights
    bg_volume = 1.0 - torch.sum(g_volumes, dim=0, keepdims=True).clip(min=0.0, max=1.0)
    g_volumes = torch.cat([g_volumes, bg_volume], dim=0)
    g_volumes = g_volumes / torch.sum(g_volumes, dim=0, keepdims=True).clip(min=0.001)
    
    return g_volumes


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        if cfg.single_gpu == True:
            torch.cuda.set_device(cfg.primary_gpus[0])

        self.volume_size = cfg.mweight_volume.volume_size

        # pose detector
        self.pose_detector = load_vibe(cfg.vibe.module)(
            pretrained={
                'hmr': cfg.vibe.hmr_pretrained_path,
                'gru': cfg.vibe.gru_pretrained_path
            },
            seqlen=cfg.vibe.seqlen,
            batch_size=cfg.train.batch_size,
            n_layers=cfg.vibe.tgru.n_layers,
            hidden_size=cfg.vibe.tgru.hidden_size,
            add_linear=cfg.vibe.tgru.add_linear,
            bidirectional=cfg.vibe.tgru.bidirectional,
            use_residual=cfg.vibe.tgru.use_residual,
        )

        # motion discriminator
        self.motion_discriminator = load_motion_discriminator(cfg.motion_discriminator.module)(
            rnn_size=cfg.motion_discriminator.hidden_size,
            input_size=69,
            n_layers=cfg.motion_discriminator.n_layers,
            output_size=1,
            feature_pool=cfg.motion_discriminator.feature_pool,
        ) # TODO

        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(
                                        total_bones=cfg.total_bones)

        # motion weight volume
        self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(
            embedding_size=cfg.mweight_volume.embedding_size,
            volume_size=cfg.mweight_volume.volume_size,
            total_bones=cfg.total_bones
        )

        # non-rigid motion st positional encoding
        self.get_non_rigid_embedder = \
            load_positional_embedder(cfg.non_rigid_embedder.module)

        # non-rigid motion MLP
        _, non_rigid_pos_embed_size = \
            self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires, 
                                        cfg.non_rigid_motion_mlp.i_embed)
        self.non_rigid_mlp = \
            load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
                pos_embed_size=non_rigid_pos_embed_size,
                condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
                mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
                mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
                skips=cfg.non_rigid_motion_mlp.skips).to(cfg.primary_gpus[0])
        # self.non_rigid_mlp = \
        #     nn.DataParallel(
        #         self.non_rigid_mlp,
        #         device_ids=cfg.secondary_gpus,
        #         output_device=cfg.secondary_gpus[0])

        # canonical positional encoding
        get_embedder = load_positional_embedder(cfg.embedder.module)
        cnl_pos_embed_fn, cnl_pos_embed_size = \
            get_embedder(cfg.canonical_mlp.multires, 
                         cfg.canonical_mlp.i_embed)
        self.pos_embed_fn = cnl_pos_embed_fn

        # canonical mlp 
        skips = [4]
        self.cnl_mlp = \
            load_canonical_mlp(cfg.canonical_mlp.module)(
                input_ch=cnl_pos_embed_size, 
                mlp_depth=cfg.canonical_mlp.mlp_depth, 
                mlp_width=cfg.canonical_mlp.mlp_width,
                skips=skips).to(cfg.primary_gpus[0])
        # self.cnl_mlp = \
        #     nn.DataParallel(
        #         self.cnl_mlp,
        #         device_ids=cfg.secondary_gpus,
        #         output_device=cfg.primary_gpus[0])

        # pose decoder MLP
        self.pose_decoder = \
            load_pose_decoder(cfg.pose_decoder.module)(
                embedding_size=cfg.pose_decoder.embedding_size,
                mlp_width=cfg.pose_decoder.mlp_width,
                mlp_depth=cfg.pose_decoder.mlp_depth)
    

    def deploy_mlps_to_secondary_gpus(self):
        self.cnl_mlp = self.cnl_mlp.to(cfg.secondary_gpus[0])
        if self.non_rigid_mlp:
            self.non_rigid_mlp = self.non_rigid_mlp.to(cfg.secondary_gpus[0])

        return self


    def _query_mlp(
            self,
            pos_xyz,
            pos_embed_fn, 
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)

        result = self._apply_mlp_kernals(
                        pos_flat=pos_flat,
                        pos_embed_fn=pos_embed_fn,
                        non_rigid_mlp_input=non_rigid_mlp_input,
                        non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                        chunk=chunk)

        output = {}

        raws_flat = result['raws']
        output['raws'] = torch.reshape(
                            raws_flat, 
                            list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def _apply_mlp_kernals(
            self, 
            pos_flat,
            pos_embed_fn,
            non_rigid_mlp_input,
            non_rigid_pos_embed_fn,
            chunk):
        raws = []

        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start

            xyz = pos_flat[start:end]

            if not cfg.ignore_non_rigid_motions:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)
                result = self.non_rigid_mlp(
                    pos_embed=non_rigid_embed_xyz,
                    pos_xyz=xyz,
                    condition_code=self._expand_input(non_rigid_mlp_input, total_elem)
                )
                xyz = result['xyz']

            xyz_embedded = pos_embed_fn(xyz)
            raws += [self.cnl_mlp(
                        pos_embed=xyz_embedded)]

        output = {}
        output['raws'] = torch.cat(raws, dim=0).to(cfg.primary_gpus[0])

        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], cfg.chunk):
            ret = self._render_rays(rays_flat[i:i+cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, bgcolor=None):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0]

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.

        return rgb_map, acc_map, weights, depth_map


    @staticmethod
    def _sample_motion_fields(
            pts,
            motion_scale_Rs, 
            motion_Ts, 
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 

        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            pos = (pos - cnl_bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0 
            weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], 
                                    grid=pos[None, None, None, :, :],           
                                    padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None] 
            weights_list.append(weights) 
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]

        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, 
                                                dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum

        x_skel = x_skel.reshape(orig_shape[:2]+[3])
        backwarp_motion_weights = \
            backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        results = {}
        
        if 'x_skel' in output_list: # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask
        
        return results


    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input=None,
            bgcolor=None,
            **_):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        
        mv_output = self._sample_motion_fields(
                            pts=pts,
                            motion_scale_Rs=motion_scale_Rs[0], 
                            motion_Ts=motion_Ts[0], 
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel', 'fg_likelihood_mask'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']

        query_result = self._query_mlp(
                                pos_xyz=cnl_pts,
                                non_rigid_mlp_input=non_rigid_mlp_input,
                                pos_embed_fn=pos_embed_fn,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn)
        raw = query_result['raws']
        
        rgb_map, acc_map, _, depth_map = \
            self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)

        return {'rgb' : rgb_map,  
                'alpha' : acc_map, 
                'depth': depth_map}


    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(
                                        dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts


    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)
    

    def forward(self,
                imgs, alphas, bgcolor,
                iter_val=1e7,
                **kwargs):
        H, W = imgs.shape[1: 3]
        assert H == W

        imgs = imgs.permute(0, 3, 1, 2)
        vibe_output = self.pose_detector(imgs)

        dst_Rs = vibe_output[0]['rmtx'][0][0]
        betas = vibe_output[0]['shape'][0].mean(axis=0)
        pred_cam = vibe_output[0]['cam'][0][0]
        global_orient = dst_Rs[0]
        body_pose = dst_Rs[1:]
        t_pose = torch.eye(body_pose.shape[1]).expand(body_pose.shape)
        path = osp.join("third_parties", "smpl", "models", "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
        
        smpl_output = SMPLLayer(path, batch_size=1).to('cpu')(
            betas=betas[None, :].to('cpu'), 
            body_pose=body_pose[None, :].to('cpu'),
            global_orient=global_orient[None, :].to('cpu')
        )

        joints = smpl_output['joints'][0][0:24]
        bbox = skeleton_to_bbox(joints)

        tpose_output = SMPLLayer(path, batch_size=1).to('cpu')(
            betas=betas[None, :].to('cpu'), 
            body_pose=t_pose[None, :].to('cpu')
        )

        dst_posevec = vibe_output[0]['pose'][0][0][1:, ].ravel()
        canonical_joints = tpose_output['joints'][0][0:24]
        canonical_bbox = skeleton_to_bbox(canonical_joints)
        dst_Ts = body_pose_to_body_Ts(canonical_joints)
        cnl_gtfms = get_canonical_global_tfms(canonical_joints)

        K, E = get_camera_parameters(pred_cam, H, global_orient)
        R = E[:3, :3]
        T = E[:3, 3]
        
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        ray_img = imgs[0].reshape(-1, 3) 
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        
        near, far, ray_mask = rays_intersect_3d_bbox(bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]

        near = near[:, None]
        far = far[:, None]

        motion_weights_priors = \
                approx_gaussian_bone_volumes(
                    canonical_joints,   
                    bbox[0],
                    bbox[1],
                    grid_size=self.volume_size)
        
        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        dst_posevec=dst_posevec[None, ...]
        cnl_gtfms=cnl_gtfms[None, ...]
        motion_weights_priors=motion_weights_priors[None, ...]

        # correct body pose
        if iter_val >= cfg.pose_decoder.get('kick_in_iter', 0):
            pose_out = self.pose_decoder(dst_posevec)
            refined_Rs = pose_out['Rs']
            refined_Ts = pose_out.get('Ts', None)
            
            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            dst_Rs_no_root = self._multiply_corrected_Rs(
                                        dst_Rs_no_root, 
                                        refined_Rs)
            dst_Rs = torch.cat(
                [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

            if refined_Ts is not None:
                dst_Ts = dst_Ts + refined_Ts

        non_rigid_pos_embed_fn, _ = \
            self.get_non_rigid_embedder(
                multires=cfg.non_rigid_motion_mlp.multires,                         
                is_identity=cfg.non_rigid_motion_mlp.i_embed,
                iter_val=iter_val,)

        if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter:
            # mask-out non_rigid_mlp_input 
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec

        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "non_rigid_mlp_input": non_rigid_mlp_input
        })

        motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs.to(cfg.primary_gpus[0]), 
                                            dst_Ts=dst_Ts.to(cfg.primary_gpus[0]), 
                                            cnl_gtfms=cnl_gtfms.to(cfg.primary_gpus[0]))
        motion_weights_vol = self.mweight_vol_decoder(
            motion_weights_priors=motion_weights_priors.to(cfg.primary_gpus[0]))
        motion_weights_vol=motion_weights_vol[0] # remove batch dimension

        canonical_bbox = canonical_bbox.to(cfg.primary_gpus[0])
        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'motion_weights_vol': motion_weights_vol,
            'cnl_bbox_min_xyz': canonical_bbox[0],
            'cnl_bbox_max_xyz': canonical_bbox[1],
            'cnl_bbox_scale_xyz': 2.0 / (canonical_bbox[1] - canonical_bbox[0])
        })

        rays_shape = rays_d.shape 

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos.to(cfg.primary_gpus[0]), **kwargs)
        for k in all_ret:
            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_shape)

        return all_ret
