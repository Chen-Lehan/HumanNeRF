from math import cos, sin

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SMPL_JOINT_IDX = {
    'pelvis_root': 0,
    'left_hip': 1,
    'right_hip': 2,
    'belly_button': 3,
    'left_knee': 4,
    'right_knee': 5,
    'lower_chest': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'upper_chest': 9,
    'left_toe': 10,
    'right_toe': 11,
    'neck': 12,
    'left_clavicle': 13,
    'right_clavicle': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
    'left_thumb': 22,
    'right_thumb': 23
}

SMPL_PARENT = {
    1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 
    11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 
    21: 19, 22: 20, 23: 21}

VIBE_DATA_DIR = '../data'
SMPL_MEAN_PARAMS = os.path.join(VIBE_DATA_DIR, 'smpl_mean_params.npz')
TORSO_JOINTS_NAME = [
    'pelvis_root', 'belly_button', 'lower_chest', 'upper_chest', 'left_clavicle', 'right_clavicle'
]
TORSO_JOINTS = [
    SMPL_JOINT_IDX[joint_name] for joint_name in TORSO_JOINTS_NAME
]
BONE_STDS = np.array([0.03, 0.06, 0.03])
HEAD_STDS = np.array([0.06, 0.06, 0.06])
JOINT_STDS = np.array([0.02, 0.02, 0.02])


def _to_skew_matrix(v):
    r""" Compute the skew matrix given a 3D vectors.

    Args:
        - v: Array (3, )

    Returns:
        - Array (3, 3)

    """
    vx, vy, vz = v.ravel()
    return np.array([[0, -vz, vy],
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
    skew_matrices = np.zeros(shape=(batch_size, 3, 3), dtype=np.float32)

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
    
    v1 = v1 / np.clip(np.linalg.norm(v1, axis=-1, keepdims=True), 1e-5, None)
    v2 = v2 / np.clip(np.linalg.norm(v2, axis=-1, keepdims=True), 1e-5, None)
    
    normal_vec = np.cross(v1, v2, axis=-1)
    cos_v = np.zeros(shape=(batch_size, 1))
    for i in range(batch_size):
        cos_v[i] = v1[i].dot(v2[i])

    skew_mtxs = _to_skew_matrices(normal_vec)
    
    Rs = np.zeros(shape=(batch_size, 3, 3), dtype=np.float32)
    for i in range(batch_size):
        Rs[i] = np.eye(3) + skew_mtxs[i] + \
                    (skew_mtxs[i].dot(skew_mtxs[i])) * (1./(1. + cos_v[i]))
    
    return Rs


def _construct_G(R_mtx, T):
    r""" Build 4x4 [R|T] matrix from rotation matrix, and translation vector
    
    Args:
        - R_mtx: Array (3, 3)
        - T: Array (3,)

    Returns:
        - Array (4, 4)
    """

    G = np.array(
        [[R_mtx[0, 0], R_mtx[0, 1], R_mtx[0, 2], T[0]],
         [R_mtx[1, 0], R_mtx[1, 1], R_mtx[1, 2], T[1]],
         [R_mtx[2, 0], R_mtx[2, 1], R_mtx[2, 2], T[2]],
         [0.,          0.,          0.,          1.]],
        dtype='float32')

    return G
    

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
    SIGMA = R.dot(S).dot(S).dot(R.T)

    min_x, min_y, min_z = bbox_min_xyz
    max_x, max_y, max_z = bbox_max_xyz
    zgrid, ygrid, xgrid = np.meshgrid(
        np.linspace(min_z, max_z, grid_size),
        np.linspace(min_y, max_y, grid_size),
        np.linspace(min_x, max_x, grid_size),
        indexing='ij')
    grid = np.stack([xgrid - center[0], 
                     ygrid - center[1], 
                     zgrid - center[2]],
                    axis=-1)

    dist = np.einsum('abci, abci->abc', np.einsum('abci, ij->abcj', grid, SIGMA), grid)

    return np.exp(-1 * dist)


def _std_to_scale_mtx(stds):
    r""" Build scale matrix from standard deviations
    
    Args:
        - stds: Array(3,)

    Returns:
        - Array (3, 3)
    """

    scale_mtx = np.eye(3, dtype=np.float32)
    scale_mtx[0][0] = 1.0/stds[0]
    scale_mtx[1][1] = 1.0/stds[1]
    scale_mtx[2][2] = 1.0/stds[2]

    return scale_mtx


def _rvec_to_rmtx(rvec):
    r''' apply Rodriguez Formula on rotate vector (3,)

    Args:
        - rvec: Array (3,)

    Returns:
        - Array (3, 3)
    '''
    rvec = rvec.reshape(3, 1)

    norm = np.linalg.norm(rvec)
    theta = norm
    r = rvec / (norm + 1e-5)

    skew_mtx = _to_skew_matrix(r)

    return cos(theta)*np.eye(3) + \
           sin(theta)*skew_mtx + \
           (1-cos(theta))*r.dot(r.T)


def body_pose_to_body_RTs(jangles, tpose_joints):
    r""" Convert body pose to global rotation matrix R and translation T.
    
    Args:
        - jangles (joint angles): Array (Total_Joints x 3, )
        - tpose_joints:           Array (Total_Joints, 3)

    Returns:
        - Rs: Array (Total_Joints, 3, 3)
        - Ts: Array (Total_Joints, 3)
    """

    jangles = jangles.reshape(-1, 3)
    total_joints = jangles.shape[0]
    assert tpose_joints.shape[0] == total_joints

    Rs = np.zeros(shape=[total_joints, 3, 3], dtype='float32')
    Rs[0] = _rvec_to_rmtx(jangles[0,:])

    Ts = np.zeros(shape=[total_joints, 3], dtype='float32')
    Ts[0] = tpose_joints[0,:]

    for i in range(1, total_joints):
        Rs[i] = _rvec_to_rmtx(jangles[i,:])
        Ts[i] = tpose_joints[i,:] - tpose_joints[SMPL_PARENT[i], :]
    
    return Rs, Ts


def get_canonical_global_tfms(canonical_joints):
    r""" Convert canonical joints to 4x4 global transformation matrix.
    
    Args:
        - canonical_joints: Array (Total_Joints, 3)

    Returns:
        - Array (Total_Joints, 4, 4)
    """

    total_bones = canonical_joints.shape[0]

    gtfms = np.zeros(shape=(total_bones, 4, 4), dtype='float32')
    gtfms[0] = _construct_G(np.eye(3), canonical_joints[0,:])

    for i in range(1, total_bones):
        translate = canonical_joints[i,:] - canonical_joints[SMPL_PARENT[i],:]
        gtfms[i] = gtfms[SMPL_PARENT[i]].dot(
                            _construct_G(np.eye(3), translate))

    return gtfms


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
    tpose_joints = tpose_joints.astype(np.float32)

    calibrated_bone = np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, :]
    g_volumes = []
    for joint_idx in range(0, total_joints):
        gaussian_volume = np.zeros(shape=grid_shape, dtype='float32')

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

            R = _get_rotation_mtx(calibrated_bone, target_bone)[0].astype(np.float32)

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
                                np.eye(3, dtype='float32'))
            
        g_volumes.append(gaussian_volume)
    g_volumes = np.stack(g_volumes, axis=0)

    # concatenate background weights
    bg_volume = 1.0 - np.sum(g_volumes, axis=0, keepdims=True).clip(min=0.0, max=1.0)
    g_volumes = np.concatenate([g_volumes, bg_volume], axis=0)
    g_volumes = g_volumes / np.sum(g_volumes, axis=0, keepdims=True).clip(min=0.001)
    
    return g_volumes


def rot6d_to_rotmat(x):
    x = x.view(-1,3,2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3,3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def estimate_translation_np(S, joints_2d, joints_conf, focal_length=5000., img_size=224.):
    """
    This function is borrowed from https://github.com/nkolot/SPIN/utils/geometry.py

    Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length,focal_length])
    # optical center
    center = np.array([img_size/2., img_size/2.])

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans
