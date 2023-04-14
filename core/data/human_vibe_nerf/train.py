import os
import pickle

import numpy as np
import cv2
import torch
import torch.utils.data

from core.utils.image_util import load_image, split_into_chunks
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.file_util import list_files, split_path
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox

from configs import cfg

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_path,
            seqlen,
            stride,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            ray_shoot_mode='image',
            skip=1,
            **_):

        print('[Dataset Path]', dataset_path)
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, 'images')
        self.seqlen = seqlen
        self.stride = stride
        
        framelist = self.load_train_frames()
        self.framelist = framelist[::skip]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]
        print(f' -- Total Frames: {self.get_total_frames()}')

        self.vid_indices = split_into_chunks(self.framelist, self.seqlen, self.stride)

        self.keyfilter = keyfilter
        self.bgcolor = bgcolor

        self.ray_shoot_mode = ray_shoot_mode

    def load_train_frames(self):
        img_paths = list_files(self.image_dir, exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    def load_image(self, frame_names, bg_color):
        list_imgs = []
        list_alpha_masks = []
        for frame_name in frame_names:
            imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
            orig_img = np.array(load_image(imagepath))

            maskpath = os.path.join(self.dataset_path, 
                                    'masks', 
                                    '{}.png'.format(frame_name))
            alpha_mask = np.array(load_image(maskpath))

            alpha_mask = alpha_mask / 255.
            img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
            if cfg.resize_img_scale != 1.:
                img = cv2.resize(img, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LANCZOS4)
                alpha_mask = cv2.resize(alpha_mask, None, 
                                        fx=cfg.resize_img_scale,
                                        fy=cfg.resize_img_scale,
                                        interpolation=cv2.INTER_LINEAR)
            list_imgs.append(img)
            list_alpha_masks.append(alpha_mask)

        imgs = np.stack(list_imgs, axis=0)
        alpha_masks = np.stack(list_alpha_masks, axis=0)
         
        return imgs, alpha_masks
    
    def get_total_frames(self):
        return len(self.framelist)
    
    def __len__(self):
        return len(self.vid_indices)
    
    def __getitem__(self, index):
        start_index, end_index = self.vid_indices[index]
        frame_name = self.framelist[start_index:end_index + 1]
        results = {
            'frame_name': frame_name
        }
        
        if self.bgcolor is None:
            bgcolor = (np.random.rand(3) * 255.).astype('float32')
        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')

        imgs, alphas = self.load_image(frame_name, bgcolor)
        imgs = (imgs / 255.).astype('float32')

        H, W = imgs.shape[1:3]

        results.update({
            'img_width': W,
            'img_height': H,
            'imgs': imgs,
            'alphas': alphas,
            'bgcolor': bgcolor
        })
        
        return results
    
