import os
import cv2

from functools import partial

import tensorflow as tf
import numpy as np

from lib.data import Data

class RhoData(Data):
    '''Extract and process rho-NLR data.'''
    
    def __init__(self, path, img_size, from_cam_pose=True):
        super().__init__(path, img_size, data_type='rho')
        
        self.compute_rays = partial(Data.compute_rays, self, 
                                    from_cam_pose=from_cam_pose,
                                    view_diag=tf.constant([1., 1., 1., 1.]),
                                    uniform_intrinsics=True)
        
    def load_light_rays(self):
        if not hasattr(self, 'rays'):
            raise Exception('Please call compute_rays method before loading irradiance vectors.')
        
        irradiance_vecs = self.get_mat('direction', 
                                       self.rgb_img_names, 
                                       name='brdf_export.yaml')
        
        self.rays.append(irradiance_vecs)
        self.irradiance_vecs = tf.concat(irradiance_vecs, axis=0)
        
        for num, vec in enumerate(self.irradiance_vecs):
            self.rays[-1][num] = tf.broadcast_to(vec, self.rays[0][num].shape)
        
        tf.print('Irradiance vectors loaded')