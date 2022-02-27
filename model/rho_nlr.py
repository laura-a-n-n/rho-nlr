import tensorflow as tf
import numpy as np

from model.nlr import NeuralLumigraph
from lib.harmonics import SphericalHarmonicBasis
from lib.math import dot

class RhoNeuralLumigraph(NeuralLumigraph):
    def __init__(self, 
                 n_coeffs=9, 
                 n_channels=3, 
                 e_layers=5, 
                 **kwargs):
        super().__init__(e_final_units=n_channels*n_coeffs, 
                         e_layers=e_layers,
                         **kwargs)
        self.sph_harm = SphericalHarmonicBasis(n_coeffs=n_coeffs)
        self.n_channels = n_channels
        
        self.current_l_d = None
        self.light_dir_transform = tf.identity
    
    def set_light_dirs(self, l_d):
        self.current_l_d = l_d
    
    def get_light_dirs(self):
        return self.current_l_d
    
    def cart_to_sphere(self, vec):
        x, y, z = tf.unstack(vec, axis=-1)
        return tf.math.atan2(-x, -z), tf.math.acos(y)
    
    def get_nearest_light_dirs(self, l_ds, gt, k=2):
        dots = dot(l_ds, gt)
        dists = -tf.math.abs(dots - 1.)
        vals, idxs = tf.math.top_k(dists, k=k)
        return tf.gather(tf.squeeze(gt, axis=0), idxs)
    
    def unpack_and_trace(self, batch, 
                         *args,
                         **kwargs):
        output_dict = super().unpack_and_trace(batch, 
                                               *args, 
                                               **kwargs)
        
        light_dirs = self.light_dir_transform(batch[:, 9:12])
        if kwargs['training']:
            light_dirs = tf.boolean_mask(light_dirs, output_dict['trace']['masks'])
        light_dirs = tf.boolean_mask(light_dirs, output_dict['trace']['conv_mask'])
        
        self.set_light_dirs(light_dirs)
        
        return output_dict
    
    def decode_rgb(self, coeffs):
        k_1, k_2, k_3 = tf.split(coeffs, self.n_channels, axis=-1)
        k = tf.stack([k_1, k_2, k_3], axis=-2)
        
        azimuthal, polar = self.cart_to_sphere(self.current_l_d)
        return self.sph_harm.sph_harm_reconstruct(k)(polar, azimuthal)