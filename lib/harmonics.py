import tensorflow as tf
import numpy as np

import scipy.integrate as integrate
from scipy.special import sph_harm

from lib.math import dot

class SphericalHarmonicBasis():
    def __init__(self, n_coeffs=25):
        self._n_coeffs = n_coeffs
        self.basis = self.sph_harm_basis()
    
    def get_sph_harm_function(self, l, m):
        '''Compute real spherical harmonic basis function.'''
        def basis_function(theta, phi):
            Y = sph_harm(abs(m), l, phi, theta)
            
            if m < 0:
                Y = np.sqrt(2) * Y.imag
            elif m > 0:
                Y = np.sqrt(2) * Y.real
                
            return Y.real
        
        return basis_function

    def sph_harm_basis(self):
        '''Get a specified number of basis functions.'''
        basis_functions = []

        dimension = 0
        l, m = 0, 0

        while dimension < self._n_coeffs:
            while m <= l:
                basis_functions.append(self.get_sph_harm_function(l, m))
                m += 1
                dimension += 1
            
            l += 1
            m = -l
        
        return basis_functions

    def sph_harm_coeff(self, Y, f):
        '''Compute spherical harmonic coefficients.'''
        def integrand(phi, theta):
            return f(theta, phi) * Y(theta, phi) * np.sin(theta)
        
        return integrate.dblquad(integrand, 0., np.pi, lambda x : 0., lambda x : 2*np.pi)[0]

    def sph_harm_transform(self, f, basis=None):
        '''Get spherical harmonic coefficients for a function in a basis.'''
        if basis is None:
            basis = self.basis
        
        coeffs = []

        for Y in basis:
            coeffs.append(self.sph_harm_coeff(Y, f))

        return coeffs

    def sph_harm_reconstruct_np(self, coeffs, basis=None):
        '''Reconstruct a function from basis and corresponding coefficients).'''
        if basis is None:
            basis = self.basis
        
        return lambda theta, phi : np.dot(coeffs, [f(theta, phi) for f in basis])
    
    def sph_harm_reconstruct(self, coeffs, basis=None):
        '''Reconstruct a "tensor of functions" from basis and corresponding coefficients.'''
        if basis is None:
            basis = self.basis
        
        def get_image(theta, phi):
            basis_image = tf.transpose(tf.constant([f(theta, phi) for f in basis], dtype=tf.float32))[:, tf.newaxis, :]
            
            return dot(coeffs, basis_image)
        
        return get_image