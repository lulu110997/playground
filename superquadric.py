#!/usr/bin/python3

import math

import matplotlib.pyplot as plt
import numpy as np
import pyquaternion.quaternion
from pyquaternion import Quaternion

# TODO: 3D translation

class SuperquadricObject:
    """
    Class for representing a superquadric
    """

    def __init__(self, a=0.1, b=0.1, c=0.1, eps1=1.0, eps2=1.0, pos=None, quat=None):
        """
        Create a superquadric shape based on the inside-outside equation

                                     /  /  xB_1  \ (2/eps2)    /  xB_2  \ (2/eps2) \ (eps2/eps1)     /  xB_3  \ (2/eps1)
            F(a,b,c,eps1,eps2,xB) =  |  |--------|          +  |--------|          |             +   |--------|
                                     \  \    a   /             \    b   /          /                 \    c   /

        Args:
            a, b, c: float | ellipsoid radius of the object corresponding to x-axis, y-axis and z-axis (defaults to 0.1)
            eps1, eps2: float | shape parameters of the superquadric (defaults to 1)
            pos: tuple | position of the superquadric defined in the global frame (defaults to origin)
            quat: tuple | orientation of the superquadric defined in the global frame (defaults to 1,0,0,0 -> w,x,y,z)
        """

        self.a = a
        self.b = b
        self.c = c
        self.eps1 = eps1
        self.eps2 = eps2

        if pos is None:
            self.pos = np.array([(0, 0, 0)]).reshape(3, 1)
        else:
            self.pos = np.array([pos]).reshape(3, 1)

        if quat is None:
            self.quat = Quaternion((1, 0, 0, 0))
        else:
            if isinstance(quat, tuple):
                self.quat = Quaternion(quat)
            else:
                raise ValueError("Bad input to quat argument")

    def set_pose(self, pos, quat):
        """
        Setter for updating the object's pose and the point in question
        Args:
            pos: tuple | The position vector of the object's center with respect to the world frame
            quat: tuple | The orientation of the object with respect to the world frame (w,x,y,z)
        """
        self.pos = pos
        if isinstance(quat, tuple):
            self.quat = Quaternion(quat)
        else:
            raise ValueError("Bad input to quat argument")

    def get_eps(self):
        return self.eps1, self.eps2

    def get_abc(self):
        return self.a, self.b, self.c

    def get_pose(self):
        """
        Getter for the object's pose
        Returns: tuple(np.array, Quaternion) | (xyz position, wxyz quaternion)
        """
        return self.pos, self.quat.elements

    def plot_sq(self, ax, colour, plot_type='3D', alpha=0.2):
        """
        Plot superquadric. 2D plot uses the XZ dimensions rather than XY
        """
        X, Y, Z = self._get_mesh()

        if plot_type == '2D':
            # https://stackoverflow.com/questions/43482191/matplotlib-axes-plot-vs-pyplot-plot
            return ax.plot(X, Z, alpha=alpha, linewidth=.3, color=colour)
        elif plot_type == '3D':
            return ax.plot_surface(X, Y, Z, alpha=alpha, color=colour)
        elif plot_type == 'animate':
            return X, Y, Z
        else:
            raise 'Undefined plot type. Should be either animeate, 2D or 3D'

    def _get_mesh(self):
        """
        Get expansion dimensions of object based on given pose
        """
        pos_x = self.pos[0]
        pos_y = self.pos[1]
        pos_z = self.pos[2]

        scos = lambda theta, eps: math.copysign(abs(math.cos(theta)) ** eps, math.cos(theta))
        ssin = lambda theta, eps: math.copysign(abs(math.sin(theta)) ** eps, math.sin(theta))

        WN = 40
        NN = 80

        w_array = np.linspace(-math.pi, math.pi, WN)
        n_array = np.linspace(-math.pi / 2, math.pi / 2, NN)
        w_mesh, n_mesh = np.meshgrid(w_array, n_array)

        X = np.zeros(w_mesh.shape)
        Y = np.zeros(w_mesh.shape)
        Z = np.zeros(w_mesh.shape)

        for (i, j), __ in np.ndenumerate(w_mesh):
            w = w_mesh[i, j]
            n = n_mesh[i, j]

            X[i, j] = self.a * scos(n, self.eps1) * scos(w, self.eps2)
            Y[i, j] = self.b * scos(n, self.eps1) * ssin(w, self.eps2)
            Z[i, j] = self.c * ssin(n, self.eps1)

        T = np.zeros((4, 4))
        T[0:3, 0:3] = self.quat.rotation_matrix
        T[0:3, 3] = np.array([pos_x, pos_y, pos_z]).transpose()
        T[3, :] = np.array([[0, 0, 0, 1]])

        for (i, j), __ in np.ndenumerate(X):
            xp = np.array([[X[i, j], Y[i, j], Z[i, j], 1]]).transpose()
            xp = T @ xp
            X[i, j] = xp[0]
            Y[i, j] = xp[1]
            Z[i, j] = xp[2]

        return X, Y, Z