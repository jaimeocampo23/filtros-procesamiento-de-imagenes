#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:56:01 2020

@author: jaimecalderonocampo
"""

import numpy as np
import imageio
import matplotlib.pyplot as plt
from skimage import io, color, data, morphology, filters
from mpl_toolkits import mplot3d
import Metodos as M

Mx = data.camera()
[f, c] = Mx.shape

fcorte = 30
Fourier = np.fft.fftshift(np.fft.fft2(Mx))
# LAPLACIAN FILTER
filtro = M.Laplacian_filter(f, c)
Mx2 = np.fft.ifft2(np.fft.ifftshift(filtro * Fourier))
# UNSHARP FILTER
K = 1
# filtro = M.Unsharp(f,c,Tipo= "Gauss", K, fcorte)
# filtro = M.Unsharp(f=f, c=c, fc=fcorte, k=K, tipo="Gauss")
# Mx2 = np.fft.ifft2(np.fft.ifftshift((filtro * Fourier)))
# HOMOMORPHIC FILTERING
#A = np.log(1 + Mx)
#filtro = M.pasoalto_gauss(f,c,fcorte)
#Fourier = np.fft.fftshift(np.fft.fft2(A))
#Mx2 = np.fft.ifft2(np.fft.ifftshift(filtro * Fourier))
#Gxy = 1 - np.exp(Mx2)
#Mx2 = Gxy


plt.figure(1)
plt.imshow(Mx, cmap='gray')
plt.figure(2)
plt.imshow(np.log(np.abs(Fourier)), cmap='gray')
plt.figure(3)
plt.imshow(np.abs(Mx2), cmap='gray')


xx = np.linspace(0, f, f)
yy = np.linspace(0, c, c)
X, Y = np.meshgrid(xx, yy)
Z = M.Laplacian_filter(f, c)

fig1 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z)
# ax.plot_surface(X,Y,Z2)
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_trisurf(xx, yy, Z, linewidth=0.2, antialiased=True)
