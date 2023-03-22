# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 13:38:09 2023

@author: Benoit
"""

import lib_gs as gs
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('FR_CTP_iwv.txt')
x = data[:,0:1]
y = data[:,1:2]
iwv = data[:,2:3]
  
cont = np.loadtxt('FR_contour.txt')
xc = cont[:,0:1]
yc = cont[:,1:2]
gs.plot_patch(x, y, iwv, xlabel = 'x [m]', ylabel = 'y [m]', zlabel = 'iwv [kg/m2]')
plt.plot(xc,yc,'k',lw=2)