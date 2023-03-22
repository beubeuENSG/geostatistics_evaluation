#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################
#    TP - Introduction à l'interpolation spatiale et aux géostatistiques #
##########################################################################

# P. Bosser / ENSTA Bretagne
# Version du 13/03/2022


# Numpy
import numpy as np
# Matplotlib / plot
import matplotlib.pyplot as plt
# lib geostatistic
import lib_gs as gs

# Chargement des données
data = np.loadtxt('points.dat')
# On force le format des données (matrice d'une colonne)
x_obs = data[:,0:1]
y_obs = data[:,1:2]
z_obs = data[:,2:3]

# Visualisation des données en entrée : sites de mesure
gs.plot_points(x_obs, y_obs, xlabel = 'x [m]', ylabel = 'y [m]', title = "sites d'observation")
# Visualisation des données en entrée : sites de mesure coloré en fonction de la VR
gs.plot_patch(x_obs, y_obs, z_obs, xlabel = 'x [m]', ylabel = 'y [m]', zlabel = 'z [m]', title = "observations")

# Création d'une grille planimétrique pour l'interpolation
x_grd, y_grd = np.meshgrid(np.linspace(np.floor(np.min(x_obs)), np.ceil(np.max(x_obs)), 100), np.linspace(np.floor(np.min(y_obs)), np.ceil(np.max(y_obs)), 100))

# Interpolation par les différentes méthodes
#z__grd_int = gs.interp_inv(x_obs, y_obs, z_obs, x_grd, y_grd, 4)
#z__grd_int = gs.interp_lin(x_obs, y_obs, z_obs, x_grd, y_grd)
z__grd_int = gs.interp_spl(x_obs, y_obs, z_obs, x_grd, y_grd, 1000)

# Mise en place des différentes nuées et variogrammes
# variogram_cloud = gs.calculate_variogram_cloud(x_obs, y_obs, z_obs)
# gs.plot_variogram_cloud(variogram_cloud)
# bin_width = 10
# experimental_variogram = gs.calculate_experimental_variogram(variogram_cloud, bin_width)
# gs.plot_experimental_variogram(experimental_variogram)
# gs.plot_variogram_cloud_and_experimental_variogram(variogram_cloud, experimental_variogram)
# # Utilisation de la fonction 'calculate_analytic_variogram'
# analytic_variogram, optimized_params = gs.calculate_analytic_variogram(experimental_variogram)
# # Affichage des variogrammes expérimental et analytique
# gs.plot_combined_variograms(experimental_variogram, analytic_variogram)

# Visualiation de l'interpolation PPV : lignes de niveau
gs.plot_contour_2d(x_grd, y_grd, z__grd_int, x_obs, y_obs, xlabel = 'x [m]', ylabel = 'y [m]', title = 'interpolation contours')

# Visualiation de l'interpolation PPV : surface colorée
gs.plot_surface_2d(x_grd, y_grd, z__grd_int, x_obs, y_obs, xlabel = 'x [m]', ylabel = 'y [m]', title = 'interpolation surface')

# Interpolation en un point de l'espace
zi = gs.interp_ppv(x_obs, y_obs, z_obs, np.array([[225]]),  np.array([[180]]), 4)
print("La valeur interpolée en (225,180) est "+str(zi))
plt.show()
