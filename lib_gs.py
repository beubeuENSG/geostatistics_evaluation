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

from scipy.spatial import Delaunay as delaunay
    
from matplotlib import cm

################## Modèle de fonction d'interpolation ##################

####################### Fonctions d'interpolation ######################

def interp_lin(x_obs, y_obs, z_obs, x_int, y_int):
    # Interpolation par ???
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]
    
    # Calcul de la triangulation à partir de coordonnées (x_obs, y_obs)
    tri = delaunay( np.hstack( (x_obs, y_obs) ) )
    
    z_int = np.nan*np.zeros(x_int.shape)
    for i in np.arange(0,x_int.shape[0]):
        for j in np.arange(0,x_int.shape[1]):
            # Recherche de l'index du triangle contenant le point x_int, y_int
            idx_t = tri.find_simplex( np.array([x_int[i,j], y_int[i,j]]) )
            if idx_t == -1: continue
            # Recherche des index des sommets du triangle contenant le point x_int, y_int
            idx_s = tri.simplices[idx_t,:]
            
            # Coordonnées des sommets du triangle contenant le point x0, y0
            x1 = x_obs[ idx_s[0] ] ; y1 = y_obs[ idx_s[0] ] ; z1 = z_obs[ idx_s[0] ]
            x2 = x_obs[ idx_s[1] ] ; y2 = y_obs[ idx_s[1] ] ; z2 = z_obs[ idx_s[1] ]
            x3 = x_obs[ idx_s[2] ] ; y3 = y_obs[ idx_s[2] ] ; z3 = z_obs[ idx_s[2] ]
            
            # z_int = np.nan*np.zeros(x_int.shape)
            M = np.array([[x1[0], x2[0], x3[0]], [y1[0], y2[0], y3[0]], [1, 1, 1]]).T
            Z_matrix = np.array([[z1[0], z2[0], z3[0]]]).T
            X = np.linalg.solve(M, Z_matrix)
            A = np.array([[x_int[i,j], y_int[i,j], 1]]).T
            z_int[i,j] = X.T @ A
            
    return z_int
    
def interp_inv(x_obs, y_obs, z_obs, x_int, y_int, p):
    # Interpolation par inverse des distances
    # x_obs, y_obs, z_obs : observations
    # [np.array dimension 1*n]
    # x_int, y_int, positions pour lesquelles on souhaite interpoler une valeur z_int
    # [np array dimension m*p]
    
    z_int = np.nan*np.zeros(x_int.shape)
    for i in np.arange(0,x_int.shape[0]):
        for j in np.arange(0,x_int.shape[1]):
            d = np.sqrt((x_int[i,j]-x_obs)**2+(y_int[i,j]-y_obs)**2)
            new_d = d**p
            w = 1/new_d # inverse des distances
            w_sum = np.sum(w)
            z_int[i,j] = np.sum(w*z_obs)/w_sum
    return z_int


def interp_spl(x_obs, y_obs, z_obs, x_grd, y_grd, rho):
    n = len(x_obs)
    
    # Construction des matrices A et B pour résoudre le système d'équations linéaires Aw = B
    A = np.zeros((n + 3, n + 3))
    B = np.zeros((n + 3, 1))
    
    for i in range(n):
        for j in range(n):
            dist = np.sqrt((x_obs[i] - x_obs[j]) ** 2 + (y_obs[i] - y_obs[j]) ** 2)
            if dist != 0:
                A[i, j] = (dist ** 2) * np.log(dist)
            else:
                A[i, j] = 0

            if i == j:
                A[i, j] += rho

        A[i, n] = 1
        A[i, n + 1] = x_obs[i]
        A[i, n + 2] = y_obs[i]
        A[n, i] = x_obs[i]
        A[n + 1, i] = y_obs[i]
        A[n + 2, i] = 1
    
        B[i] = z_obs[i]
    
    # Estimation des coefficients du spline en résolvant le système Aw = B
    w = np.linalg.solve(A, B)
    
    # Parcours de la grille et interpolation des valeurs
    z_grd_int = np.zeros_like(x_grd)
    for i in range(x_grd.shape[0]):
        for j in range(x_grd.shape[1]):
            x = x_grd[i, j]
            y = y_grd[i, j]
            
            z_interp = w[n] + w[n + 1] * x + w[n + 2] * y
            for k in range(n):
                dist = np.sqrt((x - x_obs[k]) ** 2 + (y - y_obs[k]) ** 2)
                z_interp += w[k] * ((dist ** 2) * np.log(dist) if dist != 0 else 0)
            
            z_grd_int[i, j] = z_interp

    return z_grd_int


def calculate_variogram_cloud(x_obs, y_obs, z_obs):
    n = len(x_obs)
    variogram_cloud = []
    
    for i in range(n):
        for j in range(i + 1, n):
            # Calcul de la distance entre les points d'observation i et j
            distance = np.sqrt((x_obs[i] - x_obs[j])**2 + (y_obs[i] - y_obs[j])**2)
            
            # Calcul de la différence des valeurs z entre les points d'observation i et j
            z_difference = np.abs(z_obs[i] - z_obs[j])
            z_difference_new = (1/2)*((z_difference)**2)
            
            # Ajout de la paire (distance, z_difference) à la nuée variographique
            variogram_cloud.append((distance, z_difference_new))
    
    return variogram_cloud


def plot_variogram_cloud(variogram_cloud):
    # Extraction des distances et différences de z à partir de la nuée variographique
    distances = [pair[0] for pair in variogram_cloud]
    z_differences = [pair[1] for pair in variogram_cloud]

    # Création du graphique
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, z_differences, marker='o', edgecolors='k', facecolors='none', alpha=0.6)
    plt.xlabel('Distance')
    plt.ylabel('1/2 Différence de z au carré')
    plt.title('Nuée variographique')
    plt.grid(True)
    plt.show()
    
    
def calculate_experimental_variogram(variogram_cloud, bin_width):
    max_distance = np.max([pair[0] for pair in variogram_cloud])
    n_bins = int(np.ceil(max_distance / bin_width))

    experimental_variogram = []

    for i in range(n_bins):
        bin_min = i * bin_width
        bin_max = (i + 1) * bin_width

        # Recherche des indices des points de la nuée variographique dans chaque classe de distance
        indices = np.where((np.array([pair[0] for pair in variogram_cloud]) >= bin_min) & (np.array([pair[0] for pair in variogram_cloud]) < bin_max))

        # Récupération des valeurs associées à ces indices
        values = np.array([pair[1] for pair in variogram_cloud])[indices]

        # Calcul de la moyenne des valeurs de chaque classe de distance
        if len(values) > 0:
            mean_value = np.mean(values)
        else:
            mean_value = np.nan

        experimental_variogram.append((bin_min + bin_width / 2, mean_value))

    return experimental_variogram



def plot_experimental_variogram(experimental_variogram):
    distances = [pair[0] for pair in experimental_variogram]
    mean_z_differences = [pair[1] for pair in experimental_variogram]

    plt.figure(figsize=(10, 6))
    plt.plot(distances, mean_z_differences, marker='o', linestyle='-')
    plt.xlabel('Distance')
    plt.ylabel('Demi-variance moyenne')
    plt.title('Variogramme expérimental')
    plt.grid(True)
    plt.show()
    

def plot_variogram_cloud_and_experimental_variogram(variogram_cloud, experimental_variogram):
    # Extraction des distances et différences de z à partir de la nuée variographique
    cloud_distances = [pair[0] for pair in variogram_cloud]
    cloud_z_differences = [pair[1] for pair in variogram_cloud]

    # Extraction des distances et demi-variances moyennes à partir du variogramme expérimental
    exp_distances = [pair[0] for pair in experimental_variogram]
    mean_z_differences = [pair[1] for pair in experimental_variogram]

    # Création du graphique
    plt.figure(figsize=(10, 6))

    # Tracer la nuée variographique
    plt.scatter(cloud_distances, cloud_z_differences, marker='o', edgecolors='k', facecolors='none', alpha=0.6, label='Nuée variographique')

    # Tracer le variogramme expérimental
    plt.plot(exp_distances, mean_z_differences, marker='o', linestyle='-', color='r', label='Variogramme expérimental')

    # Configuration du graphique
    plt.xlabel('Distance')
    plt.ylabel('Demi-variance')
    plt.title('Nuée variographique et Variogramme expérimental')
    plt.legend()
    plt.grid(True)
    plt.show()


def cubic_model(h, nugget, sill, range_):
    return np.piecewise(h, [h <= range_, h > range_],
                        [lambda h: nugget + (sill - nugget) * ((3 * h) / (2 * range_) - (1 / 2) * (h / range_) ** 3),
                         lambda h: sill])


def error_function(params, h_vals, val_vals):
    nugget, sill, range_ = params
    model_vals = cubic_model(h_vals, nugget, sill, range_)
    return np.sum((val_vals - model_vals) ** 2)


def gradient_descent(h_vals, val_vals, initial_guess, learning_rate, max_iter, tol):
    params = np.array(initial_guess)
    for _ in range(max_iter):
        gradient = np.zeros(3)
        for h, val in zip(h_vals, val_vals):
            for i in range(3):
                d_params = np.zeros(3)
                d_params[i] = 1e-8
                gradient[i] += 2 * (cubic_model(h, *params) - val) * (
                        (cubic_model(h, *(params + d_params)) - cubic_model(h, *params)) / d_params[i])

        params = params - learning_rate * gradient

        if np.linalg.norm(gradient) < tol:
            break

    return params


def calculate_analytic_variogram(exp_variogram, initial_guess=(0, 1, 100), learning_rate=1e-6, max_iter=10000, tol=1e-6):
    exp_variogram_clean = [(h, val) for h, val in exp_variogram if not np.isnan(val)]
    h_vals = np.array([pair[0] for pair in exp_variogram_clean])
    val_vals = np.array([pair[1] for pair in exp_variogram_clean])

    optimized_params = gradient_descent(h_vals, val_vals, initial_guess, learning_rate, max_iter, tol)

    analytic_variogram = [(h, cubic_model(h, *optimized_params)) for h, _ in exp_variogram]
    
    return analytic_variogram, optimized_params


def plot_combined_variograms(exp_variogram, analytic_variogram):
    exp_distances = [pair[0] for pair in exp_variogram]
    exp_mean_z_differences = [pair[1] for pair in exp_variogram]

    analytic_distances = [pair[0] for pair in analytic_variogram]
    analytic_mean_z_differences = [pair[1] for pair in analytic_variogram]

    plt.figure(figsize=(10, 6))
    plt.plot(exp_distances, exp_mean_z_differences, marker='o', linestyle='-', label='Variogramme expérimental')
    plt.plot(analytic_distances, analytic_mean_z_differences, marker='o', linestyle='-', color='r', label='Variogramme analytique')
    plt.xlabel('Distance')
    plt.ylabel('Demi-variance')
    plt.title('Variogrammes expérimental et analytique')
    plt.legend()
    plt.grid(True)
    plt.show()


############################# Visualisation ############################

def plot_contour_2d(x_grd ,y_grd ,z_grd, x_obs = np.array([]) ,y_obs = np.array([]), xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé du champ interpolé sous forme d'isolignes
    # x_grd, y_grd, z_grd : grille de valeurs interpolées
    # x_obs, y_obs : observations (facultatif)
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    
    z_grd_m = np.ma.masked_invalid(z_grd)
    fig = plt.figure()
    plt.contour(x_grd, y_grd, z_grd_m, int(np.round((np.max(z_grd_m)-np.min(z_grd_m))/4)),colors ='k')
    if x_obs.shape[0]>0:
        plt.scatter(x_obs, y_obs, marker = 'o', c = 'k', s = 5)
        plt.xlim(0.95*np.min(x_obs),np.max(x_obs)+0.05*np.min(x_obs))
        plt.ylim(0.95*np.min(y_obs),np.max(y_obs)+0.05*np.min(y_obs))
    else:
        plt.xlim(0.95*np.min(x_grd),np.max(x_grd)+0.05*np.min(x_grd))
        plt.ylim(0.95*np.min(y_grd),np.max(y_grd)+0.05*np.min(y_grd))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')

def plot_surface_2d(x_grd ,y_grd ,z_grd, x_obs = np.array([]) ,y_obs = np.array([]), minmax = [0,0], xlabel = "", ylabel = "", zlabel = "", title = "", fileo = "", cmap = cm.terrain):
    # Tracé du champ interpolé sous forme d'une surface colorée
    # x_grd, y_grd, z_grd : grille de valeurs interpolées
    # x_obs, y_obs : observations (facultatif)
    # minmax : valeurs min et max de la variable interpolée (facultatif)
    # xlabel, ylabel, zlabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    # cmap : nom de la carte de couleur
    z_grd_m = np.ma.masked_invalid(z_grd)
    fig = plt.figure()
    if minmax[0] < minmax[-1]:
        p=plt.pcolormesh(x_grd, y_grd, z_grd_m, cmap=cmap, vmin = minmax[0], vmax = minmax[-1], shading = 'auto')
    else:
        p=plt.pcolormesh(x_grd, y_grd, z_grd_m, cmap=cmap, shading = 'auto')
    if x_obs.shape[0]>0:
        plt.scatter(x_obs, y_obs, marker = 'o', c = 'k', s = 5)
        plt.xlim(0.95*np.min(x_obs),np.max(x_obs)+0.05*np.min(x_obs))
        plt.ylim(0.95*np.min(y_obs),np.max(y_obs)+0.05*np.min(y_obs))
    else:
        plt.xlim(0.95*np.min(x_grd),np.max(x_grd)+0.05*np.min(x_grd))
        plt.ylim(0.95*np.min(y_grd),np.max(y_grd)+0.05*np.min(y_grd))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    fig.colorbar(p,ax=plt.gca(),label=zlabel,fraction=0.046, pad=0.04)
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')

def plot_points(x_obs, y_obs, xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé des sites d'observations
    # x_obs, y_obs : observations
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    
    fig = plt.figure()
    ax = plt.gca()
    plt.plot(x_obs, y_obs, 'ok', ms = 4)
    ax.set_xlim(0.95*min(x_obs),max(x_obs)+0.05*min(x_obs))
    ax.set_ylim(0.95*min(y_obs),max(y_obs)+0.05*min(y_obs))
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')

def plot_patch(x_obs, y_obs, z_obs, xlabel = "", ylabel = "", zlabel = "", title = "", fileo = "", cmap = cm.terrain):
    # Tracé des valeurs observées
    # x_obs, y_obs, z_obs : observations
    # xlabel, ylabel, zlabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    # cmap : nom de la carte de couleur
    
    fig = plt.figure()
    p=plt.scatter(x_obs, y_obs, marker = 'o', c = z_obs, s = 80, cmap=cmap)
    plt.xlim(0.95*min(x_obs),max(x_obs)+0.05*min(x_obs))
    plt.ylim(0.95*min(y_obs),max(y_obs)+0.05*min(y_obs))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    fig.colorbar(p,ax=plt.gca(),label=zlabel,fraction=0.046, pad=0.04)
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')

def plot_triangulation(x_obs, y_obs, xlabel = "", ylabel = "", title = "", fileo = ""):
    # Tracé de la triangulation sur des sites d'observations
    # x_obs, y_obs : observations
    # xlabel, ylabel : étiquettes des axes (facultatif)
    # title : titre (facultatif)
    # fileo : nom du fichier d'enregistrement de la figure (facultatif)
    from scipy.spatial import Delaunay as delaunay
    tri = delaunay(np.hstack((x_obs,y_obs)))
    
    plt.figure()
    plt.triplot(x_obs[:,0], y_obs[:,0], tri.simplices)
    plt.plot(x_obs, y_obs, 'or', ms=4)
    plt.xlim(0.95*min(x_obs),max(x_obs)+0.05*min(x_obs))
    plt.ylim(0.95*min(y_obs),max(y_obs)+0.05*min(y_obs))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if not fileo == "": plt.savefig(fileo,bbox_inches='tight')
