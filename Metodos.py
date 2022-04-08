#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:55:44 2020

@author: jaimecalderonocampo
"""
import numpy as np


def pasoalto_ideal(f, c, fc):
    # filas, columnas, frecuencia de corte
    filtro = np.zeros([f, c])
    for ff in range(f):
        for cc in range(c):
            circulo = np.sqrt((f/2 - ff)**2 + (c/2 - cc)**2)
            if (circulo > fc):
                filtro[ff, cc] = 1
    return filtro


def pasobajo_ideal(f, c, fc):
    # filas, columnas, frecuencia de corte
    filtro = np.zeros([f, c])
    for ff in range(f):
        for cc in range(c):
            circulo = np.sqrt((f/2 - ff)**2 + (c/2 - cc)**2)
            if (circulo > fc):
                filtro[ff, cc] = 1
    return filtro


def pasobajo_gauss(f, c, fc):
    # filas, columnas, frecuencia de corte
    filtro = np.zeros((f, c))
    for i in range(f):
        for j in range(c):
            circulo = np.sqrt((f/2 - i)**2 + (c/2 - j)**2)
            filtro[i, j] = np.exp((- circulo**2) / (2*fc**2))

    return filtro


def pasoalto_gauss(f, c, fc):
    # filas, columnas, frecuencia de corte
    filtro = np.zeros((f, c))
    for i in range(f):
        for j in range(c):
            circulo = np.sqrt((f/2 - i)**2 + (c/2 - j)**2)
            filtro[i, j] = 1 - np.exp((- circulo**2) / (2*fc**2))

    return filtro


def butterworth_pasobajo(f, c, fc, n):
    # filas, columnas, frecuencia de corte, orden de filtro
    filtro = np.zeros((f, c))
    for i in range(f):
        for j in range(c):
            circulo = np.sqrt((f/2 - i)**2 + (c/2 - j)**2)
            filtro[i, j] = 1 / (1 + (circulo / fc)**(2*n))
    return filtro


def butterworth_pasoalto(f, c, fc):
    # filas, columnas, frecuencia de corte, orden del filtro
    filtro = np.zeros((f, c))
    for i in range(f):
        for j in range(c):
            circulo = np.sqrt((f/2 - i)**2 + (c/2 - j)**2)
            if circulo == 0:
                circulo = 0.00000001
            filtro[i, j] = 1 / (1 + (fc / circulo)**(2*2))
    return filtro


def Laplacian_filter(f, c):
    # filas, columnas
    filtro = np.zeros((f, c))
    for i in range(f):
        for j in range(c):
            circulo = ((f/2 - i)**2 + (c/2 - j)**2)
            filtro[i, j] = - 4 * (np.pi)**2 * (circulo)

    filtro = (1 - filtro)
    return filtro


def Unsharp(f, c, tipo="Filtro", k=1, fc=10, n=2):
    global nombre
    if k == 1:
        T = "Unsharp"
    else:
        T = "High Boost"

    if tipo == "Gauss":
        Filtro = pasoalto_gauss(f, c, fc)
    elif tipo == "butterworth":
        Filtro = butterworth_pasoalto(f, c, fc)
    elif tipo == "ideal":
        Filtro = pasoalto_ideal(f, c, fc)

    filtro = 1 + k * Filtro
    nombre = T
    return filtro


def homomorphic_filtering(filas, columnas, f_corte, y_l, y_h, orden):
    filtro = np.zeros((filas, columnas))
    for i in range(filas):
        for j in range(columnas):
            circulo = ((filas/2) - i)**2 + ((columnas/2) - j)**2
            filtro[i, j] = (y_h - y_l) * \
                (1 - np.exp(-orden * circulo / (f_corte**2))) + y_l
    return filtro
