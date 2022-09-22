from math import exp
from email.mime import image
from lib2to3.pgen2.literals import simple_escapes
from operator import truediv
from pickle import FALSE, TRUE
from tkinter import E
from tokenize import Double
import cv2
import numpy as np

def clamp(num, min_value, max_value):
        num = max(min(num, max_value), min_value)
        return num

def MostrarImagenes(*imagenes):
    n=1
    for imagen in imagenes:
        cv2.imshow('Imagen'+str(n),imagen)
        n=n+1
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    return

def filtroAverage(imagen,mostrar=False):
    average =  np.median(imagen)
    imagen250 = np.ones_like(imagen)*255
    imagen0 = np.zeros_like(imagen)

    condlist = [imagen>average, average<=average]
    choicelist = [imagen250, imagen0]
    
    imagenRes = np.select(condlist, choicelist)

    if(mostrar):
        cv2.imshow('FiltroAverage',imagenRes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return imagenRes

def filtroLaplaceano(imagen,c):
    centro = 1-c
    adyacentes = c/4

    def fooLaplaceano(i,j):
        #return abs(imagen[i,j] *centro) + abs(imagen[i-1,j] *adyacentes) +abs(imagen[i,j-1] *adyacentes) +abs(imagen[i+1,j] *adyacentes) + abs(imagen[i,j+1] *adyacentes)
        return (imagen[i,j] *centro) + (imagen[i-1,j] *adyacentes) +(imagen[i,j-1] *adyacentes) +(imagen[i+1,j] *adyacentes) + (imagen[i,j+1] *adyacentes)
    
    imagenaux=np.array(imagen)
    dim = imagen.shape
    for i in range(1,dim[0]-1):
        for j in range(1,dim[1]-1):
            imagenaux[i,j] = clamp(fooLaplaceano(i,j),0,255)
    return imagenaux

def filtroGaussiano(imagen,sigma):
    centro = int(3 *sigma)
    print(centro)
    sigma2 = -2*sigma*sigma
    anchura=centro*2 + 1
    
    suma=0.0
    h=[0]*(centro+1)
    for i in range(0,centro):
        radio = centro-i
        h[i] = float(exp( (radio*radio)/sigma2))
        suma=suma + h[i]
    def fooGauss1DinX(i,j,imagen):
        aux=imagen[i,j]
        for d in range(1,centro+1):
            aux= aux + (imagen[i+d,j]*h[centro-d] +imagen[i-d,j]*h[centro-d])
        return aux

    def fooGauss1DinY(i,j,imagen):
        aux=imagen[i,j]
        for d in range(1,centro+1):
            aux= aux + (imagen[i,j+d]*h[centro-d] +imagen[i,j-d]*h[centro-d])
        return aux

    imagenaux=np.array(imagen)
    imagenaux2=np.array(imagen)
    dim = imagen.shape
    for i in range(centro,dim[0]-centro):
        for j in range(centro,dim[1]-centro):
            imagenaux[i,j] = clamp(fooGauss1DinX(i,j,imagen),0,255)
    
    for i in range(centro,dim[0]-centro):
        for j in range(centro,dim[1]-centro):
            imagenaux2[i,j] = clamp(fooGauss1DinY(i,j,imagenaux),0,255)
    return imagenaux2

def filtroBlurOrSharpen(imagen,sigma,w):
    imgG=filtroGaussiano(imagen,sigma=sigma)
    imagenaux=np.array(imagen)
    w=clamp(w,-1,1)

    dim = imagen.shape
    for i in range(1,dim[0]-1):
        for j in range(1,dim[1]-1):
            imagenaux[i,j] = clamp((1+w)*imagen[i,j] - w*imgG[i,j],0,255)
    return imagenaux

def AbrirImagen(nombre, x = -1,y=-1,mostrar=False):
    path_file ="*/imagenes/"+nombre # Aqui pongan el path a las imagenes.
    

    imagen = cv2.imread(path_file,cv2.IMREAD_GRAYSCALE)
    
    if(x==-1):
        x=imagen.shape[0]
    if(y==-1):
        y=imagen.shape[1]

    imagen = cv2.resize(imagen,(x, y))

    if(mostrar):
        cv2.imshow(nombre,imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return np.array(imagen)

mat = AbrirImagen("j.png",x=500,y=300,mostrar=False)
MostrarImagenes(mat,filtroLaplaceano(mat,1,-1))