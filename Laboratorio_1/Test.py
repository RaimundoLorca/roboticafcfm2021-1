#LAB 1 - Robótica 2021-1

#Cargar librerias a usar
import cv2
import numpy as np
import matplotlib.pyplot as plot

#Cargar imagen
img = cv2.imread("Karasuno.jpg")

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img)
plot.axis("off")

#%% Segmentar por canal

R = img[:, :, 0] #Canal Rojo
G = img[:, :, 1] #Canal Verde
B = img[:, :, 2] #Canal Azul

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(G, cmap='gray')
plot.axis("off")

#Se convierte la matriz de la imagen a un vector
G_flat = G.flatten()

#Plotear histograma con Matplotlib
plot.figure()
plot.hist(G_flat, 100)

#Generar mascara 
mask2 = cv2.inRange(G, 210, 255)
mask2 = mask2/255    #Normaliza valores para que queden entre 0 y 1

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(mask2, cmap='gray')
plot.axis("off")

#Multiplicar máscara con imagen original
img_mask = np.zeros_like(img)

for c in range(img_mask.shape[2]):
  img_mask[:,:,c] = np.multiply(img[:,:,c], mask2)
  
 #Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_mask)
plot.axis("off")

#%% Detectar circulos

#Transformada hough circulos (cv2.HoughCircles)
circles = cv2.HoughCircles(edge_canny, cv2.HOUGH_GRADIENT, dp=1, minDist=9, param2=25)

# Para cada circulo detectado
for i in range(len(circles[0])):
    centro = (circles[0][i][0], circles[0][i][1])
    radio = int(circles[0][i][2])
    color = (0, 255, 0)
    thickness = 4
    img_lines2 = cv2.circle(img_lines2, centro, radio, color, thickness)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_lines2)
plot.axis('off')

