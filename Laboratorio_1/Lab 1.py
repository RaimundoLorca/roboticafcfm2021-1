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

#Plotear imagen con OpenCV
# cv2.imshow("img", img)
# cv2.waitKey(0)

#Cambiar espacio de colores (BGR A RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_rgb)
plot.axis("off")

#Cambiar espacio de colores (RGB A GRAY)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_gray, cmap='gray')
plot.axis("off")

#%% Canales de la imagen

R = img_rgb[:, :, 0] #Canal Rojo
G = img_rgb[:, :, 1] #Canal Verde
B = img_rgb[:, :, 2] #Canal Azul

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(R, cmap='gray')
plot.axis("off")

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(G, cmap='gray')
plot.axis("off")

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(B, cmap='gray')
plot.axis("off")

#Visualizando cada canal según su color (tambien se puede hacer con np.zeros_like(img))
RR = np.zeros(img_rgb.shape).astype(np.uint8)
GG = np.zeros(img_rgb.shape).astype(np.uint8)
BB = np.zeros(img_rgb.shape).astype(np.uint8)

RR[:, :, 0] = R
GG[:, :, 1] = G
BB[:, :, 2] = B

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(RR)
plot.axis("off")

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(GG)
plot.axis("off")

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(BB)
plot.axis("off")

#%% Crear mascaras

#Cargar imagen y pasarla a RGB
img = cv2.imread("Led_deco.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img)
plot.axis("off")

#Pasarla a escala de grises (GRAY)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_gray, cmap='gray')
plot.axis("off")

#Umbrales por foto
#Led_deco (90,255)
#Led_A (200,255)
#Led_beso (160,255)

#Crear matriz para la mascara
mask = np.zeros(img_gray.shape)  #Matriz vacia
a = mask.shape[0]           #Cantida de filas
b = mask.shape[1]           #Cantida de columnas

#Binarizar imagen de manera manual
for i in range(a):
    for j in range(b):
        if img_gray[i,j] < 90:
            mask[i,j] = 0
        else:
            mask[i,j] = 255

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(mask, cmap= 'gray')
plot.axis("off")

#%% Crear mascara mediante funcion cv2.inRange()
mask2 = cv2.inRange(img_gray, 90, 255)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(mask2, cmap= 'gray')
plot.axis("off")

mask2 = mask2/255    #Normaliza valores para que queden entre 0 y 1

#Multiplicar máscara con imagen original
img_mask = np.zeros_like(img)

for c in range(img_mask.shape[2]):
  img_mask[:,:,c] = np.multiply(img[:,:,c], mask2)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_mask)
plot.axis("off")

#%% Histogramas

#Se convierte la matriz de la imagen a un vector
img_flat = img_gray.flatten()

#Plotear histograma con Matplotlib
plot.figure()
plot.hist(img_flat, 100)
plot.title('Histograma imagen Led_deco en escala de grises')
plot.xlabel('Intensidad color')
plot.ylabel('Frecuencia')

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

#%% Eliminación de ruido

#Se aplica un filtro gaussiano (las dimensiones del kernel deben ser positivas
#e impares)
img_blur = cv2.GaussianBlur(img_gray, (11,11), 0)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_gray, cmap= 'gray')
plot.axis("off")

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_blur, cmap= 'gray')
plot.axis("off")

# Crear mascara mediante funcion cv2.inRange()
mask = cv2.inRange(img_blur, 90, 255)
mask = mask/255    #Normaliza valores para que queden entre 0 y 1

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(mask, cmap= 'gray')
plot.axis("off")

#Multiplicar máscara con imagen original
img_mask = np.zeros_like(img)

for c in range(img_mask.shape[2]):
  img_mask[:,:,c] = np.multiply(img[:,:,c], mask)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_mask)
plot.axis("off")

#%% Contraste

img = cv2.imread('Saturada.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img)
plot.axis("off")

img_flat = img_gray.flatten()

#Plotear histograma con Matplotlib
plot.figure()
plot.hist(img_flat, 100)

#Fijar el minimo y maximo de la imagen
min_img_gray = np.min(img_gray)
max_img_gray = np.max(img_gray)

#Cambiar tipo de datos de la matriz
img_gray = np.float32(img_gray)

#Aplicar contraste mediante una transformación lineal
for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        img_gray[i,j] = 255*((img_gray[i,j]-min_img_gray)/(max_img_gray-min_img_gray))
        
#Volver a uint8
img_gray = np.uint8(img_gray)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_gray, cmap='gray')
plot.axis("off")

img_flat2 = img_gray.flatten()

#Plotear histograma con Matplotlib
plot.figure()
plot.hist(img_flat2, 100)

#%% Contraste arreglado

img = cv2.imread('Saturada.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Fijar el minimo y maximo de la imagen
min_img_gray = 180
max_img_gray = np.max(img_gray)

#Cambiar tipo de datos de la matriz
img_gray = np.float32(img_gray)

#Aplicar contraste mediante una transformación lineal
for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):
        img_gray[i,j] = 255*((img_gray[i,j]-min_img_gray)/(max_img_gray-min_img_gray))
        if img_gray[i,j] < 0:
            img_gray[i,j] = 0
        
#Volver a uint8
img_gray = np.uint8(img_gray)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_gray, cmap='gray')
plot.axis("off")

img_flat2 = img_gray.flatten()

#Plotear histograma con Matplotlib
plot.figure()
plot.hist(img_flat2, 100)

#%% Connected Components

#Cargar imagen y pasarla a escala de grises (GRAY)
img = cv2.imread("Led_deco.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Se aplica un filtro gaussiano  (PROBAR CON Y SIN FILTRO)
img_blur = cv2.GaussianBlur(img_gray, (11,11), 0)

#Crear mascara mediante funcion cv2.inRange()
mask2 = cv2.inRange(img_blur, 90, 255)
mask2 = mask2/255    #Normaliza valores para que queden entre 0 y 1
mask2 = np.uint8(mask2)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(mask2, cmap='gray')
plot.axis("off")

#Etiquetar mediante connected components
num_object, labels = cv2.connectedComponents(mask2)

print('Objetos identificadas: {:d}'.format((num_object - 1)))

#Obtener mascara de objeto con etiqueta 2
object_2 = np.uint8( labels==2 )

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(object_2, cmap='gray')
plot.axis("off")

#Calcular area objecto
area = np.sum(object_2, axis=None)
print('Area cactus:', area)

