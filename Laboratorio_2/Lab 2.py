#LAB 2 - Robótica 2021-1

#Cargar librerias a usar
import cv2
import numpy as np
import matplotlib.pyplot as plot

#%% Connected Components

#Cargar imagen y pasarla a escala de grises (GRAY)
img = cv2.imread("Mix_v4.jpg") 
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
img_gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_RGB)
plot.axis("off")

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_gray, cmap='gray')
plot.axis("off")

#Se aplica un filtro gaussiano  (PROBAR CON Y SIN FILTRO)
img_blur = cv2.GaussianBlur(img_gray, (11,11), 0)

#Crear mascara mediante funcion cv2.inRange()
mask2 = cv2.inRange(img_blur, 40, 255) # Los valores son o 0 o 255.
mask2 = mask2/255  # Normaliza valores para que los valores sean o cero o uno. 
                   # --> mask2.dtype = float64
mask2 = np.uint8(mask2)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(mask2, cmap='gray')
plot.axis("off")

#Etiquetar mediante connected components
num_object, labels = cv2.connectedComponents(mask2)

print('Objetos identificadas:',(num_object - 1))

#Obtener mascara de objeto con etiqueta 2
object_2 = (labels==2) # --> object_2.dtype = bool

#Obtener mascara de objeto con etiqueta 2
object_2 = np.uint8( object_2 ) # --> object_2.dtype = uint8


#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(object_2, cmap='gray')
plot.axis("off")


#%% Limpiar mascara de forma manual

#Crear imagen vacia
mask_final = np.zeros_like(mask2)

#Eliminar manualmente las mascaras entregadas por la funcion cv2.connectedComponents 
#que tienen menos de 50 pixeles blancos
for i in range(1,num_object):
     object_i = np.uint8( labels==i )
     area_i = np.sum(object_i)
     if area_i > 120:
         mask_final = mask_final + object_i

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(mask_final, cmap='gray')
plot.axis("off")

#Etiquetar mediante connected components
num_object, labels = cv2.connectedComponents(mask_final)

print('Objetos identificadas:',(num_object - 1))

#%%Test cantidad pixeles c

#Obtener mascara de objeto con etiqueta 2
object_2 = np.uint8( labels==1 )

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(object_2, cmap='gray')
plot.axis("off")

print(np.sum(object_2))

#%% Crear mascara según el tipo de objeto

#Crear matrices vacias para rellenar con los distintos objetos
mike = np.zeros_like(mask_final)
champions = np.zeros_like(mask_final)
la_garra = np.zeros_like(mask_final)

#Agregar elementos segun la cantidad de pixeles que componen cada objeto
for i in range(1, num_object):
    object_i = np.uint8( labels == i)
    area_i = np.sum(object_i)
    if (area_i > 9000 and area_i < 10000):
        mike = mike + object_i
    if (area_i > 4000 and area_i < 5000):
        la_garra += object_i
    if area_i > 30000:
        champions += object_i
        
#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(mike, cmap='gray')
plot.axis("off")

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(la_garra, cmap='gray')
plot.axis("off")

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(champions, cmap='gray')
plot.axis("off")

#Multiplicar máscara con imagen original
img_mask = np.zeros_like(img_RGB)

for c in range(img_mask.shape[2]):
  img_mask[:,:,c] = np.multiply(img_RGB[:,:,c], mike)
  
 #Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_mask)
plot.axis("off")

