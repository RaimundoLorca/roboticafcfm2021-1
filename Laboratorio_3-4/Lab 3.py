#LAB 3 - Robótica 2021-1

#Cargar librerias a usar
import cv2
import numpy as np
import matplotlib.pyplot as plot

#%% Introducción al problema y cargar imagen

#La empresa de manufactura NERV S.A, dedicada al mecanizado y corte de piezas metálicas, 
#ha decidido automatizar su sistema de control de calidad con el fin de acelerar la producción y 
#reducir el número de piezas defectuosas en la línea de ensamblaje. Por esta razón, se le ha 
#encargado a usted desarrollar un sistema de visión computacional que permita identificar 
#aquellas piezas cuyas medidas escapen de las tolerancias especificadas.

#Cargar imagen
img = cv2.imread('Imagenes//batch_1.png')

#Cambiar espacio de colores (BGR A RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_rgb)
plot.axis('off')

#%% Segmentar imagen (1/2)

#Segementar/binarizar respecto al fondo
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

#Extraer canal Hue para plotear un histograma
H = img_hsv[:, :, 0]
img_flat = H.flatten()

#Plotear histograma con Matplotlib
plot.figure()
plot.hist(img_flat, 100)

#%% Segmentar imagen (2/2)

#Definir hsv thresholds
lower_hsv = np.array( [90, 100, 100] )
upper_hsv = np.array( [120, 255, 255] )

#Crear mascara mediante funcion cv2.inRange()
mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
mask = 255 - mask

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(mask, cmap = 'gray')
plot.axis('off')

#%% Aplicar ConnectedComponents

#Etiquetar mediante connected components
num_labels, labels = cv2.connectedComponents(mask)

#Visualizar pieza número 1
pieza_1 = np.uint8( (labels==1)*255 )

plot.figure()
plot.imshow(pieza_1, cmap='gray')
plot.axis('off')

#%% Detectar bordes mediante Sobel

#Sobel (cv2.Sobel)
sobel_x = cv2.Sobel(pieza_1, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(pieza_1, cv2.CV_64F, 0, 1, ksize=5)

#Calcular valor absoluto valores de la matriz
sobel_x = np.abs(sobel_x)
sobel_y = np.abs(sobel_y)

#Unir ambas matrices
edge_sobel = cv2.bitwise_or(sobel_x, sobel_y)
edge_sobel = np.uint8(edge_sobel)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(np.uint8(sobel_x), cmap='gray')
plot.axis('off')

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(np.uint8(sobel_y), cmap='gray')
plot.axis('off')

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(edge_sobel, cmap='gray')
plot.axis('off')

#%% Detectar bordes mediante Canny

#Canny (cv2.Canny)
edge_canny = cv2.Canny(pieza_1, 100, 200)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(edge_canny, cmap='gray')
plot.axis('off')

#%% Detectar rectas

# Utilizar HoughLines sobre la imagen
lines = cv2.HoughLines(edge_canny, rho=1.0, theta=np.pi/360, threshold=60)

img_lines = np.copy(img_rgb)
   
for line in lines:
    
    #Obtener parámetros (rho, theta)
    rho, theta = line[0]
    
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
        
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
        
    color=(255, 0, 255)
    thickness = 2
    img_lines = cv2.line(img_lines, (x1,y1), (x2,y2), color, thickness)
    
#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_lines)
plot.axis('off')

print('Número de lineas detectadas:' , len(lines))

#%% Filtrar lineas

#Reestructurar lines en dos np.array
rho = np.zeros( len(lines) )
theta = np.zeros( len(lines) )
  
for i in range(len(lines)):
    r, t = lines[i][0]
    if r<0:
        theta[i] = abs( t - np.pi )
    else:
        theta[i] = t
    rho[i] = abs(r)

#Plotear espacio de Hough
plot.figure()
plot.plot(rho,theta, '.')
plot.xlabel('rho')
plot.ylabel('theta')
plot.title('Hough Space')
plot.show()

#Inicializar una lista vacía
rectas = list()
for r, t in zip(rho, theta):
    #Encontrar los index en rho y theta que cumplan la condición
    index = np.where( (np.abs(rho - r) < 30) & (np.abs(theta - t) < np.pi/6) )
        
    rho_index = []
    theta_index = []
    for i in index:
        rho_index.append(rho[i])
        theta_index.append(theta[i])
    
    r = np.mean(rho_index)
    t = np.mean(theta_index)

    #Si (rho_mean, theta_mean) no se ha agregado
    if (r, t) not in rectas:
        rectas.append( (r, t) )

img_lines2 = np.copy(img_rgb)

for line in rectas:
    
    #Obtener parámetros (rho, theta)
    rho, theta = line
    
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
        
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
        
    color=(255, 0, 255)
    thickness = 2
    img_lines2 = cv2.line(img_lines2, (x1,y1), (x2,y2), color, thickness)
    
#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_lines2)
plot.axis('off')


print('Número de lineas detectadas:' , len(rectas))

#%% Obtener angulo de corte y posiciones perforaciones

#Obtener ángulo de corte
cut_angle = []
line_plot = []

#Para cada una de las líneas detectadas
for recta in rectas:
  rho, theta = recta

  #Si el ángulo es cercano a 30°
  if abs(theta*180/np.pi - 30.0) < 5.0:

    #Agregar ángulo a la lista (en grados)
    cut_angle.append( theta*(180/np.pi) )
    line_plot.append(recta)

#Promediar resultado y obtener estimación de ángulo
cut_angle = 90 - np.mean(cut_angle)
print('Ángulo de corte identificado: {:f} grados.'.format(cut_angle))

#Transformada hough circulos (cv2.HoughCircles)
circles = cv2.HoughCircles(edge_canny, cv2.HOUGH_GRADIENT, dp=1, minDist=9, param1=50, param2=25)

#Separar perforaciones
perf_izq = []
perf_der = []
for c in circles[0]:

    #Si es de las perforaciones menores (radio < 15px)
    if c[2] < 15:
        perf_izq.append( list(c) )

    #Si es la perforación mayor   
    else:
        perf_der = c

print('\nPerforaciones izquierda', perf_izq)
print('\nPerforación derecha', perf_der)

#%% Plotear rectas y circulos detectados

img_lines3 = np.copy(img_rgb)

for line in line_plot:
    
    #Obtener parámetros (rho, theta)
    rho, theta = line
    
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
        
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
        
    color=(255, 0, 0)
    thickness = 3
    img_lines2 = cv2.line(img_lines3, (x1,y1), (x2,y2), color, thickness)
    
#Para cada circulo detectado
for circle in perf_izq:
  centro = (int(circle[0]), int(circle[1]))
  radio = int(circle[2])
  color = (0, 255, 0)
  thickness = 3
  img = cv2.circle(img_lines3, centro, radio, color, thickness)

centro = (int(perf_der[0]), int(perf_der[1]))
radio = int(circle[2])
color = (255, 0, 255)
thickness = 3
img = cv2.circle(img_lines3, centro, radio, color, thickness)

#Plotear imagen con Matplotlib
plot.figure()
plot.imshow(img_lines3)
plot.axis('off')

#%% Definir funciones que detecte las piezas defectuosas

def detect_faulty_parts(img):
    """
    -> Image (np.array)
    Detecta en la imagen aquellas piezas que no cumplan con las especificaciones de diseño.
    
    :param np.array img:
        imagen en RGB que contiene las piezas a analizar.
    
    :returns:
        imagen con las piezas defectuosas identificadas con rojo.
    """
    # segementar/binarizar respecto al background (parte 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # definir hsv thresholds
    lower_hsv = np.array( [90, 100, 100] )
    upper_hsv = np.array( [120, 255, 255] )

    # segmentar e invertir
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = 255 - mask

    # inicializar imágen que contendrá las máscaras de las piezas malas
    faulty_parts = np.zeros_like(mask)

    # etiquetar mediante connected components (parte 1)
    num_parts, labels = cv2.connectedComponents(mask)

    # procesar cada una de las piezas en la imagen
    for i in range(1, num_parts):

      # separar pieza
      part_img = (labels == i)*255
      part_img = np.uint8(part_img)

      # obtener bordes/edges (parte 2)
      edges = cv2.Canny(part_img, 100, 255)

      # aplicar transformada de Hough de rectas y obtner ángulo de corte (parte 3)
      lines = cv2.HoughLines(edges, rho=1.0, theta=np.pi/180, threshold=90)

      # obtener ángulo de corte
      cut_angle = []
      # para cada una de las líneas detectadas
      for line in lines:
        rho, theta = line[0]

        # si el ángulo es cercano a 30°
        if abs(theta*180/np.pi - 30.0) < 5.0:

          # agregar ángulo a la lista (en grados)
          cut_angle.append( theta*(180/np.pi) )

      # promediar resultado y obtener estimación de ángulo
      cut_angle = 90 - np.mean(cut_angle)

      # aplicar transformada de Hough de circulos y obtener dimensiones perfs. (parte 3)
      circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=9, param1=50, param2=27)

      # separar perforaciones
      perf_izq = []
      perf_der = []
      for c in circles[0]:

          # si es de las perforaciones menores (radio < 15px)
          if c[2] < 15:
              perf_izq.append( list(c) )

          # si es la perforación mayor   
          else:
              perf_der = c

      # checkear cumplimiento especificaciones
      # (ejemplo, suponer que piezas están correctas)
      VALID_PART = True
        
      try:
        # si ángulo de corte está fuera de la tolerancia
        assert abs( cut_angle - 60 ) < 60*0.05

        assert abs( abs(perf_izq[0][0] - perf_der[0]) - 127 ) < 127*0.05
        assert abs( abs(perf_izq[1][0] - perf_der[0]) - 127 ) < 127*0.05

        assert abs( abs(perf_izq[0][1] - perf_der[1]) - 42 ) < 43*0.1
        assert abs( abs(perf_izq[1][1] - perf_der[1]) - 42 ) < 43*0.1
      
      except AssertionError:
        # pieza está mala
        VALID_PART = False
      
      # si la pieza está defectuosa
      if not(VALID_PART):
        faulty_parts = faulty_parts + part_img

    # aplicar máscara roja (faulty_parts) sobre la imagen original
    faulty_parts = faulty_parts/255
    res = np.zeros_like(img)
    res[:,:,0] = np.multiply(img[:, :, 0], 1 - faulty_parts) + faulty_parts*255
    res[:,:,1] = np.multiply(img[:, :, 1], 1 - faulty_parts)
    res[:,:,2] = np.multiply(img[:, :, 2], 1 - faulty_parts)

    return res

#%% Probar la función
test_batch = cv2.imread('Imagenes//batch_1.png')
test_batch = cv2.cvtColor(test_batch, cv2.COLOR_BGR2RGB)

fig = plot.figure(figsize=(10, 10))
plot.imshow(test_batch)
plot.axis('off')

ret = detect_faulty_parts(test_batch)

fig = plot.figure(figsize=(10, 10))
plot.imshow(ret)
plot.axis('off')

#%% Descartar piezas con el logo viejo

def detect_faulty_parts_2(img):
    """
    -> Image (np.array)
    Detecta en la imagen aquellas piezas que no cumplan con las especificaciones de diseño.
    
    :param np.array img:
        imagen en RGB que contiene las piezas a analizar.
    
    :returns:
        imagen con las piezas defectuosas identificadas con rojo.
    """

    # segementar/binarizar respecto al background (parte 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # definir hsv thresholds
    lower_hsv = np.array( [90, 100, 100] )
    upper_hsv = np.array( [120, 255, 255] )

    # segmentar e invertir
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = 255 - mask

    # inicializar imágen que contendrá las máscaras de las piezas malas
    faulty_parts = np.zeros_like(mask)

    # etiquetar mediante connected components (parte 1)
    num_parts, labels = cv2.connectedComponents(mask)

    # segmentar logos rojos desde la imagen original
    logos = cv2.inRange(hsv, np.array( [0, 100, 100] ), np.array( [10, 255, 255] ))

    # inicializar sift
    sift = cv2.xfeatures2d.SIFT_create()

    # procesar cada una de las piezas en la imagen
    for i in range(1, num_parts):

      # aislar pieza y logo correspondiente
      part_img = np.uint8( (labels==i)*255 )
      part_logo = np.multiply(logos, part_img)*255

      # checkear si el logo es del modelo antiguo

      # obtener descriptores referencia mediante sift.detectAndCompute
      ref = cv2.imread('Imagenes//old_model_logo.png')
      ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
      kp_ref, des_ref = sift.detectAndCompute(ref, mask=None)

      # obtener descriptores del logo mediante sift.detectAndCompute
      kp_logo, des_logo = sift.detectAndCompute(part_logo, mask=None)

      # realizar brute force matching entre los descriptores
      BFMatcher = cv2.BFMatcher()
      matches = BFMatcher.knnMatch(des_logo, des_ref, k=2)
      
      # filtrar buenos matches
      good_matches = []
      for match_1, match_2 in matches:
        if match_1.distance < 0.75*match_2.distance:
          good_matches.append(match_1)

      # si no hay matches (no es logo antiguo
      if len(good_matches) < 20:
          continue

      # obtener bordes/edges (parte 2)
      edges = cv2.Canny(part_img, 100, 255)

      # aplicar transformada de Hough de rectas y obtner ángulo de corte (parte 3)
      lines = cv2.HoughLines(edges, rho=1.0, theta=np.pi/180, threshold=90)

      # obtener ángulo de corte
      cut_angle = []
      # para cada una de las líneas detectadas
      for line in lines:
        rho, theta = line[0]

        # si el ángulo es cercano a 30°
        if abs(theta*180/np.pi - 30.0) < 5.0:

          # agregar ángulo a la lista (en grados)
          cut_angle.append( theta*(180/np.pi) )

      # promediar resultado y obtener estimación de ángulo
      cut_angle = 90 - np.mean(cut_angle)

      # aplicar transformada de Hough de circulos y obtener dimensiones perfs. (parte 3)
      circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=9, param1=50, param2=27)

      # separar perforaciones
      perf_izq = []
      perf_der = []
      for c in circles[0]:

          # si es de las perforaciones menores (radio < 15px)
          if c[2] < 15:
              perf_izq.append( list(c) )

          # si es la perforación mayor   
          else:
              perf_der = c

      # checkear cumplimiento especificaciones
      # (ejemplo, suponer que piezas están correctas)
      VALID_PART = True
        
      try:
        # si ángulo de corte está fuera de la tolerancia
        assert abs( cut_angle - 60 ) < 60*0.05

        assert abs( abs(perf_izq[0][0] - perf_der[0]) - 127 ) < 127*0.05
        assert abs( abs(perf_izq[1][0] - perf_der[0]) - 127 ) < 127*0.05

        assert abs( abs(perf_izq[0][1] - perf_der[1]) - 44 ) < 44*0.1
        assert abs( abs(perf_izq[1][1] - perf_der[1]) - 44 ) < 44*0.1
      
      except AssertionError:
        # pieza está mala
        VALID_PART = False
      
      # si la pieza está defectuosa
      if not(VALID_PART):
        faulty_parts = faulty_parts + part_img

    # aplicar máscara roja (faulty_parts) sobre la imagen original
    faulty_parts = faulty_parts/255
    res = np.zeros_like(img)
    res[:,:,0] = np.multiply(img[:, :, 0], 1 - faulty_parts) + faulty_parts*255
    res[:,:,1] = np.multiply(img[:, :, 1], 1 - faulty_parts)
    res[:,:,2] = np.multiply(img[:, :, 2], 1 - faulty_parts)

    return res



#%% Probar la función
test_batch = cv2.imread('Imagenes//batch_2.png')
test_batch = cv2.cvtColor(test_batch, cv2.COLOR_BGR2RGB)

fig = plot.figure(figsize=(10, 10))
plot.imshow(test_batch)
plot.axis('off')

ret = detect_faulty_parts_2(test_batch)

fig = plot.figure(figsize=(10, 10))
plot.imshow(ret)
plot.axis('off')