import numpy as np
import cv2
import os
import imutils
from pathlib import Path
import random

def crop_coord(img, iter_d, iter_e, iter_o, iter_c):
    '''
    Recortar imagen utilizando tecnicas de preprocesamiento
    '''

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh, None, iterations = iter_d)
    thresh = cv2.erode(thresh, None, iterations = iter_e)
    kernel = np.ones((7, 7), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iter_o)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iter_c)

    edges = cv2.Canny(closed,107,255)
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #Por si hay errores
    if not cnts:
        return 0,0,0,0
    
    c = max(cnts, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)
    

    return x,y,w,h
    
'''Probaré con los dos números para ver qué recortes se realizaron de forma incorrecta y veré la proporción de este nuevo dataset con el original. 
   Si no se han perdido muchos entonces emplearé ese.
   También usaré el de 6 para comparar la precisión entre ambos
'''

def crop_img(img):
    #Estrategia de preprocesamiento estocástica
    width, height = 0,0
    x,y = 0,0
    #Se itera hasta que el valor de iteraciones llegue a 40 o bien se cumpla la otra condición. Esto es para encontrar el mejor crop, que diga el recorte
    iterations = 0
    while width < 235 and height < 225 and iterations < 100:
        dilate = random.randint(4,12)
        erode = random.randint(4,12)
        opened = random.randint(1,4)
        closed = random.randint(1,4)

        x_pos, y_pos, w, h, = crop_coord(img, iter_d = dilate, iter_e = erode, iter_o = opened, iter_c = closed)
        if x_pos == 0 and y_pos == 0 and w == 0 and h == 0:
            continue
        width = w
        height = h
        x = x_pos
        y = y_pos
        iterations+= 1
        if iterations >= 100:
            return img
    print(f"Best crop at: width: {width} and height: {height}. Coordinates: X inicial: {x} X final: {x + width} Y inicial: {y} Y final: {y + height} Numero de iteraciones: {iterations}")
    return img[y:y+height,x:x+width]
    
'''Probaré con los dos números para ver qué recortes se realizaron de forma incorrecta y veré la proporción de este nuevo dataset con el original. 
   Si no se han perdido muchos entonces emplearé ese.
   También usaré el de 6 para comparar la precisión entre ambos
'''

path_training = Path("masoudnickparvar/brain-tumor-mri-dataset/versions/1/Training")
path_testing = Path("masoudnickparvar/brain-tumor-mri-dataset/versions/1/Testing")

subdirs = ['pituitary', 'notumor', 'glioma', 'meningioma']

print(os.listdir(path_training))

path_cropped = Path("datasets/brain-tumor-mri-datasets/Training")
path_cropped_testing = Path("datasets/brain-tumor-mri-datasets/Testing")

NEW_SIZE = 256

if not os.path.exists(path_cropped) and not os.path.exists(path_cropped_testing):
    os.makedirs(path_cropped)
    os.makedirs(path_cropped_testing)

for label in subdirs:
    path_train = os.path.join(path_training,label)
    new_path = None
    
    
    #Si no existe este directorio, entonces se crea
    if not os.path.exists(os.path.join(path_cropped,label)):
        os.mkdir(os.path.join(path_cropped,label))
        new_path = os.path.join(path_cropped,label)

    for image in os.listdir(path_train):
        if os.path.isfile(os.path.join(path_train, image)):
            cropped_image = crop_img(cv2.imread(Path(os.path.join(path_train,image))))
            resized_image = cv2.resize(cropped_image,(NEW_SIZE,NEW_SIZE))
            cv2.imwrite(os.path.join(new_path,image),resized_image) #Guardar imagen recortada


        
for label in subdirs:
    
    path_test = os.path.join(path_testing, label)
    new_path_testing = None
    
    #Si no existe este directorio, entonces se crea
    if not os.path.exists(os.path.join(path_cropped_testing, label)):

        os.mkdir(os.path.join(path_cropped_testing, label))
        new_path_testing = os.path.join(path_cropped_testing, label)
        print(new_path_testing)

    for image in os.listdir(path_test):
        if os.path.isfile(os.path.join(path_test, image)):
            cropped_image = crop_img(cv2.imread(Path(os.path.join(path_test,image))))
            resized_image = cv2.resize(cropped_image,(NEW_SIZE,NEW_SIZE))
            print(os.path.join(path_testing, image))
            cv2.imwrite(os.path.join(new_path_testing,image),resized_image) #Guardar imagen recortada     
        
    