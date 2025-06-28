import cv2 as cv
import os
from datetime import datetime

from cv2.data import haarcascades
from tensorflow import timestamp

# Creation un dossier pour sauvegarder les images
save_dir ="captured_faces"
os.makedirs(save_dir,exist_ok=True)

# Charger le classificateur Haar pour les visages
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# OUvrir le webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir le webcam")
    exit()
while True:
# lire le flux video
    ret,frame = cap.read()
    if not ret:
        print("Erreur de lecture de video")
        break;
    # COnversion en niveaux gris
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    # Detection de visages
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    # Affichage au nombre des visages detectes
    count = len(faces)
    cv.putText(frame,f"Visages detectes : {count}",(10,30),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    # Dessiner les rectangles
    for(x,y,w,h) in faces :
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    # Enregistrer un image >=1
    if count > 0 :
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/face _ {timestamp}.png"
        cv.imwrite(filename,frame)
        print(f"INFO | Image sauuvegarde {filename}")

        # Afficher le flux
        cv.imshow("Webcam  - Detection visage ",frame)

        # Quitter si user appuie sur 'q'
        if cv.waitKey(1)&0xFF ==ord('q'):
            break
# Liberer les ressources

cap.release()
cv.destroyAllWindows()
