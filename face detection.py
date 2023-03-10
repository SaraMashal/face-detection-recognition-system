import os 
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Saved Pictures")

#recognizer = cv2.face_LBPHFaceRecognizer.create()
cascPath = 'C:/Users/sara/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("jpeg"):
            path =  os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            print(label, path)
            if not label in label_ids:

                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            print (label_ids)
            
            PIL_image = Image.open(path).convert("L")
            image_array = np.array(PIL_image, "uint8")
            print (image_array)
            faces = faceCascade.detectMultiScale(
        image_array,
        flags=cv2.FONT_HERSHEY_SIMPLEX
    )
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

                
#print (y_labels)
#print (x_train)
with open ("labels.pickle", 'wb' ) as f:
    pickle.dump(label_ids, f)
    
#recognizer.train(x_train, np.array(y_labels))
#recognizer.save("trainner.yml")
                
                

