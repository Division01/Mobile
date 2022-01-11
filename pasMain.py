import faceDetector
import maskDetector
import moodDetector
import time
import cv2
import numpy as np

from PIL import Image

emojis = {
    'angry': cv2.imread("./emoji/angry.png"),
    'disgust': cv2.imread("./emoji/disgust.png"),
    'fear': cv2.imread("./emoji/fear.jpeg"),
    'happy': cv2.imread("./emoji/happy.jpeg"),
    'sad': cv2.imread("./emoji/sad.jpeg"),
    'surprise': cv2.imread("./emoji/surprise.jpeg"),
    'neutral': cv2.imread("./emoji/neutral.jpeg"),
    'mask': cv2.imread("./emoji/mask.jpeg")
}


#cap = cv2.VideoCapture(0)


def Filtreur(face_cascade, MODEL, MODEL_MASK, img):
    #print("Appel au filtreur")
    faces = faceDetector.detect_faces(img, face_cascade)
    if faces is not None:
        for item in faces:
            mask = maskDetector.analyze(MODEL_MASK, item[1])

            if not mask:
                # https://data-flair.training/blogs/face-mask-detection-with-python/
                mood = "mask"
            else:
                mood = moodDetector.analyze(MODEL, item[0])
            (x,y,w,h) = item[2]

            emoji = emojis[mood]
            # Check if the rotation has been calculated
            if item[3] is not None:
                
                emoji = Image.fromarray(emoji)
                emoji = np.array(emoji.rotate(int(-item[3])))

            # formatte l'emoji exactement à la taille de la tête détectée
            emoji = faceDetector.process_face(emoji, target_size=(w, h), to_gray=False)
            img[y:y+h, x:x+w, :] = emoji

    #cv2.imshow('img',img)

    return img