from kivy.app import App 
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
import cv2
import numpy as np
import moodDetector
import pasMain
from keras.models import load_model 

class SelfieCameraApp(App):

    def build(self):
        self.bool_filtr = False
        self.camera_obj = Camera()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.MODEL = moodDetector.loadModel()
        self.MODEL_MASK = load_model("./data/mask_detector.model")

        # button 
        button_obj = Button(text = "Click here")
        button_obj.size_hint = (.5, .2)
        button_obj.pos_hint = {'x' : .25, 'y' : .25}
        button_obj.bind(on_press= self.filtray)

        #Layout
        layout = BoxLayout()
        layout.add_widget(self.camera_obj)
        layout.add_widget(button_obj)
        
        return layout

    def _on_config_change(self, *largs):
        return super()._on_config_change(*largs)
        return super().on_resume()        print("in")
        self.camera_obj.texture.add_reload_observer(self.test)

        return super().on_resume()


    def test(self, texture):
        print("in")

    def filtray(self, *args):
        if self.bool_filtr == True :
            print("Là ça devrait plus filtrer")
            self.bool_filtr = False

        elif self.bool_filtr == False :
            print("Maintenant ça filtre je crois")
            self.bool_filtr = True
        print(self.camera_obj.texture)
        #cv2.imshow('img',self.camera_obj.texture)


if __name__ == '__main__':
    a = SelfieCameraApp()
    a.run()
    
    print(a.bool_filtr)
    if SelfieCameraApp.bool_filtr :
        print(SelfieCameraApp.bool_filtr)
        SelfieCameraApp.camera_obj = pasMain.Filtreur(SelfieCameraApp.face_cascade, SelfieCameraApp.MODEL, SelfieCameraApp.MODEL_MASK, SelfieCameraApp.camera_obj)
    