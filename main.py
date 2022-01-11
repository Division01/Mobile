from kivymd.app import MDApp 
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivymd.uix.label import MDLabel
import cv2
import moodDetector
import pasMain
from keras.models import load_model 

class MainApp(MDApp):

    def build(self):
        self.bool_filtr = False

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.MODEL = moodDetector.loadModel()
        self.MODEL_MASK = load_model("./data/mask_detector.model")

        #Layout
        layout = MDBoxLayout()
        self.image = Image()
        self.label = MDLabel()
        layout.add_widget(self.image)
        layout.add_widget(self.label)
        self.filter_button = MDRaisedButton(
            text = "Filter",
            pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None))
        self.filter_button.bind(on_press=self.filtray)
        layout.add_widget(self.filter_button)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0/30.0)
        return layout

    def load_video(self, *args):
        if self.bool_filtr : 
            ret, frame = self.capture.read()
            # Frame initialize
            self.image_frame = frame
            self.image_frame = pasMain.Filtreur(self.face_cascade, self.MODEL, self.MODEL_MASK, frame)
            buffer = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture
        else : 
            ret, frame = self.capture.read()
            # Frame initialize
            self.image_frame = frame
            buffer = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def filtray(self, *args):
        if self.bool_filtr == True :
            print("Là ça devrait plus filtrer")
            self.bool_filtr = False

        elif self.bool_filtr == False :
            print("Maintenant ça filtre je crois")
            self.bool_filtr = True


if __name__ == '__main__':
    MainApp().run()