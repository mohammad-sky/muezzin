__author__ = 'bunkus'
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from module import *
import cv2

class CamApp(App):

    def build(self):
        self.img1=Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def update(self, dt):
        # display image from cam in opencv window
        
        pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

        # ret, frame = self.capture.read()
        
        while self.capture.isOpened():
            
            # Read a frame.
            ok, frame = self.capture.read()
            
            # Check if frame is not read properly.
            if not ok:
                
                # Continue to the next iteration to read the next frame and ignore the empty camera frame.
                continue
            
            # Flip the frame horizontally for natural (selfie-view) visualization.
            frame = cv2.flip(frame, 1)
            
            # Get the width and height of the frame
            frame_height, frame_width, _ =  frame.shape
            
            # Resize the frame while keeping the aspect ratio.
            frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
            
            # Perform Pose landmark detection.
            frame, landmarks = detectPose(frame, pose_video, display=False)
            
            # Check if the landmarks are detected.
            if landmarks:
                
                # Perform the Pose Classification.
                frame, _ = classifyPose(landmarks, frame, display=False)
            
            # Display the frame.
            cv2.imshow('Pose Classification', frame)
            
            # Wait until a key is pressed.
            # Retreive the ASCII code of the key pressed
            k = cv2.waitKey(1) & 0xFF
            
            # Check if 'ESC' is pressed.
            if(k == 27):
                
                # Break the loop.
                break

        # Release the VideoCapture object and close the windows.
        self.capture.release()
        cv2.destroyAllWindows()
                
        
        
        cv2.imshow("CV2 Image", frame)
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer. 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.img1.texture = texture1

if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()