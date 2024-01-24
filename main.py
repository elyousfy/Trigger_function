from ultralytics import YOLO
import cv2
from utils import *


global face_counter_global

face_counter_global=0
   

if __name__=="__main__":
        
    model = YOLO('yolov8n-face-lindevs.pt')

    cam=cv2.VideoCapture(0) #change the 0 depending on source of video stream
   
    while True:
        
        results=detect_face(cam,model)
        
        for result in results:
                
                # to display a video stream with real time detection uncomment this, and comment detect_and_show_images:
                #detection_Live_stream(result,frame)
                
                #this function continuously monitors the webcam
                trigger_function(face_counter_global,result,show=False,save=False)
        
                
                curr_faces = len(result)
                if curr_faces > face_counter_global: #if number of faces increases, enter the next part
                
                        results=detect_face(cam,model,delay_s=0.5) #invoke YOLO to detect new face 
                        try:
                            result = results[0]
                            
                            
                        except:
                            pass
                        
                 #update the number of faces in counter
                face_counter_global=trigger_function(face_counter_global,result)
                

        
    # If the input is the q key, exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the webcam and close all the windows
    cam.release()
    cv2.destroyAllWindows()