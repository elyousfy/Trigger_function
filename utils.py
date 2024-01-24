import cv2
from ultralytics import YOLO
from PIL import Image
import time


def detection_Live_stream(result,frame):

    boxes = result.boxes
    for box in boxes:
        
    # bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

    # put box in cam
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        
            # object details
        org = [x1, y1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(frame, "Face", org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', frame)
       
       

 
        
def detect_and_show_images(face_counter_local,result,delay_ms=100):
   
    # Save the image if a new face has entered the frame
     
    curr_faces = len(result) # get the current number of faces
    if curr_faces > face_counter_local: # compare with the previous number of faces
        
        
        
        im_array = result.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        
        # Generate a unique file name based on the current timestamp
        filename = f"face_{time.strftime('%Y%m%d%H%M%S')}.jpg"
        # Save the image to the file
        im.save(filename)
        print(f"Saved image to {filename}")
    face_counter_local=curr_faces
    return face_counter_local
    
    

    

            
        

