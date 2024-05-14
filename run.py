import cv2
import pyttsx3
import time as t
import torch
from ultralytics import YOLO
# give the path of model 
model = YOLO('Source\Model\model.pt') 
# if you want to predict on custom date (photos or videos) just replace zero "0" with path of data
cap = cv2.VideoCapture(0) 
count=0
while True: 
    success, frame = cap.read()
    frame=cv2.flip(frame,1)
    # remove classes for more outputs
    results = model.predict(source=frame,show=True,save=True,classes=[0,1,2],device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) 
    names = model.names
    check=[]
    for r in results:
        for c in r.boxes.cls:
            check.append(names[int(c)])
    if "face" in check:
        if "close eyes" in check:
            if count==0:
                start_time = float(t.time())
                count=1
            end_time=float(t.time())
            run_time=end_time-start_time      
            # setting alert massage for 1 second you can change it
            if run_time>1:                
                print("alert")
                alert = pyttsx3.init()
                alert.say("Wake up")
                alert.runAndWait()
                alert= None
                print(run_time)
        elif "open eyes" in check:
            alert= None
            print("no alarm")
            count=0
    #press ESC for exit
    if cv2.waitKey(2) & 0xFF == 27: 
        break
cap.release()
cv2.destroyAllWindows()
