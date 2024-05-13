import cv2
import torch
from ultralytics import YOLO

# give the path of model 
model = YOLO("sample_model\model.onnx")

# if you want to predict on custom date (photos or videos) just replace zero "0" with path of data
cap = cv2.VideoCapture(0)

while cap.isOpened():

    _,frame = cap.read()
    
    results = model.predict(source=frame,show=True,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    #press ESC or 'q' or 'Q' for exit
    if cv2.waitKey(2) & 0xFF == 27 or ord('q') or ord('Q'):
        
        break
    
cap.release()

cv2.destroyAllWindows()