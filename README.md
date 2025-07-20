
# Drowsiness-Detection-&-Alert-System-Using-Deep-Learning

This Deep Learning project detects driver drowsiness and sleepiness, helping prevent accidents by alerting the driver.

We trained a custom YOLOv8 model on over 7,000 images to detect multiple facial features. The system triggers an alert when the driver’s eyes remain closed for more than one second and stops alerting once they are open again.

## **DEMO VIDEO**
### <strong>NOTE: <em>UnMute blow video if muted</em></strong>

https://github.com/sonir746/Drowsiness_Detection_and_Alert_System/assets/169639197/17d75760-a0d8-4437-878f-4a88cf71dbd0

<br>

## Packages Used

| **Package** | **Version** |
| ----------- | ----------- |
| Python      | 3.8         |
| OpenCV      | 4.8.1       |
| Yolo        | 8.0.2       |
| PyTorch     | 2.1.2       |
| LabelImg    | 1.8.6       |
| Pyttsx3     | 2.90        |


## Run Locally

### NOTE: Orignal Source folder uploaded [here](https://drive.google.com/drive/folders/1tMfPCD_p2vVAQL-wzFfSUY-NJERg_ypK?usp=sharing)

https://github.com/user-attachments/assets/aa412ba4-39ac-4b33-837b-dc03c2ff63bc

<br>

Clone the project

```bash
  git clone https://github.com/sonir746/Drowsiness_Detection_and_Alert_System.git
```

Go to the project directory

```bash
  cd <directory_path>
```

Install dependencies

```bash
  pip install -r requirements.txt
```
## Training

Use labelimg library to label the images to create a label data set and train our model to 

<img src="Source\Images\Labeling.png" alt="loading..." style="width:800px;"/>

### Traning Data
<img src="Source\Images\TraningData.png" alt="loading..." style="width:800px;"/>

## Traning Rusult

<img src="Source\Images\results.png" alt="loading..." style="width:800px;"/>

## Testing
1.	After that we provide the camera access or input image to model.

2.	Model will extract the feature from the model and perform conditional  operation on the given feature.

3.	After that it will either alert or continue the execution till any interruption.

## Source Code

```Python

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

```

## Input Image
<img src="Source\Images\input.jpg" alt="loading..." style="width:600px;"/>

## Predicted Image
<img src="Source\Images\output.jpg" alt="loading..." style="width:600px;"/>



## Auther

👨🏻‍💼RAHUL SONI

[![linkedin](https://img.shields.io/twitter/url?url=https%3A%2F%2Fwww.linkedin.com&style=social&logo=Linkedin&logoColor=White&label=Linkedin&labelColor=blue&color=blue&cacheSeconds=3600
)](https://www.linkedin.com/in/rahul-soni-004861227)
[![GitHub](https://img.shields.io/twitter/url?url=https%3A%2F%2Fgithub.com%2F&style=social&logo=GitHub&logoColor=Black&label=GitHub&labelColor=abcdef&color=fedcba&cacheSeconds=3600
)](https://github.com/sonir746)



## Feedback

If you have any feedback, please reach out to us at rahulsoni7469@gmail.com

Or

Report any issue here
<br>
👇👇👇
<br>
[![GitHub](https://img.shields.io/twitter/url?url=https%3A%2F%2Fgithub.com&style=social&logo=GitHub&label=issue&labelColor=grey&color=grey
)](https://github.com/sonir746/Drowsiness_Detection_and_Alert_System/issues)

