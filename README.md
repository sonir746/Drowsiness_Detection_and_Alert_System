
# Drowsiness-Detection-&-Alert-System-Using-Deep-Learning

This Deep Learning project detect the drowsiness and sleepiness and alert the human being to meet an accident.

We use latest deep learning model which provide the accuracy approximately than 75%-85%. We use YOLOv8 model which can detect multiple features of human face.

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

Clone the project

```bash
  git clone https://github.com/rahulsoni746/Drowsiness_Detection_and_Alert_System.git
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
)](https://github.com/rahulsoni746)



## Feedback

If you have any feedback, please reach out to us at rahulsoni@gmail.com

Or

Report any issue here
<br>
👇👇👇
<br>
[![GitHub](https://img.shields.io/twitter/url?url=https%3A%2F%2Fgithub.com&style=social&logo=GitHub&label=issue&labelColor=grey&color=grey
)](https://github.com/rahulsoni746/Drowsiness_Detection_and_Alert_System/issues)

