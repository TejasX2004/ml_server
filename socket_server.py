import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
from collections import deque
from PIL import Image
import base64
import io

# Initialize the YOLOv10 model with the pre-trained weights
yolo_model = YOLO("weights/yolov10n.pt")

# Load the PyTorch model
class GenderModel(nn.Module):
    def __init__(self):
        super(GenderModel, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 2)

    def forward(self, x):
        return self.model(x)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gender_model = GenderModel()
gender_model.load_state_dict(torch.load("best_model.pth"), strict=False)
gender_model = gender_model.to(device)
gender_model.eval()

cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                  "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush", "weapon"]

gender_classes = ['male', 'female']

# Deque to store last N predictions for smoothing
prediction_deque = deque(maxlen=5)

# Function to stabilize gender prediction
def stabilize_prediction(predictions):
    if len(predictions) > 0:
        return max(set(predictions), key=predictions.count)
    return "unknown"

# Preprocessing transformation for input images
preprocess = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def process_frame(data):
    stream_id = data.get('streamId', 'unknown')
    frame_data = data['frame']

    # Decode base64 image
    header, encoded = frame_data.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_data))
    
    # Convert the PIL image to OpenCV format
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # YOLO object detection
    results = yolo_model.predict(frame, conf=0.25, device=device)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            className = cocoClassNames[cls]

            if className in ["person", "weapon"]:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                if className == "person":
                    person_crop = frame[y1:y2, x1:x2]
                    person_crop = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                    person_crop = preprocess(person_crop).unsqueeze(0).to(device)

                    with torch.no_grad():
                        gender_conf = gender_model(person_crop)[0]
                        gender_idx = torch.argmax(gender_conf).item()
                        gender_label = gender_classes[gender_idx]

                    prediction_deque.append(gender_label)
                    stabilized_gender = stabilize_prediction(list(prediction_deque))
                    
                    if stabilized_gender == "male":
                        color = (255, 0, 0)
                    else:
                        color = (255, 20, 147)
                        
                    label = stabilized_gender
                else:
                    label = className
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                conf = math.ceil(box.conf[0] * 100) / 100
                label = f"{label}: {conf}"
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, color, -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    processed_image_data = base64.b64encode(buffer).decode('utf-8')
    processed_image_data = f"data:image/jpeg;base64,{processed_image_data}"

    return processed_image_data, stream_id
