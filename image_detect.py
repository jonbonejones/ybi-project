import argparse

import cv2
import numpy as np

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--image", default='C:/Users/Dell/OneDrive/Desktop/Project/data_for_colab/image_dataset/traffic_lights/106.jpg', help="image for prediction")
parser.add_argument("--config", default='C:/Users/Dell/OneDrive/Desktop/Project/data_for_colab/yolov3-tiny-obj.cfg', help="YOLO config path")
parser.add_argument("--weights", default='C:/Users/Dell/OneDrive/Desktop/Project/yolov3-tiny-obj_5000.weights', help="YOLO weights path")
parser.add_argument("--names", default='C:/Users/Dell/OneDrive/Desktop/Project/data_for_colab/obj.names.names', help="class names path")
args = parser.parse_args()

CONF_THRESH, NMS_THRESH = 0.8, 0.8

# Load the network
net = cv2.dnn.readNet(args.config, args.weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores

img = cv2.imread(args.image)


img = cv2.imread("C:/Users/Dell/OneDrive/Desktop/Project/data_for_colab/image_dataset/hospitals/5.jpg")

height, width,channels = img.shape
img = cv2.resize(img, None, fx=1.0,fy=1.0)

#cap = cv2.VideoCapture('C:/Users/91973/Downloads/yolo_object_detection/vid1.mp4')


#cap = cv2.VideoCapture(0)
#ret, img = cap.read()
#height, width, channels = img.shape


blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_outputs = net.forward(output_layers)
#ret, img = cap.read()

class_ids, confidences, b_boxes = [], [], []
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONF_THRESH:
            center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            b_boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))

# Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

# Draw the filtered bounding boxes with their class to the image
with open(args.names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

for index in indices:
    x, y, w, h = b_boxes[index]
    cv2.rectangle(img, (x, y), (x + w, y + h), colors[index], 2)
    cv2.putText(img, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[index], 2)

    

cv2.imshow("image", img)

if cv2.waitKey(10) & 0xFF == ord('q'):
    cap.release()
    cv2.destroyAllWindows()
     
#cv2.waitKey(0)
#cv2.destroyAllWindows()
