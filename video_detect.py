from imutils.video import VideoStream
import numpy as np
from imutils.video import FPS
import imutils
import time

import argparse

import cv2
import numpy as np

RTSP_URL="rtsp://admin:MNVOBD@192.168.2.6:554/H.264"
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--image", default='C:/Users/Dell/OneDrive/Desktop/Project/data_for_colab/image_dataset/schools/66.jpg', help="image for prediction")
parser.add_argument("--config", default='C:/Users/Dell/OneDrive/Desktop/Project/data_for_colab/yolov3-tiny-obj.cfg', help="YOLO config path")
parser.add_argument("--weights", default='C:/Users/Dell/OneDrive/Desktop/Project/yolov3-tiny-obj_5000.weights', help="YOLO weights path")
parser.add_argument("--names", default='C:/Users/Dell/OneDrive/Desktop/Project/data_for_colab/obj.names.names', help="class names path")
args = parser.parse_args()

CONF_THRESH, NMS_THRESH = 0.5, 0.5

# Load the network
net = cv2.dnn.readNet(args.config, args.weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores

#img = cv2.imread(args.image)
classes = []
with open("C:/Users/Dell/OneDrive/Desktop/Project/data_for_colab/obj.names.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
colors= np.random.uniform(0,255,size=(len(classes),3))


#img = cv2.imread("D:/classprogram/detection/data_for_colab/data/number_plate/101.png")
#img = cv2.resize(img, None, fx=0.4, fy=0.4)
#height, width,channels = img.shape
cap = cv2.VideoCapture('C:/Users/Dell/OneDrive/Desktop/Project/VID_20201017_182114.mp4')
#cap=cv2.VideoCapture("TrafficVideo.mp4") #0 for 1st webcam
#cap=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0

while True:
    _,frame= cap.read() # 
    frame_id+=1
    
    height,width,channels = frame.shape
    #detecting objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(120,120),(0,0,0),True,crop=False) #reduce 416 to 320    

        
    net.setInput(blob)
    #layer_outputs = net.forward(output_layers)
    outs = net.forward(output_layers)
    #print(outs[1])


    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #onject detected
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected
                print(class_id)
            

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)


    for i in range(len(boxes)):
        global label
        #if i in indexes:
            #print(i)
            
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
                    
        print(label)
                    
                    
        confidence= confidences[i]
        color = colors[class_ids[i]]
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
            

    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
    
    cv2.imshow("Image",frame)
    key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
    
    if key == 27: #esc key stops the process
        break;
    
cap.release()    
cv2.destroyAllWindows()
